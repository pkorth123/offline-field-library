"""Validated knowledge-pack manifests and local, deduplicated imports."""

from __future__ import annotations

import json
import mimetypes
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .database import LibraryDatabase
from .documents import SUPPORTED_EXTENSIONS, file_sha256
from .ingestion import Embedder, ingest_paths

MANIFEST_VERSION = 1
AUTHORITIES = {
    "official",
    "government",
    "professional",
    "community",
    "anecdotal",
    "derived",
    "unknown",
}
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")


@dataclass(frozen=True)
class PackSource:
    key: str
    title: str
    file: str | None = None
    url: str | None = None
    publisher: str | None = None
    author: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None
    source_class: str = "other"
    authority: str = "unknown"
    applicability: list[str] = field(default_factory=list)
    license_note: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class PackManifest:
    slug: str
    title: str
    description: str
    brief: str
    sources: list[PackSource]
    report_file: str | None = None
    schema_version: int = MANIFEST_VERSION


@dataclass
class PackImportReport:
    slug: str
    total_sources: int = 0
    indexed: int = 0
    preserved: int = 0
    remote_only: int = 0
    deduplicated: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    pack_path: Path | None = None


def load_manifest(pack_dir: Path) -> PackManifest:
    root = pack_dir.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"manifest not found: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"could not read manifest: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("manifest root must be a JSON object")
    version = raw.get("schema_version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"unsupported manifest schema_version {version!r}; expected {MANIFEST_VERSION}"
        )

    slug = _required_string(raw, "slug")
    if not SLUG_PATTERN.fullmatch(slug):
        raise ValueError("slug must be 3-64 lowercase letters, numbers, or hyphens")
    source_items = raw.get("sources")
    if not isinstance(source_items, list) or not source_items:
        raise ValueError("sources must be a non-empty array")

    sources: list[PackSource] = []
    seen_keys: set[str] = set()
    for position, item in enumerate(source_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"source #{position} must be a JSON object")
        source = _parse_source(item, position)
        if source.key in seen_keys:
            raise ValueError(f"duplicate source key: {source.key}")
        seen_keys.add(source.key)
        if source.file:
            _resolve_pack_file(root, source.file)
        sources.append(source)

    report_file = _optional_string(raw, "report_file")
    if report_file:
        _resolve_pack_file(root, report_file)
    return PackManifest(
        schema_version=version,
        slug=slug,
        title=_required_string(raw, "title"),
        description=_optional_string(raw, "description") or "",
        brief=_optional_string(raw, "brief") or "",
        sources=sources,
        report_file=report_file,
    )


def import_pack(
    database: LibraryDatabase,
    pack_dir: Path,
    embedder: Embedder | None,
    embedding_model: str | None,
) -> PackImportReport:
    root = pack_dir.expanduser().resolve()
    manifest = load_manifest(root)
    database.initialize()
    collection_id = database.upsert_collection(
        manifest.slug, manifest.title, manifest.description, manifest.brief
    )
    report = PackImportReport(slug=manifest.slug, total_sources=len(manifest.sources))
    imported_sources: list[dict[str, Any]] = []

    for source in manifest.sources:
        artifact_id: int | None = None
        object_path: Path | None = None
        sha256: str | None = None
        original_filename: str | None = None
        media_type: str | None = None
        was_deduplicated = False
        if source.file:
            incoming = _resolve_pack_file(root, source.file)
            original_filename = incoming.name
            sha256 = file_sha256(incoming)
            existing = database.artifact_by_hash(sha256)
            if existing and Path(existing["object_path"]).is_file():
                object_path = Path(existing["object_path"])
                artifact_id = int(existing["id"])
                media_type = str(existing["media_type"])
                was_deduplicated = True
                report.deduplicated += 1
            else:
                object_path = _object_path(database.library_dir, sha256, incoming.suffix)
                object_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(incoming, object_path)
                media_type = mimetypes.guess_type(incoming.name)[0] or "application/octet-stream"
                artifact_id = database.upsert_artifact(
                    sha256, object_path, object_path.stat().st_size, media_type
                )

        source_id = database.upsert_source(
            collection_id=collection_id,
            source_key=source.key,
            title=source.title,
            artifact_id=artifact_id,
            original_filename=original_filename,
            source_url=source.url,
            publisher=source.publisher,
            author=source.author,
            published_at=source.published_at,
            retrieved_at=source.retrieved_at,
            source_class=source.source_class,
            authority=source.authority,
            applicability=source.applicability,
            license_note=source.license_note,
            notes=source.notes,
        )

        status = "remote_only"
        error_message: str | None = None
        if object_path is None:
            report.remote_only += 1
        elif object_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            status = "preserved"
            report.preserved += 1
        else:
            ingest_report = ingest_paths(
                database,
                [object_path],
                embedder=embedder,
                embedding_model=embedding_model,
            )
            if ingest_report.failures:
                status = "error"
                error_message = ingest_report.failures[0][1]
                report.failures.append((source.key, error_message))
            else:
                document_id = database.document_id(object_path)
                if document_id is None:
                    status = "error"
                    error_message = "indexed document record was not created"
                    report.failures.append((source.key, error_message))
                else:
                    database.link_source_document(source_id, document_id)
                    status = "indexed"
                    report.indexed += 1
        database.set_source_index_status(source_id, status, error_message)
        imported_sources.append(
            {
                **asdict(source),
                "sha256": sha256,
                "object_path": (
                    str(object_path.relative_to(database.library_dir)) if object_path else None
                ),
                "media_type": media_type,
                "index_status": status,
                "deduplicated": was_deduplicated,
                "index_error": error_message,
            }
        )

    destination = database.library_dir / "packs" / manifest.slug
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = {
        **asdict(manifest),
        "imported_at": datetime.now(UTC).isoformat(),
        "sources": imported_sources,
    }
    (destination / "manifest.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if manifest.report_file:
        shutil.copy2(_resolve_pack_file(root, manifest.report_file), destination / "report.md")
    report.pack_path = destination
    return report


def _parse_source(raw: dict[str, Any], position: int) -> PackSource:
    key = _required_string(raw, "key", f"source #{position}")
    if not KEY_PATTERN.fullmatch(key):
        raise ValueError(
            f"source #{position} key must use lowercase letters, numbers, "
            "dots, underscores, or hyphens"
        )
    authority = _optional_string(raw, "authority") or "unknown"
    if authority not in AUTHORITIES:
        allowed = ", ".join(sorted(AUTHORITIES))
        raise ValueError(f"source #{position} authority must be one of: {allowed}")
    source_file = _optional_string(raw, "file")
    url = _optional_string(raw, "url")
    if not source_file and not url:
        raise ValueError(f"source #{position} must provide file or url")
    if url and urlparse(url).scheme not in {"http", "https"}:
        raise ValueError(f"source #{position} url must use http or https")
    applicability = raw.get("applicability", [])
    if not isinstance(applicability, list) or not all(
        isinstance(item, str) and item.strip() for item in applicability
    ):
        raise ValueError(f"source #{position} applicability must be an array of strings")
    return PackSource(
        key=key,
        title=_required_string(raw, "title", f"source #{position}"),
        file=source_file,
        url=url,
        publisher=_optional_string(raw, "publisher"),
        author=_optional_string(raw, "author"),
        published_at=_optional_string(raw, "published_at"),
        retrieved_at=_optional_string(raw, "retrieved_at"),
        source_class=_optional_string(raw, "source_class") or "other",
        authority=authority,
        applicability=[item.strip() for item in applicability],
        license_note=_optional_string(raw, "license_note"),
        notes=_optional_string(raw, "notes"),
    )


def _required_string(raw: dict[str, Any], key: str, context: str = "manifest") -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} {key} must be a non-empty string")
    return value.strip()


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string when provided")
    return value.strip() or None


def _resolve_pack_file(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"pack file escapes the pack directory: {relative_path}")
    if not candidate.is_file():
        raise ValueError(f"pack file not found: {relative_path}")
    return candidate


def _object_path(library_dir: Path, sha256: str, suffix: str) -> Path:
    safe_suffix = suffix.lower() if re.fullmatch(r"\.[a-z0-9]{1,10}", suffix.lower()) else ""
    return library_dir / "objects" / sha256[:2] / f"{sha256}{safe_suffix}"
