import json
from pathlib import Path

import pytest

from offline_field_library.database import LibraryDatabase
from offline_field_library.documents import extract
from offline_field_library.packs import import_pack, load_manifest


def _write_manifest(pack: Path, slug: str = "ridge-generator") -> None:
    manifest = {
        "schema_version": 1,
        "slug": slug,
        "title": "Ridge Generator",
        "description": "Operation and repair references.",
        "brief": "Research operation, maintenance, and repair.",
        "report_file": "research-report.md",
        "sources": [
            {
                "key": "owner-manual",
                "title": "Owner Manual",
                "file": "sources/manual.md",
                "url": "https://example.com/manual",
                "publisher": "Ridge Power",
                "source_class": "manufacturer_manual",
                "authority": "official",
                "applicability": ["RG-2000"],
            },
            {
                "key": "wiring-diagram",
                "title": "Wiring Diagram",
                "file": "sources/wiring.png",
                "source_class": "illustration",
                "authority": "official",
            },
            {
                "key": "forum-thread",
                "title": "Owner Discussion",
                "url": "https://example.com/forum/thread",
                "source_class": "forum",
                "authority": "anecdotal",
            },
        ],
    }
    (pack / "manifest.json").write_text(json.dumps(manifest))


def test_pack_import_preserves_indexes_and_deduplicates_sources(tmp_path: Path) -> None:
    pack = tmp_path / "incoming"
    (pack / "sources").mkdir(parents=True)
    (pack / "sources" / "manual.md").write_text("Change the oil every fifty hours.")
    (pack / "sources" / "wiring.png").write_bytes(b"fake-png-data")
    (pack / "research-report.md").write_text("# Coverage\n\nOwner manual located.")
    _write_manifest(pack)
    database = LibraryDatabase(tmp_path / "library")

    first = import_pack(database, pack, embedder=None, embedding_model=None)
    second = import_pack(database, pack, embedder=None, embedding_model=None)

    assert first.indexed == 1
    assert first.preserved == 1
    assert first.remote_only == 1
    assert second.deduplicated == 2
    assert database.counts() == (1, 1, 0)
    assert database.list_collections()[0]["source_count"] == 3
    statuses = {row["source_key"]: row["index_status"] for row in database.list_sources(first.slug)}
    assert statuses == {
        "forum-thread": "remote_only",
        "owner-manual": "indexed",
        "wiring-diagram": "preserved",
    }
    assert len(list((database.library_dir / "objects").rglob("*.*"))) == 2
    assert (first.pack_path / "manifest.json").is_file()
    assert (first.pack_path / "report.md").is_file()


def test_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    pack = tmp_path / "incoming"
    pack.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    manifest = {
        "schema_version": 1,
        "slug": "unsafe-pack",
        "title": "Unsafe",
        "sources": [{"key": "escape", "title": "Escape", "file": "../outside.txt"}],
    }
    (pack / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="escapes the pack directory"):
        load_manifest(pack)


def test_html_extraction_ignores_scripts_and_keeps_image_alt_text(tmp_path: Path) -> None:
    page = tmp_path / "guide.html"
    page.write_text(
        """
        <html><head><style>.hidden{}</style><script>ignore me</script></head>
        <body><h1>Maintenance</h1><p>Close the fuel valve.</p>
        <img src="valve.png" alt="Fuel valve location"></body></html>
        """
    )

    text = extract(page)[0].text

    assert "Maintenance" in text
    assert "Close the fuel valve" in text
    assert "[Image: Fuel valve location]" in text
    assert "ignore me" not in text
