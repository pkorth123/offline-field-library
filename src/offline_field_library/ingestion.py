"""Incremental ingestion pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from .database import LibraryDatabase
from .documents import chunk_sections, discover, extract, file_sha256
from .models import IngestReport


class Embedder(Protocol):
    def embed(self, texts: Sequence[str], model: str) -> list[list[float]]: ...


def ingest_paths(
    database: LibraryDatabase,
    paths: Sequence[Path],
    embedder: Embedder | None,
    embedding_model: str | None,
    batch_size: int = 32,
    max_chars: int = 1_200,
    overlap_chars: int = 180,
) -> IngestReport:
    database.initialize()
    report = IngestReport()
    discovered = discover(paths)
    if not discovered:
        for path in paths:
            report.failures.append((path, "no supported PDF, text, Markdown, or RST files found"))
        return report

    for path in discovered:
        try:
            digest = file_sha256(path)
            if database.document_is_current(
                path, digest, require_embeddings=embedding_model is not None
            ):
                report.unchanged += 1
                continue
            chunks = chunk_sections(extract(path), max_chars, overlap_chars)
            if not chunks:
                raise ValueError("no extractable text found")
            if embedder is not None and embedding_model is not None:
                vectors: list[list[float]] = []
                for start in range(0, len(chunks), batch_size):
                    texts = [chunk.text for chunk in chunks[start : start + batch_size]]
                    vectors.extend(embedder.embed(texts, embedding_model))
                if len(vectors) != len(chunks):
                    raise RuntimeError("embedding model returned an unexpected number of vectors")
                chunks = [
                    replace(chunk, embedding=vector)
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ]
            updated = database.replace_document(path, digest, chunks, embedding_model)
            if updated:
                report.updated += 1
            else:
                report.added += 1
            report.chunks += len(chunks)
        except Exception as error:
            report.failures.append((path, str(error)))
    return report
