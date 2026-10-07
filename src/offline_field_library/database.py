"""Portable SQLite persistence for documents, chunks, FTS, and vectors."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .models import Chunk

SCHEMA_VERSION = "1"


class LibraryDatabase:
    def __init__(self, library_dir: Path):
        self.library_dir = library_dir.expanduser().resolve()
        self.path = self.library_dir / "index.sqlite3"

    def initialize(self) -> None:
        self.library_dir.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    page INTEGER,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB,
                    embedding_dimensions INTEGER,
                    UNIQUE(document_id, sequence)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    text,
                    content='chunks',
                    content_rowid='id',
                    tokenize='unicode61 remove_diacritics 2'
                );
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text)
                    VALUES ('delete', old.id, old.text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text)
                    VALUES ('delete', old.id, old.text);
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
                END;
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def document_is_current(
        self, path: Path, sha256: str, require_embeddings: bool = False
    ) -> bool:
        if not self.path.exists():
            return False
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT d.sha256,
                       COUNT(c.id) AS chunk_count,
                       SUM(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedded_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.path = ?
                GROUP BY d.id
                """,
                (str(path.resolve()),),
            ).fetchone()
        if not row or str(row["sha256"]) != sha256:
            return False
        return not require_embeddings or int(row["chunk_count"]) == int(row["embedded_count"] or 0)

    def replace_document(
        self,
        path: Path,
        sha256: str,
        chunks: Sequence[Chunk],
        embedding_model: str | None,
    ) -> bool:
        """Atomically insert or replace one document. Returns True when updated."""
        resolved = path.resolve()
        stat = resolved.stat()
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            previous_model = connection.execute(
                "SELECT value FROM metadata WHERE key = 'embedding_model'"
            ).fetchone()
            if embedding_model and previous_model and previous_model["value"] != embedding_model:
                raise ValueError(
                    "This library was indexed with a different embedding model "
                    f"({previous_model['value']}). Use that model or create a new library."
                )
            existing = connection.execute(
                "SELECT id FROM documents WHERE path = ?", (str(resolved),)
            ).fetchone()
            updated = existing is not None
            if existing:
                connection.execute("DELETE FROM documents WHERE id = ?", (existing["id"],))
            cursor = connection.execute(
                """
                INSERT INTO documents(path, title, kind, sha256, size_bytes, mtime_ns, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(resolved),
                    resolved.stem,
                    resolved.suffix.lower().lstrip("."),
                    sha256,
                    stat.st_size,
                    stat.st_mtime_ns,
                    now,
                ),
            )
            document_id = int(cursor.lastrowid)
            for chunk in chunks:
                vector, dimensions = _encode_vector(chunk.embedding)
                connection.execute(
                    """
                    INSERT INTO chunks(
                        document_id, sequence, page, start_char, end_char, text,
                        embedding, embedding_dimensions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        chunk.sequence,
                        chunk.page,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.text,
                        vector,
                        dimensions,
                    ),
                )
            if embedding_model:
                connection.execute(
                    "INSERT OR REPLACE INTO metadata(key, value) VALUES ('embedding_model', ?)",
                    (embedding_model,),
                )
        return updated

    def get_metadata(self, key: str) -> str | None:
        if not self.path.exists():
            return None
        with self.connect() as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def counts(self) -> tuple[int, int, int]:
        if not self.path.exists():
            return 0, 0, 0
        with self.connect() as connection:
            documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            embedded = connection.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        return int(documents), int(chunks), int(embedded)

    def lexical_rows(self, query: str, limit: int) -> list[sqlite3.Row]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        with self.connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT c.id, c.text, c.page, d.path, d.title, bm25(chunks_fts) AS rank
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
                    JOIN documents d ON d.id = c.document_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit),
                )
            )

    def embedded_rows(self) -> list[sqlite3.Row]:
        if not self.path.exists():
            return []
        with self.connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT c.id, c.text, c.page, c.embedding, c.embedding_dimensions,
                           d.path, d.title
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.embedding IS NOT NULL
                    """
                )
            )


def decode_vector(blob: bytes, dimensions: int) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32)
    if vector.size != dimensions:
        raise ValueError("Stored embedding has invalid dimensions")
    return vector


def _encode_vector(vector: list[float] | None) -> tuple[bytes | None, int | None]:
    if vector is None:
        return None, None
    array = np.asarray(vector, dtype=np.float32)
    return array.tobytes(), int(array.size)


def _fts_query(query: str) -> str:
    terms = [term for term in query.replace("_", " ").split() if any(ch.isalnum() for ch in term)]
    cleaned = []
    for term in terms:
        safe = "".join(ch for ch in term if ch.isalnum() or ch in {"-", "'"})
        if safe:
            cleaned.append('"' + safe.replace('"', '""') + '"')
    return " OR ".join(cleaned)
