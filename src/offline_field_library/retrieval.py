"""Hybrid FTS5 and local-vector retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .database import LibraryDatabase, decode_vector
from .models import SearchResult


class QueryEmbedder(Protocol):
    def embed(self, texts: Sequence[str], model: str) -> list[list[float]]: ...


def search(
    database: LibraryDatabase,
    query: str,
    limit: int = 5,
    embedder: QueryEmbedder | None = None,
    embedding_model: str | None = None,
    candidate_limit: int = 50,
) -> list[SearchResult]:
    if limit < 1:
        raise ValueError("limit must be positive")

    lexical = database.lexical_rows(query, candidate_limit)
    semantic = []
    if embedder is not None and embedding_model is not None:
        query_vectors = embedder.embed([query], embedding_model)
        if query_vectors:
            semantic = _semantic_rows(database, query_vectors[0], candidate_limit)

    rows_by_id = {int(row["id"]): row for row in lexical}
    rows_by_id.update({int(row["id"]): row for row, _score in semantic})
    lexical_ranks = {int(row["id"]): rank for rank, row in enumerate(lexical, start=1)}
    semantic_ranks = {int(row["id"]): rank for rank, (row, _score) in enumerate(semantic, start=1)}

    # Reciprocal rank fusion is stable even though BM25 and cosine use different scales.
    fused: list[tuple[int, float]] = []
    for chunk_id in rows_by_id:
        score = 0.0
        if chunk_id in lexical_ranks:
            score += 0.45 / (60 + lexical_ranks[chunk_id])
        if chunk_id in semantic_ranks:
            score += 0.55 / (60 + semantic_ranks[chunk_id])
        fused.append((chunk_id, score))
    fused.sort(key=lambda item: (-item[1], item[0]))

    results: list[SearchResult] = []
    for chunk_id, score in fused[:limit]:
        row = rows_by_id[chunk_id]
        results.append(
            SearchResult(
                chunk_id=chunk_id,
                path=Path(row["path"]),
                title=str(row["title"]),
                text=str(row["text"]),
                page=int(row["page"]) if row["page"] is not None else None,
                score=score,
                lexical_rank=lexical_ranks.get(chunk_id),
                semantic_rank=semantic_ranks.get(chunk_id),
            )
        )
    return results


def _semantic_rows(database: LibraryDatabase, query_vector: list[float], limit: int):
    query = np.asarray(query_vector, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0:
        return []
    scored = []
    for row in database.embedded_rows():
        vector = decode_vector(row["embedding"], int(row["embedding_dimensions"]))
        if vector.size != query.size:
            continue
        denominator = query_norm * float(np.linalg.norm(vector))
        similarity = float(np.dot(query, vector) / denominator) if denominator else 0.0
        scored.append((row, similarity))
    scored.sort(key=lambda item: (-item[1], int(item[0]["id"])))
    return scored[:limit]
