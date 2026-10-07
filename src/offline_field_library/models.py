"""Shared domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TextSection:
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    sequence: int
    text: str
    page: int | None
    start_char: int
    end_char: int
    embedding: list[float] | None = None


@dataclass(frozen=True)
class SearchResult:
    chunk_id: int
    path: Path
    title: str
    text: str
    page: int | None
    score: float
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    source_url: str | None = None
    publisher: str | None = None
    source_class: str | None = None
    authority: str | None = None

    @property
    def location(self) -> str:
        suffix = f", page {self.page}" if self.page is not None else ""
        if self.source_class:
            return f"{self.title}{suffix}"
        return f"{self.path}{suffix}"


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    chunks: int = 0
    failures: list[tuple[Path, str]] = field(default_factory=list)
