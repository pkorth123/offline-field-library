"""Document discovery, extraction, and deterministic chunking."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from html.parser import HTMLParser
from pathlib import Path

import pymupdf

from .models import Chunk, TextSection

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".rst", ".html", ".htm"}


def discover(paths: Iterable[Path]) -> list[Path]:
    """Return supported files from explicit files and recursively scanned folders."""
    found: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.add(path)
        elif path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                    found.add(candidate.resolve())
    return sorted(found, key=lambda item: str(item).lower())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(path: Path) -> list[TextSection]:
    """Extract plain text while retaining PDF page numbers."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        sections: list[TextSection] = []
        with pymupdf.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                text = _clean_text(page.get_text("text"))
                if text:
                    sections.append(TextSection(text=text, page=page_number))
        return sections

    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        parser = _HTMLTextExtractor()
        parser.feed(text)
        text = parser.text
    cleaned = _clean_text(text)
    return [TextSection(text=cleaned)] if cleaned else []


def chunk_sections(
    sections: Iterable[TextSection], max_chars: int = 1_200, overlap_chars: int = 180
) -> list[Chunk]:
    """Split sections on paragraph/sentence boundaries with a small overlap."""
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")

    chunks: list[Chunk] = []
    sequence = 0
    for section in sections:
        text = section.text.strip()
        cursor = 0
        while cursor < len(text):
            hard_end = min(cursor + max_chars, len(text))
            end = _best_break(text, cursor, hard_end) if hard_end < len(text) else hard_end
            piece = text[cursor:end].strip()
            if piece:
                piece_start = cursor + len(text[cursor:end]) - len(text[cursor:end].lstrip())
                chunks.append(
                    Chunk(
                        sequence=sequence,
                        text=piece,
                        page=section.page,
                        start_char=piece_start,
                        end_char=piece_start + len(piece),
                    )
                )
                sequence += 1
            if end >= len(text):
                break
            next_cursor = max(end - overlap_chars, cursor + 1)
            cursor = _next_word_start(text, next_cursor)
    return chunks


def _best_break(text: str, start: int, hard_end: int) -> int:
    minimum = start + int((hard_end - start) * 0.55)
    window = text[minimum:hard_end]
    for pattern in (r"\n\s*\n", r"(?<=[.!?])\s+", r"\s+"):
        matches = list(re.finditer(pattern, window))
        if matches:
            return minimum + matches[-1].end()
    return hard_end


def _next_word_start(text: str, position: int) -> int:
    while position < len(text) and position > 0 and not text[position - 1].isspace():
        position += 1
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class _HTMLTextExtractor(HTMLParser):
    _SKIPPED = {"script", "style", "noscript", "svg"}
    _BLOCKS = {
        "article",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIPPED:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self._BLOCKS:
            self._parts.append("\n")
        if tag == "img":
            alt = next((value for name, value in attrs if name == "alt" and value), None)
            if alt:
                self._parts.append(f"[Image: {alt}]")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIPPED and self._skip_depth:
            self._skip_depth -= 1
            return
        if not self._skip_depth and tag in self._BLOCKS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)
