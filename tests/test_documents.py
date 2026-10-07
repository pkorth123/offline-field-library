from pathlib import Path

import pymupdf

from offline_field_library.documents import chunk_sections, discover, extract
from offline_field_library.models import TextSection


def test_discover_supported_files_recursively(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "guide.md").write_text("guide")
    (tmp_path / "nested" / "notes.txt").write_text("notes")
    (tmp_path / "image.jpg").write_bytes(b"no")

    assert [path.name for path in discover([tmp_path])] == ["guide.md", "notes.txt"]


def test_chunking_preserves_page_and_overlap() -> None:
    text = "First sentence. " * 30
    chunks = chunk_sections([TextSection(text=text, page=4)], max_chars=140, overlap_chars=25)

    assert len(chunks) > 2
    assert all(chunk.page == 4 for chunk in chunks)
    assert all(len(chunk.text) <= 140 for chunk in chunks)
    assert chunks[1].start_char < chunks[0].end_char


def test_pdf_extraction_retains_page_numbers(tmp_path: Path) -> None:
    path = tmp_path / "manual.pdf"
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "First page route")
        document.new_page().insert_text((72, 72), "Second page shelter")
        document.save(path)

    sections = extract(path)

    assert [(section.page, section.text) for section in sections] == [
        (1, "First page route"),
        (2, "Second page shelter"),
    ]
