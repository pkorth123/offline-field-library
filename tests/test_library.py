from pathlib import Path

from offline_field_library.database import LibraryDatabase
from offline_field_library.ingestion import ingest_paths
from offline_field_library.rag import answer
from offline_field_library.retrieval import search


class FakeBackend:
    def embed(self, texts, model):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("water" in lowered or "hydration" in lowered),
                    float("radio" in lowered),
                    0.1,
                ]
            )
        return vectors

    def generate(self, messages, model, temperature=0.1):
        assert "[1] Source:" in messages[-1]["content"]
        return "Carry water in the field [1]."


def test_incremental_hybrid_search_and_answer(tmp_path: Path) -> None:
    source = tmp_path / "field-notes.md"
    source.write_text("Hydration protocol: carry two liters of water on the ridge.")
    database = LibraryDatabase(tmp_path / "library")
    backend = FakeBackend()

    first = ingest_paths(database, [source], backend, "fake-embed")
    second = ingest_paths(database, [source], backend, "fake-embed")

    assert first.added == 1
    assert first.chunks == 1
    assert second.unchanged == 1

    results = search(database, "water", embedder=backend, embedding_model="fake-embed")
    assert results[0].path == source
    assert results[0].lexical_rank == 1
    assert results[0].semantic_rank == 1
    assert answer("What should I carry?", results, backend, "fake-chat") == (
        "Carry water in the field [1]."
    )


def test_keyword_only_ingestion_and_search(tmp_path: Path) -> None:
    source = tmp_path / "radio.txt"
    source.write_text("Use channel seven for the evening radio check.")
    database = LibraryDatabase(tmp_path / "library")

    report = ingest_paths(database, [source], None, None)
    results = search(database, "evening radio", limit=3)

    assert report.added == 1
    assert database.counts() == (1, 1, 0)
    assert len(results) == 1
    assert "channel seven" in results[0].text


def test_changed_document_is_replaced(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("alpha marker")
    database = LibraryDatabase(tmp_path / "library")
    ingest_paths(database, [source], None, None)

    source.write_text("bravo marker")
    report = ingest_paths(database, [source], None, None)

    assert report.updated == 1
    assert not search(database, "alpha")
    assert search(database, "bravo")[0].path == source


def test_keyword_only_document_can_be_backfilled_with_embeddings(tmp_path: Path) -> None:
    source = tmp_path / "water.txt"
    source.write_text("Carry water on the ridge.")
    database = LibraryDatabase(tmp_path / "library")
    ingest_paths(database, [source], None, None)

    report = ingest_paths(database, [source], FakeBackend(), "fake-embed")

    assert report.updated == 1
    assert database.counts() == (1, 1, 1)
