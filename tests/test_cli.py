import json
from pathlib import Path

from typer.testing import CliRunner

from offline_field_library.cli import app

runner = CliRunner()


def test_keyword_only_cli_flow(tmp_path: Path) -> None:
    source = tmp_path / "manual.md"
    source.write_text("The rendezvous point is the north trailhead.")
    library = tmp_path / "index"

    ingested = runner.invoke(
        app, ["ingest", str(source), "--library", str(library), "--no-embeddings"]
    )
    found = runner.invoke(
        app, ["search", "rendezvous", "--library", str(library), "--keyword-only"]
    )
    status = runner.invoke(app, ["status", "--library", str(library)])

    assert ingested.exit_code == 0, ingested.output
    assert "Indexed 1 new" in ingested.output
    assert found.exit_code == 0, found.output
    assert "north trailhead" in found.output
    assert "Documents: 1" in status.output


def test_import_list_and_search_collection(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    (pack / "sources").mkdir(parents=True)
    (pack / "sources" / "manual.txt").write_text(
        "The collection-only drain valve is behind the service cover."
    )
    (pack / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "slug": "test-generator",
                "title": "Test Generator",
                "description": "A test knowledge pack.",
                "brief": "Find maintenance instructions.",
                "sources": [
                    {
                        "key": "manual",
                        "title": "Test Generator Manual",
                        "file": "sources/manual.txt",
                        "url": "https://example.com/manual",
                        "authority": "official",
                        "source_class": "manufacturer_manual",
                    }
                ],
            }
        )
    )
    library = tmp_path / "library"

    imported = runner.invoke(
        app, ["import-pack", str(pack), "--library", str(library), "--no-embeddings"]
    )
    collections = runner.invoke(app, ["collections", "--library", str(library)])
    sources = runner.invoke(app, ["sources", "test-generator", "--library", str(library)])
    found = runner.invoke(
        app,
        [
            "search",
            "drain valve",
            "--collection",
            "test-generator",
            "--library",
            str(library),
            "--keyword-only",
        ],
    )

    assert imported.exit_code == 0, imported.output
    assert "1 indexed" in imported.output
    assert "test-generator — Test Generator" in collections.output
    assert "manual — Test Generator Manual [official/indexed]" in sources.output
    assert "collection-only drain valve" in found.output
    assert "Test Generator Manual" in found.output
