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
