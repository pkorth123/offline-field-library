"""Command-line interface for the Offline Field Library."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .database import LibraryDatabase
from .ingestion import ingest_paths
from .models import SearchResult
from .ollama_backend import OllamaBackend
from .rag import answer, source_list
from .retrieval import search as retrieve

app = typer.Typer(
    name="offline-field-library",
    help="Build and chat with a private, fully local field-document library.",
    no_args_is_help=True,
)

LibraryOption = Annotated[
    Path,
    typer.Option(
        "--library",
        "-l",
        help="Folder containing the portable SQLite index.",
        envvar="OFL_LIBRARY",
    ),
]


@app.command("init")
def initialize(library: LibraryOption = Path(".field-library")) -> None:
    """Create an empty local library."""
    database = LibraryDatabase(library)
    database.initialize()
    typer.echo(f"Library ready: {database.path}")


@app.command()
def ingest(
    paths: Annotated[list[Path], typer.Argument(help="Files or folders to index.")],
    library: LibraryOption = Path(".field-library"),
    embedding_model: Annotated[
        str, typer.Option(help="Ollama embedding model.")
    ] = "nomic-embed-text",
    no_embeddings: Annotated[
        bool, typer.Option(help="Build a keyword-only index without Ollama.")
    ] = False,
    chunk_chars: Annotated[
        int, typer.Option(min=100, help="Maximum characters per passage.")
    ] = 1_200,
    overlap_chars: Annotated[int, typer.Option(min=0, help="Passage overlap in characters.")] = 180,
) -> None:
    """Incrementally index supported files (PDF, text, Markdown, and RST)."""
    if overlap_chars >= chunk_chars:
        raise typer.BadParameter("--overlap-chars must be smaller than --chunk-chars")
    database = LibraryDatabase(library)
    backend = None if no_embeddings else OllamaBackend()
    embedding_model_or_none = None if no_embeddings else embedding_model
    report = ingest_paths(
        database,
        paths,
        embedder=backend,
        embedding_model=embedding_model_or_none,
        max_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )
    typer.echo(
        f"Indexed {report.added} new, {report.updated} updated, "
        f"{report.unchanged} unchanged ({report.chunks} new passages)."
    )
    for path, message in report.failures:
        typer.echo(f"Failed: {path} — {message}", err=True)
    if report.failures:
        raise typer.Exit(code=1)


@app.command()
def status(library: LibraryOption = Path(".field-library")) -> None:
    """Show index size and model configuration."""
    database = LibraryDatabase(library)
    documents, chunks, embedded = database.counts()
    model = database.get_metadata("embedding_model") or "none (keyword-only)"
    typer.echo(f"Library: {database.path}")
    typer.echo(f"Documents: {documents}")
    typer.echo(f"Passages: {chunks} ({embedded} embedded)")
    typer.echo(f"Embedding model: {model}")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Words or question to find.")],
    library: LibraryOption = Path(".field-library"),
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=50)] = 5,
    keyword_only: Annotated[
        bool, typer.Option(help="Do not call the embedding model for this search.")
    ] = False,
) -> None:
    """Search the library and print matching passages."""
    database = _ready_database(library)
    try:
        results = _retrieve(database, query, limit, keyword_only)
    except RuntimeError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error
    if not results:
        typer.echo("No matching passages found.")
        raise typer.Exit(code=1)
    for number, result in enumerate(results, start=1):
        ranks = []
        if result.lexical_rank is not None:
            ranks.append(f"keyword #{result.lexical_rank}")
        if result.semantic_rank is not None:
            ranks.append(f"semantic #{result.semantic_rank}")
        typer.echo(f"\n[{number}] {result.location} ({', '.join(ranks)})")
        typer.echo(result.text)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer from the library.")],
    library: LibraryOption = Path(".field-library"),
    model: Annotated[str, typer.Option(help="Ollama chat model.")] = "llama3.2:3b",
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=20)] = 5,
    keyword_only: Annotated[
        bool, typer.Option(help="Retrieve by keywords without an embedding call.")
    ] = False,
) -> None:
    """Answer one question using retrieved passages with citations."""
    database = _ready_database(library)
    backend = OllamaBackend()
    try:
        results = _retrieve(database, question, limit, keyword_only, backend)
        response = answer(question, results, backend, model)
    except RuntimeError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(response)
    if results:
        typer.echo(f"\nSources:\n{source_list(results)}")


@app.command()
def chat(
    library: LibraryOption = Path(".field-library"),
    model: Annotated[str, typer.Option(help="Ollama chat model.")] = "llama3.2:3b",
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=20)] = 5,
    keyword_only: Annotated[
        bool, typer.Option(help="Retrieve by keywords without an embedding call.")
    ] = False,
) -> None:
    """Start a multi-turn, cited chat. Type /exit to stop."""
    database = _ready_database(library)
    backend = OllamaBackend()
    history: list[dict[str, str]] = []
    typer.echo("Offline Field Library chat. Type /exit to stop.")
    while True:
        try:
            question = typer.prompt("You").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            return
        if question.lower() in {"/exit", "/quit", "exit", "quit"}:
            return
        if not question:
            continue
        try:
            retrieval_query = _chat_retrieval_query(question, history)
            results = _retrieve(database, retrieval_query, limit, keyword_only, backend)
            response = answer(question, results, backend, model, history)
        except RuntimeError as error:
            typer.echo(f"Error: {error}", err=True)
            continue
        typer.echo(f"\nAssistant: {response}")
        if results:
            typer.echo(f"\nSources:\n{source_list(results)}")
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]
        )


def _ready_database(library: Path) -> LibraryDatabase:
    database = LibraryDatabase(library)
    if not database.path.exists():
        typer.echo(
            f"Library not found: {database.path}. Run 'offline-field-library init'.", err=True
        )
        raise typer.Exit(code=2)
    return database


def _retrieve(
    database: LibraryDatabase,
    query: str,
    limit: int,
    keyword_only: bool,
    backend: OllamaBackend | None = None,
) -> list[SearchResult]:
    embedding_model = None if keyword_only else database.get_metadata("embedding_model")
    embedder = (backend or OllamaBackend()) if embedding_model else None
    return retrieve(
        database,
        query,
        limit=limit,
        embedder=embedder,
        embedding_model=embedding_model,
    )


def _chat_retrieval_query(question: str, history: list[dict[str, str]]) -> str:
    """Include the last user turn so short follow-up questions retrieve coherently."""
    previous_question = next(
        (message["content"] for message in reversed(history) if message["role"] == "user"), None
    )
    return f"{previous_question}\n{question}" if previous_question else question


def main() -> None:
    app()
