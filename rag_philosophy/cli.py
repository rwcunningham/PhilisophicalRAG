from __future__ import annotations

from typing import NoReturn

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import get_settings
from .index import build_index
from .loaders import load_corpus
from .rag import PhilosophyRAG


app = typer.Typer(help="Build and query a philosophy RAG counterargument system.")
console = Console()


@app.command("index")
def index_command(
    reset: bool = typer.Option(True, "--reset/--no-reset", help="Delete the existing index before rebuilding."),
) -> None:
    """Build the vector index from the local corpus."""
    settings = get_settings()
    try:
        source_count, chunk_count = build_index(settings, reset=reset)
    except Exception as exc:
        _exit_with_error(exc)
    console.print(
        Panel.fit(
            f"Indexed [bold]{source_count}[/bold] source document(s) into [bold]{chunk_count}[/bold] chunks.\n"
            f"Index: {settings.index_dir}",
            title="Index built",
        )
    )


@app.command()
def sources() -> None:
    """List source documents found in the corpus directory."""
    settings = get_settings()
    try:
        documents = load_corpus(settings.corpus_dir)
    except Exception as exc:
        _exit_with_error(exc)

    table = Table(title=f"Sources in {settings.corpus_dir}")
    table.add_column("Source")
    table.add_column("Author")
    table.add_column("Title")
    table.add_column("Page")
    table.add_column("Characters", justify="right")

    for document in documents:
        metadata = document.metadata
        table.add_row(
            str(metadata.get("source", "")),
            str(metadata.get("author", "")),
            str(metadata.get("title", "")),
            str(metadata.get("page", "")),
            str(len(document.page_content)),
        )

    console.print(table)


@app.command()
def ask(
    claim: str = typer.Argument(..., help="The user's philosophical claim to challenge."),
    show_context: bool = typer.Option(False, "--show-context", help="Show retrieved passages after the answer."),
) -> None:
    """Generate one sourced counterargument."""
    settings = get_settings()
    try:
        rag = PhilosophyRAG(settings)
        response = rag.answer(claim)
    except Exception as exc:
        _exit_with_error(exc)
    _print_response(response.answer, response.retrieval_query, response.warnings)
    if show_context:
        _print_context(response.sources)


@app.command()
def chat() -> None:
    """Run an interactive debate loop."""
    settings = get_settings()
    try:
        rag = PhilosophyRAG(settings)
    except Exception as exc:
        _exit_with_error(exc)
    history: list[tuple[str, str]] = []

    console.print(Panel.fit("Type a philosophical claim. Use /quit to exit.", title="Philosophy RAG Chat"))
    while True:
        claim = typer.prompt("You")
        if claim.strip().lower() in {"/q", "/quit", "quit", "exit"}:
            break
        try:
            response = rag.answer(claim, history=history)
        except Exception as exc:
            _exit_with_error(exc)
        _print_response(response.answer, response.retrieval_query, response.warnings)
        history.append(("User", claim))
        history.append(("Assistant", response.answer))


def _print_response(answer: str, retrieval_query: str, warnings: list[str]) -> None:
    console.print(Panel(answer, title="Counterargument", expand=False))
    console.print(f"[dim]Retrieval query:[/dim] {retrieval_query}")
    for warning in warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def _print_context(sources) -> None:
    for source in sources:
        metadata = source.metadata
        title = " | ".join(
            str(metadata.get(key))
            for key in ("author", "title", "work", "source", "page")
            if metadata.get(key)
        )
        console.print(Panel(source.text, title=f"{source.citation} {title}", expand=False))


def _exit_with_error(exc: Exception) -> NoReturn:
    console.print(f"[red]Error:[/red] {exc}")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
