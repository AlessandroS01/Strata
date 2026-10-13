"""Terminal output renderers for Strata command-line interface.

Provides rich visual representations for ingestion summaries, sync operations,
hybrid retrieval results, and system status telemetry.
"""

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from strata.core.models import SearchResult, SyncStats

_console = Console()


def get_console() -> Console:
    """Return default rich console instance."""
    return _console


def render_ingest_summary(
    documents_count: int,
    sections_count: int,
    chunks_count: int,
    console: Console | None = None,
) -> None:
    """Render structured table and summary of ingested documents, sections, and chunks."""
    c = console or _console

    table = Table(
        title="Ingestion Summary",
        title_style="bold green",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Entity", style="cyan", no_wrap=True)
    table.add_column("Count", style="green", justify="right")

    table.add_row("Total Documents", str(documents_count))
    table.add_row("Total Sections", str(sections_count))
    table.add_row("Total Chunks", str(chunks_count))

    c.print(table)
    c.print(
        f"[bold green]Ingestion complete:[/bold green] created "
        f"[cyan]{documents_count}[/cyan] documents, "
        f"[cyan]{sections_count}[/cyan] sections, and "
        f"[cyan]{chunks_count}[/cyan] chunks."
    )


def render_sync_summary(
    report: SyncStats,
    console: Console | None = None,
) -> None:
    """Render differential synchronization statistics and execution telemetry."""
    c = console or _console

    table = Table(
        title="Index Synchronization Summary",
        title_style="bold cyan",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Operation", style="cyan", no_wrap=True)
    table.add_column("Record Count", style="yellow", justify="right")

    table.add_row("Added Records", str(report.new_chunks))
    table.add_row("Updated Records", "0")
    table.add_row("Deleted Records", str(report.orphaned_chunks))
    table.add_row("Unchanged Records", str(report.unchanged_chunks))
    table.add_row("Total Active Chunks", str(report.total_chunks))
    table.add_row("Total Documents", str(report.total_documents))
    table.add_row("Elapsed Execution Time", f"{report.elapsed_ms:.2f} ms")

    c.print(table)


def render_search_results(
    results: list[SearchResult],
    query: str,
    console: Console | None = None,
) -> None:
    """Render hybrid search results inside styled Rich panels."""
    c = console or _console

    if not results:
        c.print(f"[yellow]No matching chunks found for query:[/yellow] [bold]{query}[/bold]")
        return

    c.print(
        f"[bold cyan]Found {len(results)} matching result(s) for query:[/bold cyan] "
        f'[bold italic]"{query}"[/bold italic]\n'
    )

    for i, res in enumerate(results, 1):
        target_chunk = res.chunk
        chunk_text = target_chunk.chunk.text if target_chunk else "No chunk text available"
        heading_path = target_chunk.chunk.heading_path if target_chunk else "N/A"
        doc_title = (
            target_chunk.document_title
            if target_chunk and target_chunk.document_title
            else "Untitled"
        )

        parent_context = (
            target_chunk.parent_context
            if target_chunk and target_chunk.parent_context
            else (res.parent_section.content if res.parent_section else None)
        )

        # Header metadata
        meta_text = Text.assemble(
            ("Document: ", "bold cyan"),
            (f"{doc_title}\n", "white"),
            ("Heading Path: ", "bold cyan"),
            (f"{heading_path}\n", "yellow"),
            ("Rerank Score: ", "bold cyan"),
            (f"{res.score:.4f}", "bold green"),
        )

        elements: list[Text | Panel] = [meta_text]

        # Matching Chunk text panel
        elements.append(
            Panel(
                Text(chunk_text, style="white"),
                title="[bold green]Matching Chunk[/bold green]",
                border_style="green",
            )
        )

        # Parent Section Context if expanded
        if parent_context:
            sec_title = res.parent_section.title if res.parent_section else heading_path
            elements.append(
                Panel(
                    Text(parent_context, style="dim white"),
                    title=f"[bold blue]Parent Section Context ({sec_title})[/bold blue]",
                    border_style="blue",
                )
            )

        # Adjacent Chunks window if present
        if res.adjacent_chunks:
            adj_texts = [
                f"[Chunk {adj.chunk.index}]: {adj.chunk.text}" for adj in res.adjacent_chunks
            ]
            elements.append(
                Panel(
                    Text("\n---\n".join(adj_texts), style="dim"),
                    title=f"[bold yellow]Adjacent Context Chunks ({len(res.adjacent_chunks)})[/bold yellow]",
                    border_style="yellow",
                )
            )

        panel = Panel(
            Group(*elements),
            title=f"[bold magenta]Result #{i}[/bold magenta] (Latency: {res.total_latency_ms:.2f}ms)",
            border_style="cyan",
            padding=(1, 2),
        )
        c.print(panel)


def render_status(
    doc_count: int,
    section_count: int,
    chunk_count: int,
    vector_count: int,
    bm25_status: str,
    bm25_size_bytes: int,
    console: Console | None = None,
) -> None:
    """Render database, vector collection, and sparse index telemetry."""
    c = console or _console

    table = Table(
        title="Strata System Status & Telemetry",
        title_style="bold cyan",
        header_style="bold green",
        show_lines=True,
    )
    table.add_column("Subsystem", style="cyan", no_wrap=True)
    table.add_column("Metric", style="yellow")
    table.add_column("Value", style="green", justify="right")

    table.add_row("SQLite (Canonical)", "Total Documents", str(doc_count))
    table.add_row("SQLite (Canonical)", "Total Sections", str(section_count))
    table.add_row("SQLite (Canonical)", "Total Chunks", str(chunk_count))
    table.add_row("Qdrant (Dense Vectors)", "Active Vector Points", str(vector_count))
    table.add_row("BM25 (Sparse Index)", "Index File Status", bm25_status)
    table.add_row(
        "BM25 (Sparse Index)",
        "Index Byte Size on Disk",
        f"{bm25_size_bytes:,} bytes",
    )

    c.print(table)


def render_error(
    message: str,
    title: str = "Error",
    console: Console | None = None,
) -> None:
    """Render domain error message in styled alert panel."""
    c = console or _console
    c.print(
        Panel(
            Text(message, style="bold red"),
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
        )
    )
