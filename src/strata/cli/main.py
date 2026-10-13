"""Command-line interface entrypoint for Strata air-gapped knowledge engine.

Orchestrates document ingestion, index synchronization, hybrid retrieval,
and system status telemetry using Typer and Rich.
"""

from pathlib import Path
from typing import Annotated

import typer

from strata.cli import views
from strata.cli.dependencies import get_dependencies
from strata.core.exceptions import (
    ModelInferenceError,
    StrataError,
    SyncConflictError,
)

app = typer.Typer(
    name="strata",
    help="100% offline personal knowledge base and hybrid RAG engine.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command(name="ingest", help="Ingest markdown document(s) into canonical SQLite store.")
def ingest(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to a markdown file or directory of markdown files to ingest.",
            exists=False,
        ),
    ],
    tag: Annotated[
        list[str] | None,
        typer.Option(
            "--tag",
            "-t",
            help="Optional tag(s) to attach to ingested documents.",
        ),
    ] = None,
) -> None:
    """Ingest markdown files into SQLite with AST section and chunk extraction."""
    deps = get_dependencies()

    target_path = path.resolve()
    if not target_path.exists():
        views.render_error(f"Path does not exist: {target_path}", title="Ingestion Error")
        raise typer.Exit(code=1)

    try:
        if target_path.is_file():
            documents = [deps.reader.read_file(target_path)]
        elif target_path.is_dir():
            documents = deps.reader.scan_directory(target_path)
        else:
            views.render_error(
                f"Path is neither a regular file nor directory: {target_path}",
                title="Ingestion Error",
            )
            raise typer.Exit(code=1)

        if not documents:
            views.render_error(
                f"No markdown documents found at: {target_path}",
                title="Ingestion Notice",
            )
            raise typer.Exit(code=1)

        total_docs = 0
        total_sections = 0
        total_chunks = 0

        for doc in documents:
            if tag:
                # Merge user tags with existing frontmatter tags, preserving order
                merged_tags = list(dict.fromkeys(list(doc.tags) + list(tag)))
                doc = doc.model_copy(update={"tags": merged_tags})

            sections = deps.chunker.extract_sections(doc)
            chunks = deps.chunker.chunk_document(doc)

            # Reconcile existing document by path to ensure ACID freshness
            existing = deps.sqlite_store.get_document_by_path(doc.file_path)
            if existing is not None:
                deps.sqlite_store.delete_document(existing.id)

            deps.sqlite_store.add_document(doc)
            deps.sqlite_store.add_sections(sections)
            deps.sqlite_store.add_chunks(chunks)

            total_docs += 1
            total_sections += len(sections)
            total_chunks += len(chunks)

        views.render_ingest_summary(
            documents_count=total_docs,
            sections_count=total_sections,
            chunks_count=total_chunks,
        )

    except StrataError as exc:
        views.render_error(f"Strata domain error during ingestion: {exc}")
        raise typer.Exit(code=1) from exc
    except (OSError, ValueError) as exc:
        views.render_error(f"Failed to ingest documents from {target_path}: {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="add", help="Alias for ingest command.", hidden=True)
def add_alias(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to a markdown file or directory to ingest.",
            exists=False,
        ),
    ],
    tag: Annotated[
        list[str] | None,
        typer.Option(
            "--tag",
            "-t",
            help="Optional tag(s) to attach to ingested documents.",
        ),
    ] = None,
) -> None:
    """CLI alias for ingest command."""
    ingest(path=path, tag=tag)


@app.command(name="sync", help="Synchronize derived vector and sparse indices against SQLite.")
def sync(
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            "-b",
            help="Batch size override for embedding and vector upsert.",
        ),
    ] = None,
) -> None:
    """Trigger differential index updates across Qdrant and BM25."""
    deps = get_dependencies()

    try:
        report = deps.sync_engine.sync(batch_size=batch_size)
        views.render_sync_summary(report)
    except SyncConflictError as exc:
        views.render_error(f"Synchronization conflict: {exc}", title="Sync Conflict")
        raise typer.Exit(code=1) from exc
    except StrataError as exc:
        views.render_error(f"Strata domain error during sync: {exc}", title="Sync Error")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        views.render_error(f"Unexpected error during sync: {exc}", title="Sync Error")
        raise typer.Exit(code=1) from exc


@app.command(name="search", help="Execute decoupled hybrid retrieval without an LLM.")
def search(
    query: Annotated[
        str,
        typer.Argument(
            help="Search query text.",
        ),
    ],
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-l",
            "--top-k",
            "-k",
            help="Maximum number of ranked results to return.",
        ),
    ] = None,
    tag: Annotated[
        list[str] | None,
        typer.Option(
            "--tag",
            "-t",
            help="Filter candidates by metadata tags.",
        ),
    ] = None,
    expand_section: Annotated[
        bool,
        typer.Option(
            "--expand-section",
            "-e",
            help="Hydrate parent Section text for Small-to-Big context expansion.",
        ),
    ] = False,
    window_size: Annotated[
        int,
        typer.Option(
            "--window-size",
            "-w",
            help="Number of adjacent sibling chunks to fetch on each side.",
        ),
    ] = 0,
) -> None:
    """Execute hybrid retrieval returning ranked chunks inside Rich panels."""
    if not query or not query.strip():
        views.render_error("Search query cannot be empty or blank.", title="Validation Error")
        raise typer.Exit(code=1)

    if limit is not None and limit < 1:
        views.render_error(
            f"Limit must be >= 1, got {limit}.",
            title="Validation Error",
        )
        raise typer.Exit(code=1)

    if window_size < 0:
        views.render_error(
            f"Window size must be >= 0, got {window_size}.",
            title="Validation Error",
        )
        raise typer.Exit(code=1)

    deps = get_dependencies()

    try:
        results = deps.searcher.search(
            query=query.strip(),
            limit=limit,
            tags=tag,
            expand_section=expand_section,
            window_size=window_size,
        )
        views.render_search_results(results=results, query=query.strip())
    except ModelInferenceError as exc:
        views.render_error(f"Model inference failed: {exc}", title="Inference Error")
        raise typer.Exit(code=1) from exc
    except StrataError as exc:
        views.render_error(f"Strata search error: {exc}", title="Search Error")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        views.render_error(f"Unexpected search error: {exc}", title="Search Error")
        raise typer.Exit(code=1) from exc


@app.command(name="status", help="Display canonical storage and derived index telemetry.")
def status() -> None:
    """Query and display storage, vector points, and sparse index status."""
    deps = get_dependencies()

    try:
        doc_count = deps.sqlite_store.count_documents()
        chunk_count = deps.sqlite_store.count_chunks()
        section_count = (
            deps.sqlite_store.count_sections()
            if hasattr(deps.sqlite_store, "count_sections")
            else 0
        )

        vector_count = deps.vector_store.count()

        bm25_path = deps.settings.bm25_index_path
        if bm25_path and Path(bm25_path).is_file():
            bm25_status = "Active (persisted)"
            bm25_size = Path(bm25_path).stat().st_size
        else:
            bm25_status = "Not found (unindexed)"
            bm25_size = 0

        views.render_status(
            doc_count=doc_count,
            section_count=section_count,
            chunk_count=chunk_count,
            vector_count=vector_count,
            bm25_status=bm25_status,
            bm25_size_bytes=bm25_size,
        )
    except StrataError as exc:
        views.render_error(f"Failed to query system status: {exc}", title="Status Error")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        views.render_error(f"Unexpected status error: {exc}", title="Status Error")
        raise typer.Exit(code=1) from exc


def main() -> None:
    """CLI application entrypoint."""
    app()


if __name__ == "__main__":
    main()
