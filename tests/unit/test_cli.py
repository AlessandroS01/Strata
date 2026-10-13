"""Unit and integration test suite for Strata command-line interface.

Tests CLI commands (ingest, sync, search, status), dependency container wiring,
error shielding, and Rich presentation outputs using typer.testing.CliRunner.
"""

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from strata.cli.dependencies import (
    CLIDependencies,
    create_dependencies,
    get_dependencies,
    reset_dependencies,
    set_dependency_override,
)
from strata.cli.main import app
from strata.core.config import Settings
from strata.core.exceptions import (
    ModelInferenceError,
    StrataError,
    SyncConflictError,
)
from strata.core.models import (
    Chunk,
    Document,
    RetrievedChunk,
    SearchResult,
    Section,
    SyncStats,
)
from strata.indexing.bm25_index import BM25Index
from strata.indexing.embedder import FastEmbedder
from strata.indexing.qdrant_store import QdrantVectorStore
from strata.indexing.sync_engine import IndexSyncEngine
from strata.ingestion.chunker import MarkdownChunker
from strata.ingestion.reader import LocalFileReader
from strata.retrieval.hybrid_search import StrataHybridSearcher
from strata.retrieval.reranker import FlashRankReranker
from strata.storage.sqlite_store import SQLiteStore


@dataclass
class MockCLIDependencies:
    """Mock container bundling MagicMock adapters for testing.

    Ensures all adapter methods are typed as MagicMock rather than static FunctionType,
    allowing `.return_value`, `.side_effect`, and mock assertions without type warnings.
    """

    settings: Settings
    sqlite_store: MagicMock
    vector_store: MagicMock
    sparse_index: MagicMock
    embedder: MagicMock
    reranker: MagicMock
    searcher: MagicMock
    sync_engine: MagicMock
    chunker: MagicMock
    reader: MagicMock

    @property
    def hybrid_searcher(self) -> MagicMock:
        """Alias property for searcher mock."""
        return self.searcher


@pytest.fixture
def runner() -> CliRunner:
    """Provide Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Provide isolated Settings rooted in a temporary test directory."""
    data_dir = tmp_path / "strata_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        data_dir=data_dir,
        db_path=data_dir / "canonical.db",
        qdrant_path=data_dir / "qdrant",
        bm25_index_path=data_dir / "bm25.json",
    )


@pytest.fixture
def mock_dependencies(test_settings: Settings) -> Generator[MockCLIDependencies, None, None]:
    """Provide fully mocked CLIDependencies registered as the active test override."""
    sqlite_store: MagicMock = MagicMock()
    vector_store: MagicMock = MagicMock()
    sparse_index: MagicMock = MagicMock()
    embedder: MagicMock = MagicMock()
    reranker: MagicMock = MagicMock()
    searcher: MagicMock = MagicMock()
    sync_engine: MagicMock = MagicMock()
    chunker: MagicMock = MagicMock()
    reader: MagicMock = MagicMock()

    # Sensible defaults for mocks
    sqlite_store.get_document_by_path.return_value = None
    sqlite_store.count_documents.return_value = 0
    sqlite_store.count_sections.return_value = 0
    sqlite_store.count_chunks.return_value = 0
    vector_store.count.return_value = 0

    deps = MockCLIDependencies(
        settings=test_settings,
        sqlite_store=sqlite_store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
        searcher=searcher,
        sync_engine=sync_engine,
        chunker=chunker,
        reader=reader,
    )

    set_dependency_override(deps)
    yield deps
    reset_dependencies()


# ==============================================================================
# 1. DEPENDENCY CONTAINER & FACTORY TESTS
# ==============================================================================


def test_dependency_factory_wiring(test_settings: Settings) -> None:
    """Verify create_dependencies configures all storage and retrieval adapters."""
    deps = create_dependencies(test_settings)
    assert isinstance(deps.sqlite_store, SQLiteStore)
    assert isinstance(deps.vector_store, QdrantVectorStore)
    assert isinstance(deps.sparse_index, BM25Index)
    assert isinstance(deps.embedder, FastEmbedder)
    assert isinstance(deps.reranker, FlashRankReranker)
    assert isinstance(deps.searcher, StrataHybridSearcher)
    assert isinstance(deps.sync_engine, IndexSyncEngine)
    assert isinstance(deps.chunker, MarkdownChunker)
    assert isinstance(deps.reader, LocalFileReader)

    # Verify alias property
    assert deps.hybrid_searcher is deps.searcher


def test_dependency_override_and_reset(test_settings: Settings) -> None:
    """Verify global dependency override mechanism for test isolation."""
    reset_dependencies()
    mock_deps = MagicMock(spec=CLIDependencies)
    set_dependency_override(mock_deps)
    assert get_dependencies(test_settings) is mock_deps

    reset_dependencies()
    fresh_deps = get_dependencies(test_settings)
    assert fresh_deps is not mock_deps
    assert isinstance(fresh_deps.sqlite_store, SQLiteStore)


# ==============================================================================
# 2. INGEST COMMAND TESTS
# ==============================================================================


def test_ingest_single_file_success(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
    tmp_path: Path,
) -> None:
    """Verify single markdown document ingestion commits records to SQLite."""
    test_file = tmp_path / "sample.md"
    test_file.write_text("# Test Title\n\nContent here.", encoding="utf-8")

    sample_doc = Document(
        title="Test Title",
        file_path=str(test_file.resolve()),
        raw_content="# Test Title\n\nContent here.",
        doc_hash="hash-sample-123",
        tags=["initial"],
    )
    sample_section = Section(
        id="sec-1",
        document_id=sample_doc.id,
        title="Test Title",
        level=1,
        heading_path="Test Title",
        content="Content here.",
        index=0,
        token_count=5,
    )
    sample_chunk = Chunk(
        id="chunk-1",
        document_id=sample_doc.id,
        chunk_hash="chunk-hash-1",
        text="Content here.",
        heading_path="Test Title",
        index=0,
        token_count=5,
        tags=["initial"],
    )

    mock_dependencies.reader.read_file.return_value = sample_doc
    mock_dependencies.chunker.extract_sections.return_value = [sample_section]
    mock_dependencies.chunker.chunk_document.return_value = [sample_chunk]

    result = runner.invoke(app, ["ingest", str(test_file), "--tag", "cli-test"])

    assert result.exit_code == 0
    assert "Ingestion Summary" in result.stdout
    assert "Total Documents" in result.stdout
    assert "Total Sections" in result.stdout
    assert "Total Chunks" in result.stdout
    assert "created 1 documents, 1 sections, and 1 chunks" in result.stdout

    mock_dependencies.sqlite_store.add_document.assert_called_once()
    mock_dependencies.sqlite_store.add_sections.assert_called_once_with([sample_section])
    mock_dependencies.sqlite_store.add_chunks.assert_called_once_with([sample_chunk])


def test_ingest_directory_success(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
    tmp_path: Path,
) -> None:
    """Verify directory scan ingestion creates records for multiple documents."""
    doc_dir = tmp_path / "notes"
    doc_dir.mkdir()
    (doc_dir / "note1.md").write_text("# Note 1", encoding="utf-8")
    (doc_dir / "note2.md").write_text("# Note 2", encoding="utf-8")

    doc1 = Document(
        title="Note 1",
        file_path=str((doc_dir / "note1.md").resolve()),
        raw_content="# Note 1",
        doc_hash="hash-1",
    )
    doc2 = Document(
        title="Note 2",
        file_path=str((doc_dir / "note2.md").resolve()),
        raw_content="# Note 2",
        doc_hash="hash-2",
    )

    sec1 = Section(
        id="s1",
        document_id=doc1.id,
        title="Note 1",
        level=1,
        heading_path="Note 1",
        content="",
        index=0,
        token_count=0,
    )
    sec2 = Section(
        id="s2",
        document_id=doc2.id,
        title="Note 2",
        level=1,
        heading_path="Note 2",
        content="",
        index=0,
        token_count=0,
    )

    empty_chunks: list[Chunk] = []
    mock_dependencies.reader.scan_directory.return_value = [doc1, doc2]
    mock_dependencies.chunker.extract_sections.side_effect = [[sec1], [sec2]]
    mock_dependencies.chunker.chunk_document.return_value = empty_chunks

    result = runner.invoke(app, ["ingest", str(doc_dir), "--tag", "docs"])

    assert result.exit_code == 0
    assert "created 2 documents, 2 sections, and 0 chunks" in result.stdout
    assert mock_dependencies.sqlite_store.add_document.call_count == 2


def test_ingest_replaces_existing_document(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
    tmp_path: Path,
) -> None:
    """Verify re-ingesting an updated file replaces the prior record in SQLite."""
    test_file = tmp_path / "existing.md"
    test_file.write_text("# Existing Doc", encoding="utf-8")

    existing_doc = Document(
        id="old-doc-id",
        title="Old Version",
        file_path=str(test_file.resolve()),
        raw_content="# Old Version",
        doc_hash="hash-old",
    )
    new_doc = Document(
        id="new-doc-id",
        title="New Version",
        file_path=str(test_file.resolve()),
        raw_content="# New Version",
        doc_hash="hash-new",
    )

    empty_sections: list[Section] = []
    empty_chunks: list[Chunk] = []
    mock_dependencies.reader.read_file.return_value = new_doc
    mock_dependencies.sqlite_store.get_document_by_path.return_value = existing_doc
    mock_dependencies.chunker.extract_sections.return_value = empty_sections
    mock_dependencies.chunker.chunk_document.return_value = empty_chunks

    result = runner.invoke(app, ["ingest", str(test_file)])

    assert result.exit_code == 0
    mock_dependencies.sqlite_store.delete_document.assert_called_once_with("old-doc-id")
    mock_dependencies.sqlite_store.add_document.assert_called_once_with(new_doc)


def test_ingest_command_alias_add(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
    tmp_path: Path,
) -> None:
    """Verify 'strata add' alias delegates to ingest command correctly."""
    test_file = tmp_path / "add_alias.md"
    test_file.write_text("# Add Alias", encoding="utf-8")

    doc = Document(
        title="Add Alias",
        file_path=str(test_file.resolve()),
        raw_content="# Add Alias",
        doc_hash="hash-alias",
    )
    empty_sections: list[Section] = []
    empty_chunks: list[Chunk] = []
    mock_dependencies.reader.read_file.return_value = doc
    mock_dependencies.chunker.extract_sections.return_value = empty_sections
    mock_dependencies.chunker.chunk_document.return_value = empty_chunks

    result = runner.invoke(app, ["add", str(test_file)])

    assert result.exit_code == 0
    assert "Ingestion complete" in result.stdout


def test_ingest_nonexistent_path_fails(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
    tmp_path: Path,
) -> None:
    """Verify non-existent input path exits with code 1 and no unhandled traceback."""
    bogus_path = tmp_path / "does_not_exist.md"

    result = runner.invoke(app, ["ingest", str(bogus_path)])

    assert result.exit_code == 1
    assert "Path does not exist" in result.stdout
    assert "Traceback (most recent call last)" not in result.output


def test_ingest_empty_directory_fails(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
    tmp_path: Path,
) -> None:
    """Verify empty directory with zero markdown files yields clean error message."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    empty_docs: list[Document] = []
    mock_dependencies.reader.scan_directory.return_value = empty_docs

    result = runner.invoke(app, ["ingest", str(empty_dir)])

    assert result.exit_code == 1
    assert "No markdown documents found" in result.stdout
    assert "Traceback (most recent call last)" not in result.output


def test_ingest_domain_exception_handling(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
    tmp_path: Path,
) -> None:
    """Verify Strata domain errors during ingestion exit non-zero without traceback."""
    test_file = tmp_path / "bad.md"
    test_file.write_text("# Bad", encoding="utf-8")

    mock_dependencies.reader.read_file.side_effect = StrataError("Disk read corruption")

    result = runner.invoke(app, ["ingest", str(test_file)])

    assert result.exit_code == 1
    assert "Strata domain error during ingestion" in result.stdout
    assert "Disk read corruption" in result.stdout
    assert "Traceback (most recent call last)" not in result.output


# ==============================================================================
# 3. SYNC COMMAND TESTS
# ==============================================================================


def test_sync_command_success(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
) -> None:
    """Verify sync command triggers sync engine and renders sync summary."""
    mock_report = SyncStats(
        total_documents=5,
        total_chunks=20,
        new_chunks=4,
        unchanged_chunks=14,
        orphaned_chunks=2,
        elapsed_ms=123.45,
    )
    mock_dependencies.sync_engine.sync.return_value = mock_report

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    assert "Index Synchronization Summary" in result.stdout
    assert "Added Records" in result.stdout
    assert "Updated Records" in result.stdout
    assert "Deleted Records" in result.stdout
    assert "Unchanged Records" in result.stdout
    assert "123.45 ms" in result.stdout
    mock_dependencies.sync_engine.sync.assert_called_once_with(batch_size=None)


def test_sync_command_with_batch_size_option(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
) -> None:
    """Verify sync command forwards custom batch size option."""
    mock_report = SyncStats(total_documents=1, total_chunks=1, elapsed_ms=10.0)
    mock_dependencies.sync_engine.sync.return_value = mock_report

    result = runner.invoke(app, ["sync", "--batch-size", "16"])

    assert result.exit_code == 0
    mock_dependencies.sync_engine.sync.assert_called_once_with(batch_size=16)


def test_sync_conflict_error_handling(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
) -> None:
    """Verify SyncConflictError produces non-zero exit code and no traceback."""
    mock_dependencies.sync_engine.sync.side_effect = SyncConflictError(
        "Differential hash conflict detected"
    )

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 1
    assert "Synchronization conflict" in result.stdout
    assert "Differential hash conflict detected" in result.stdout
    assert "Traceback (most recent call last)" not in result.output


def test_sync_unexpected_exception_handling(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
) -> None:
    """Verify unexpected runtime errors in sync exit with code 1 without traceback."""
    mock_dependencies.sync_engine.sync.side_effect = RuntimeError("Database locked")

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 1
    assert "Unexpected error during sync" in result.stdout
    assert "Database locked" in result.stdout
    assert "Traceback (most recent call last)" not in result.output


# ==============================================================================
# 4. SEARCH COMMAND TESTS
# ==============================================================================


def test_search_command_success_with_panels(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
) -> None:
    """Verify search command renders matching chunks, heading paths, context, and scores."""
    sample_chunk = Chunk(
        id="chunk-42",
        document_id="doc-1",
        chunk_hash="chash-42",
        text="Raft leader election mechanism involves randomized election timeouts.",
        heading_path="Architecture > Consensus > Leader Election",
        index=2,
        token_count=12,
        tags=["consensus"],
    )
    retrieved_chunk = RetrievedChunk(
        chunk=sample_chunk,
        score=0.9254,
        document_title="Distributed Systems Guide",
        parent_context="Comprehensive consensus protocol specification.",
        rerank_score=0.9254,
    )
    parent_sec = Section(
        id="sec-1",
        document_id="doc-1",
        title="Leader Election",
        level=2,
        heading_path="Architecture > Consensus > Leader Election",
        content="Comprehensive consensus protocol specification.",
        index=1,
        token_count=100,
    )
    adj_chunk = RetrievedChunk(
        chunk=Chunk(
            id="chunk-43",
            document_id="doc-1",
            chunk_hash="chash-43",
            text="Following election, heartbeats maintain authority.",
            heading_path="Architecture > Consensus > Leader Election",
            index=3,
            token_count=8,
        ),
        score=0.0,
    )

    search_result = SearchResult(
        query="leader election",
        chunk=retrieved_chunk,
        score=0.9254,
        parent_section=parent_sec,
        adjacent_chunks=[adj_chunk],
        total_latency_ms=18.5,
    )
    mock_dependencies.searcher.search.return_value = [search_result]

    result = runner.invoke(
        app,
        [
            "search",
            "leader election",
            "--limit",
            "3",
            "--tag",
            "consensus",
            "--expand-section",
            "--window-size",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Matching Chunk" in result.stdout
    assert "Raft leader election mechanism" in result.stdout
    assert "Heading Path:" in result.stdout
    assert "Architecture > Consensus > Leader Election" in result.stdout
    assert "Parent Section Context" in result.stdout
    assert "Comprehensive consensus protocol specification." in result.stdout
    assert "Rerank Score:" in result.stdout
    assert "0.9254" in result.stdout
    assert "Adjacent Context Chunks" in result.stdout

    mock_dependencies.searcher.search.assert_called_once_with(
        query="leader election",
        limit=3,
        tags=["consensus"],
        expand_section=True,
        window_size=1,
    )


def test_search_command_no_results(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
) -> None:
    """Verify search displaying no results exits 0 with informative message."""
    empty_results: list[SearchResult] = []
    mock_dependencies.searcher.search.return_value = empty_results

    result = runner.invoke(app, ["search", "nonexistent term"])

    assert result.exit_code == 0
    assert "No matching chunks found for query" in result.stdout


def test_search_blank_query_fails(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
) -> None:
    """Verify blank or whitespace-only query fails with code 1."""
    result = runner.invoke(app, ["search", "   "])

    assert result.exit_code == 1
    assert "Search query cannot be empty or blank" in result.stdout
    assert "Traceback (most recent call last)" not in result.output


def test_search_invalid_limit_fails(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
) -> None:
    """Verify limit < 1 fails with validation error message."""
    result = runner.invoke(app, ["search", "valid query", "--limit", "0"])

    assert result.exit_code == 1
    assert "Limit must be >= 1" in result.stdout
    assert "Traceback (most recent call last)" not in result.output


def test_search_invalid_window_size_fails(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
) -> None:
    """Verify negative window size fails with validation error message."""
    result = runner.invoke(app, ["search", "valid query", "--window-size", "-2"])

    assert result.exit_code == 1
    assert "Window size must be >= 0" in result.stdout
    assert "Traceback (most recent call last)" not in result.output


def test_search_model_inference_error_handling(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
) -> None:
    """Verify ModelInferenceError in search exits with code 1 and clean error panel."""
    mock_dependencies.searcher.search.side_effect = ModelInferenceError(
        "ONNX Runtime execution failed"
    )

    result = runner.invoke(app, ["search", "test query"])

    assert result.exit_code == 1
    assert "Model inference failed" in result.stdout
    assert "ONNX Runtime execution failed" in result.stdout
    assert "Traceback (most recent call last)" not in result.output


def test_search_strata_error_handling(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
) -> None:
    """Verify Strata domain errors exit non-zero without traceback."""
    mock_dependencies.searcher.search.side_effect = StrataError("Storage hydration timeout")

    result = runner.invoke(app, ["search", "test query"])

    assert result.exit_code == 1
    assert "Strata search error" in result.stdout
    assert "Traceback (most recent call last)" not in result.output


# ==============================================================================
# 5. STATUS COMMAND TESTS
# ==============================================================================


def test_status_command_success_with_bm25_file(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
    test_settings: Settings,
) -> None:
    """Verify status command queries SQLite, Qdrant, and BM25 disk file."""
    # Create dummy BM25 file on disk
    assert test_settings.bm25_index_path is not None
    test_settings.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
    test_settings.bm25_index_path.write_text('{"chunks": []}', encoding="utf-8")

    mock_dependencies.sqlite_store.count_documents.return_value = 8
    mock_dependencies.sqlite_store.count_sections.return_value = 24
    mock_dependencies.sqlite_store.count_chunks.return_value = 75
    mock_dependencies.vector_store.count.return_value = 75

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Strata System Status & Telemetry" in result.stdout
    assert "Total Documents" in result.stdout
    assert "8" in result.stdout
    assert "Total Sections" in result.stdout
    assert "24" in result.stdout
    assert "Total Chunks" in result.stdout
    assert "75" in result.stdout
    assert "Active Vector Points" in result.stdout
    assert "Active (persisted)" in result.stdout


def test_status_command_success_with_bm25_missing(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
    test_settings: Settings,
) -> None:
    """Verify status command gracefully indicates unindexed BM25 file."""
    assert test_settings.bm25_index_path is not None
    if test_settings.bm25_index_path.exists():
        test_settings.bm25_index_path.unlink()

    mock_dependencies.sqlite_store.count_documents.return_value = 0
    mock_dependencies.sqlite_store.count_sections.return_value = 0
    mock_dependencies.sqlite_store.count_chunks.return_value = 0
    mock_dependencies.vector_store.count.return_value = 0

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Not found (unindexed)" in result.stdout
    assert "0 bytes" in result.stdout


def test_status_command_strata_error_handling(
    runner: CliRunner,
    mock_dependencies: MockCLIDependencies,
) -> None:
    """Verify error during status inspection exits with code 1 without traceback."""
    mock_dependencies.sqlite_store.count_documents.side_effect = StrataError(
        "Corrupted database file"
    )

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 1
    assert "Failed to query system status" in result.stdout
    assert "Corrupted database file" in result.stdout
    assert "Traceback (most recent call last)" not in result.output


# ==============================================================================
# 6. END-TO-END INTEGRATION TEST WITH REAL STORAGE ADAPTERS
# ==============================================================================


def test_end_to_end_cli_lifecycle(
    runner: CliRunner,
    test_settings: Settings,
    tmp_path: Path,
) -> None:
    """Verify end-to-end ingest and status using real SQLiteStore and MarkdownChunker."""
    reset_dependencies()

    # Create real SQLiteStore and real chunker/reader
    real_store = SQLiteStore(db_path=test_settings.db_path)
    real_chunker = MarkdownChunker(settings=test_settings, min_chunk_size=5)
    real_reader = LocalFileReader()

    mock_vector: MagicMock = MagicMock()
    mock_vector.count.return_value = 0
    mock_sparse: MagicMock = MagicMock()
    mock_embed: MagicMock = MagicMock()
    mock_rerank: MagicMock = MagicMock()
    mock_search: MagicMock = MagicMock()
    mock_sync: MagicMock = MagicMock()

    real_deps = MockCLIDependencies(
        settings=test_settings,
        sqlite_store=cast(MagicMock, real_store),
        vector_store=mock_vector,
        sparse_index=mock_sparse,
        embedder=mock_embed,
        reranker=mock_rerank,
        searcher=mock_search,
        sync_engine=mock_sync,
        chunker=cast(MagicMock, real_chunker),
        reader=cast(MagicMock, real_reader),
    )
    set_dependency_override(real_deps)

    try:
        # Create a real markdown document
        md_file = tmp_path / "article.md"
        md_file.write_text(
            "# Hybrid Search in Strata\n\n"
            "## Architecture\n\n"
            "Strata combines Qdrant and BM25 using Reciprocal Rank Fusion.\n\n"
            "## Storage\n\n"
            "SQLite is the canonical source of truth.\n",
            encoding="utf-8",
        )

        # 1. Ingest
        ingest_res = runner.invoke(app, ["ingest", str(md_file), "--tag", "architecture"])
        assert ingest_res.exit_code == 0
        assert "created 1 documents" in ingest_res.stdout

        # Verify real SQLite database state
        assert real_store.count_documents() == 1
        assert real_store.count_sections() >= 2
        assert real_store.count_chunks() >= 2

        # 2. Check Status
        status_res = runner.invoke(app, ["status"])
        assert status_res.exit_code == 0
        assert "Total Documents" in status_res.stdout
        assert "Total Sections" in status_res.stdout
        assert "Total Chunks" in status_res.stdout

    finally:
        real_store.close()
        reset_dependencies()
