"""Unit tests for strata.core models, configuration, and protocols."""

from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from strata.core.config import Settings, get_settings
from strata.core.models import (
    AskResult,
    BenchmarkSample,
    BenchmarkScorecard,
    Chunk,
    ChunkDiff,
    Citation,
    Document,
    DocumentOutline,
    DocumentOutlineItem,
    ExpandedContext,
    RetrievedChunk,
    SearchResult,
    Section,
    SyncStats,
)
from strata.core.protocols import (
    Chunker,
    DocumentStore,
    Embedder,
    HybridSearcher,
    LLMGenerator,
    Reranker,
    SparseIndex,
    VectorStore,
)


def test_settings_defaults() -> None:
    """Verify default configuration values."""
    settings = get_settings()
    assert settings.collection_name == "strata_chunks"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_dimension == 384
    assert settings.reranker_model == "BAAI/bge-reranker-base"
    assert settings.chunk_size == 350
    assert settings.dense_top_k == 20
    assert settings.sparse_top_k == 20
    assert settings.rrf_k == 60
    assert settings.final_top_k == 3
    assert settings.data_dir == Path("./data")
    assert settings.db_path == Path("./data/canonical.db")
    assert settings.qdrant_path == Path("./data/qdrant")


def test_settings_custom_values() -> None:
    """Verify settings can be instantiated with custom values."""
    custom = Settings(
        chunk_size=500,
        final_top_k=5,
        llm_model="llama3.2:3b",
    )
    assert custom.chunk_size == 500
    assert custom.final_top_k == 5
    assert custom.llm_model == "llama3.2:3b"


def test_document_creation_and_immutability() -> None:
    """Verify Document model initialization, default values, and immutability."""
    doc = Document(
        title="Test Document",
        file_path="/path/to/test.md",
        raw_content="# Heading\nHello world.",
        doc_hash="a1b2c3d4",
        tags=["unit-test", "architecture"],
    )
    assert doc.title == "Test Document"
    assert len(doc.id) > 0
    assert doc.tags == ["unit-test", "architecture"]
    assert doc.metadata == {}

    # Verify immutability (frozen=True)
    with pytest.raises(ValidationError):
        setattr(doc, "title", "New Title")  # noqa: B010


def test_document_extra_fields_forbidden() -> None:
    """Verify extra fields cannot be added to Document."""
    with pytest.raises(ValidationError):
        Document.model_validate(
            {
                "title": "Invalid",
                "file_path": "invalid.md",
                "raw_content": "content",
                "doc_hash": "hash",
                "unsupported_field": "fail",
            }
        )


def test_section_validation() -> None:
    """Verify Section level bounds and properties."""
    sec = Section(
        document_id="doc-123",
        title="Architecture",
        level=2,
        heading_path="Overview > Architecture",
        content="Architecture details...",
        index=0,
        token_count=120,
    )
    assert sec.level == 2
    assert sec.heading_path == "Overview > Architecture"

    # Level must be between 1 and 6
    with pytest.raises(ValidationError):
        Section.model_validate(
            {
                "document_id": "doc-123",
                "title": "Invalid Level",
                "level": 0,
                "heading_path": "Path",
                "content": "Text",
                "index": 0,
            }
        )

    with pytest.raises(ValidationError):
        Section.model_validate(
            {
                "document_id": "doc-123",
                "title": "Invalid Level",
                "level": 7,
                "heading_path": "Path",
                "content": "Text",
                "index": 0,
            }
        )


def test_chunk_creation() -> None:
    """Verify Chunk domain model creation and properties."""
    chunk = Chunk(
        document_id="doc-123",
        chunk_hash="sha256hashvalue",
        text="Sample chunk content text.",
        heading_path="Overview > Architecture",
        index=0,
        parent_section_id="sec-456",
        token_count=50,
        tags=["tech"],
    )
    assert chunk.document_id == "doc-123"
    assert chunk.parent_section_id == "sec-456"
    assert chunk.chunk_hash == "sha256hashvalue"


def test_chunk_diff() -> None:
    """Verify ChunkDiff holds new, unchanged, and orphaned collections."""
    c1 = Chunk(
        document_id="doc-1",
        chunk_hash="h1",
        text="c1",
        heading_path="H1",
        index=0,
    )
    diff = ChunkDiff(
        document_id="doc-1",
        new_chunks=[c1],
        unchanged_hashes=["h2", "h3"],
        orphaned_hashes=["h0"],
    )
    assert len(diff.new_chunks) == 1
    assert diff.unchanged_hashes == ["h2", "h3"]
    assert diff.orphaned_hashes == ["h0"]


def test_sync_stats() -> None:
    """Verify SyncStats default values and constraints."""
    stats = SyncStats(
        total_documents=5,
        total_chunks=50,
        new_chunks=10,
        unchanged_chunks=40,
        orphaned_chunks=2,
        elapsed_ms=145.2,
    )
    assert stats.total_documents == 5
    assert stats.new_chunks == 10
    assert stats.elapsed_ms == 145.2


def test_retrieved_chunk_and_search_result() -> None:
    """Verify RetrievedChunk and SearchResult structures."""
    chunk = Chunk(
        document_id="doc-1",
        chunk_hash="h1",
        text="Distributed consensus rules",
        heading_path="Consensus",
        index=0,
    )
    retrieved = RetrievedChunk(
        chunk=chunk,
        score=0.95,
        dense_rank=1,
        sparse_rank=2,
        dense_score=0.88,
        sparse_score=14.2,
        rrf_score=0.032,
        rerank_score=0.95,
        parent_context="Full consensus section",
        document_title="Raft Notes",
    )
    assert retrieved.score == 0.95
    assert retrieved.rerank_score == 0.95
    assert retrieved.parent_context == "Full consensus section"

    result = SearchResult(
        query="consensus rules",
        chunks=[retrieved],
        dense_count=1,
        sparse_count=1,
        total_latency_ms=45.0,
    )
    assert len(result.chunks) == 1
    assert result.total_latency_ms == 45.0


def test_citation_formatting() -> None:
    """Verify Citation string representation follows [Doc: <title>, § <heading>]."""
    citation = Citation(
        document_title="raft-notes.md",
        heading_path="Leader Election Mechanics",
        chunk_id="chunk-123",
        source_index=1,
    )
    assert citation.format_tag() == "[Doc: raft-notes.md, § Leader Election Mechanics]"


def test_ask_result() -> None:
    """Verify AskResult initialization."""
    citation = Citation(
        document_title="raft-notes.md",
        heading_path="Leader Election",
        chunk_id="chunk-123",
        source_index=1,
    )
    result = AskResult(
        query="How does leader election work?",
        answer="Leaders send periodic heartbeats [Doc: raft-notes.md, § Leader Election].",
        citations=[citation],
        latency_ms=1200.5,
        time_to_first_token_ms=350.0,
        tokens_generated=85,
        tokens_per_second=70.8,
    )
    assert result.tokens_generated == 85
    assert len(result.citations) == 1
    assert result.time_to_first_token_ms == 350.0


def test_outline_and_expanded_context() -> None:
    """Verify agent tools models."""
    item = DocumentOutlineItem(
        level=1,
        title="Intro",
        heading_path="Intro",
        section_id="sec-1",
        token_count=100,
    )
    outline = DocumentOutline(
        document_id="doc-1",
        document_title="Doc Title",
        items=[item],
    )
    assert len(outline.items) == 1

    c0 = Chunk(document_id="doc-1", chunk_hash="h0", text="p", heading_path="H", index=0)
    c1 = Chunk(document_id="doc-1", chunk_hash="h1", text="target", heading_path="H", index=1)
    c2 = Chunk(document_id="doc-1", chunk_hash="h2", text="s", heading_path="H", index=2)

    expanded = ExpandedContext(
        target_chunk_id="c1",
        document_id="doc-1",
        window=1,
        preceding_chunks=[c0],
        target_chunk=c1,
        succeeding_chunks=[c2],
        combined_text="p\ntarget\ns",
    )
    assert expanded.combined_text == "p\ntarget\ns"


def test_benchmark_models() -> None:
    """Verify BenchmarkSample and BenchmarkScorecard models."""
    sample = BenchmarkSample(
        query="What is Strata?",
        ground_truth="An air-gapped personal knowledge engine.",
        contexts=["Strata is an air-gapped knowledge base."],
    )
    assert sample.query == "What is Strata?"

    scorecard = BenchmarkScorecard(
        retrieval_strategy="Hybrid (RRF) + bge-reranker-base",
        reranker_model="BAAI/bge-reranker-base",
        context_precision=0.94,
        context_recall=0.92,
        faithfulness=0.96,
        answer_relevance=0.95,
        avg_retrieval_latency_ms=158.0,
        sample_count=50,
    )
    assert scorecard.context_precision == 0.94
    assert scorecard.faithfulness == 0.96


def test_runtime_protocol_checks() -> None:
    """Verify runtime structural subtyping checkability for Protocols."""

    class DummyDocumentStore:
        def add_document(self, document: Document) -> None:
            pass

        def get_document(self, document_id: str) -> Document | None:
            return None

        def get_document_by_path(self, file_path: str) -> Document | None:
            return None

        def list_documents(self, tags: list[str] | None = None) -> list[Document]:
            return []

        def update_document(self, document: Document) -> None:
            pass

        def delete_document(self, document_id: str) -> None:
            pass

        def add_sections(self, sections: list[Section]) -> None:
            pass

        def get_section(self, section_id: str) -> Section | None:
            return None

        def get_sections_by_document_id(self, document_id: str) -> list[Section]:
            return []

        def add_chunks(self, chunks: list[Chunk]) -> None:
            pass

        def get_chunk(self, chunk_id: str) -> Chunk | None:
            return None

        def get_chunks_by_document_id(self, document_id: str) -> list[Chunk]:
            return []

        def get_all_chunks(self) -> list[Chunk]:
            return []

        def get_active_chunk_hashes(self, document_id: str) -> list[str]:
            return []

        def delete_chunks_by_hashes(self, chunk_hashes: list[str]) -> None:
            pass

        def get_document_outline(self, document_id: str) -> DocumentOutline:
            return DocumentOutline(document_id=document_id, document_title="Outline", items=[])

        def get_adjacent_chunks(self, chunk_id: str, window: int = 1) -> list[Chunk]:
            return []

        def count_documents(self) -> int:
            return 0

        def count_chunks(self) -> int:
            return 0

    assert isinstance(DummyDocumentStore(), DocumentStore)

    class DummyVectorStore:
        def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
            pass

        def delete_by_document_id(self, document_id: str) -> None:
            pass

        def delete_by_chunk_hashes(self, chunk_hashes: list[str]) -> None:
            pass

        def search(
            self,
            query_vector: list[float],
            limit: int,
            payload_filter: Mapping[str, object] | None = None,
        ) -> list[RetrievedChunk]:
            return []

        def count(self) -> int:
            return 0

    assert isinstance(DummyVectorStore(), VectorStore)

    class DummyEmbedder:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.0]]

        def embed_query(self, query: str) -> list[float]:
            return [0.0]

        @property
        def dimension(self) -> int:
            return 384

    assert isinstance(DummyEmbedder(), Embedder)

    class DummySparseIndex:
        def index_chunks(self, chunks: list[Chunk]) -> None:
            pass

        def search(self, query: str, limit: int) -> list[RetrievedChunk]:
            return []

        def clear(self) -> None:
            pass

        @property
        def count(self) -> int:
            return 0

    assert isinstance(DummySparseIndex(), SparseIndex)

    class DummyReranker:
        def rerank(
            self, query: str, candidates: list[RetrievedChunk], top_k: int
        ) -> list[RetrievedChunk]:
            return candidates[:top_k]

    assert isinstance(DummyReranker(), Reranker)

    class DummyLLMGenerator:
        def generate(self, prompt: str, system_prompt: str | None = None) -> str:
            return "response"

        def stream(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
            yield "token"

    assert isinstance(DummyLLMGenerator(), LLMGenerator)

    class DummyChunker:
        def chunk_document(self, document: Document) -> list[Chunk]:
            return []

        def extract_sections(self, document: Document) -> list[Section]:
            return []

    assert isinstance(DummyChunker(), Chunker)

    class DummyHybridSearcher:
        def search(
            self,
            query: str,
            top_k: int | None = None,
            payload_filter: Mapping[str, object] | None = None,
        ) -> SearchResult:
            return SearchResult(query=query, chunks=[])

    assert isinstance(DummyHybridSearcher(), HybridSearcher)

    # Incomplete class must fail isinstance
    class IncompleteStore:
        def count_documents(self) -> int:
            return 0

    assert not isinstance(IncompleteStore(), DocumentStore)
