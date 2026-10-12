"""Unit test suite for Strata hybrid retrieval coordinator (StrataHybridSearcher).

Tests verify:
- Mandatory injection of Reranker protocol implementation
- End-to-end hybrid retrieval across dense ANN, sparse BM25, RRF fusion, and cross-encoder reranking
- Final reranking score assignment and winner reordering
- Small-to-Big context expansion (Section hydration with deduplication caching)
- Sliding window sibling chunk expansion
- Payload filtering and tag forwarding to vector store
- Validation guardrails and edge-case handling (empty query, bad limits)
- Data desync tolerance (missing chunk in SQLite)
- Settings default resolution (final_top_k)
- Protocol structural subtyping conformance
- End-to-end latency calculation inside coordinator
"""

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest

from strata.core.config import Settings
from strata.core.exceptions import ModelInferenceError
from strata.core.models import (
    Chunk,
    Document,
    DocumentOutline,
    RetrievedChunk,
    SearchResult,
    Section,
)
from strata.core.protocols import (
    DocumentStore,
    Embedder,
    HybridSearcher,
    Reranker,
    SparseIndex,
    VectorStore,
)
from strata.retrieval.hybrid_search import StrataHybridSearcher

# ==============================================================================
# In-Memory Test Adapters
# ==============================================================================


class InMemoryDocumentStore(DocumentStore):
    """In-memory SQLite canonical storage mock."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.sections: dict[str, Section] = {}
        self.chunks: dict[str, Chunk] = {}
        self.recorded_section_fetches: list[str] = []

    def add_document(self, document: Document) -> None:
        self.documents[document.id] = document

    def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    def get_document_by_path(self, file_path: str) -> Document | None:
        return next((d for d in self.documents.values() if d.file_path == file_path), None)

    def list_documents(self, tags: list[str] | None = None) -> list[Document]:
        return list(self.documents.values())

    def update_document(self, document: Document) -> None:
        self.documents[document.id] = document

    def delete_document(self, document_id: str) -> None:
        self.documents.pop(document_id, None)

    def add_sections(self, sections: list[Section]) -> None:
        for s in sections:
            self.sections[s.id] = s

    def get_section(self, section_id: str) -> Section | None:
        self.recorded_section_fetches.append(section_id)
        return self.sections.get(section_id)

    def get_section_by_id(self, section_id: str) -> Section | None:
        self.recorded_section_fetches.append(section_id)
        return self.sections.get(section_id)

    def get_sections_by_document_id(self, document_id: str) -> list[Section]:
        return [s for s in self.sections.values() if s.document_id == document_id]

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self.chunks[c.id] = c

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def get_chunks_by_document_id(self, document_id: str) -> list[Chunk]:
        return [c for c in self.chunks.values() if c.document_id == document_id]

    def get_all_chunks(self) -> list[Chunk]:
        return list(self.chunks.values())

    def get_active_chunk_hashes(self, document_id: str) -> list[str]:
        return [c.chunk_hash for c in self.chunks.values() if c.document_id == document_id]

    def delete_chunks_by_hashes(self, chunk_hashes: list[str]) -> None:
        hashes = set(chunk_hashes)
        self.chunks = {cid: c for cid, c in self.chunks.items() if c.chunk_hash not in hashes}

    def get_document_outline(self, document_id: str) -> DocumentOutline:
        return DocumentOutline(document_id=document_id, document_title="Mock Doc", items=[])

    def get_adjacent_chunks(self, chunk_id: str, window: int = 1) -> list[Chunk]:
        target = self.chunks.get(chunk_id)
        if target is None:
            return []
        min_idx = target.index - window
        max_idx = target.index + window
        matched = [
            c
            for c in self.chunks.values()
            if c.document_id == target.document_id and min_idx <= c.index <= max_idx
        ]
        return sorted(matched, key=lambda c: c.index)

    def count_documents(self) -> int:
        return len(self.documents)

    def count_chunks(self) -> int:
        return len(self.chunks)


class InMemoryVectorStore(VectorStore):
    """In-memory VectorStore mock recording queries and returning preset hits."""

    def __init__(self) -> None:
        self.results_to_return: list[RetrievedChunk] = []
        self.recorded_calls: list[dict[str, Any]] = []

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
        tags: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.recorded_calls.append(
            {
                "query_vector": query_vector,
                "limit": limit,
                "payload_filter": payload_filter,
                "tags": tags,
            }
        )
        return self.results_to_return[:limit]

    def count(self) -> int:
        return len(self.results_to_return)


class InMemorySparseIndex(SparseIndex):
    """In-memory BM25 sparse index mock recording queries and returning preset hits."""

    def __init__(self) -> None:
        self.results_to_return: list[RetrievedChunk] = []
        self.recorded_calls: list[dict[str, Any]] = []

    def index_chunks(self, chunks: list[Chunk]) -> None:
        pass

    def search(self, query: str, limit: int) -> list[RetrievedChunk]:
        self.recorded_calls.append({"query": query, "limit": limit})
        return self.results_to_return[:limit]

    def clear(self) -> None:
        self.results_to_return.clear()

    @property
    def count(self) -> int:
        return len(self.results_to_return)


class InMemoryEmbedder(Embedder):
    """In-memory dense embedder mock."""

    def __init__(self, dimension: int = 4) -> None:
        self._dim = dimension
        self.recorded_queries: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dim for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        self.recorded_queries.append(query)
        return [0.5] * self._dim

    def embed_text(self, text: str) -> list[float]:
        self.recorded_queries.append(text)
        return [0.5] * self._dim

    @property
    def dimension(self) -> int:
        return self._dim


class InMemoryReranker(Reranker):
    """In-memory cross-encoder reranker mock for testing."""

    def __init__(self, score_offset: float = 0.0) -> None:
        self.score_offset: float = score_offset
        self.recorded_calls: list[dict[str, Any]] = []
        self.custom_scores: dict[str, float] | None = None

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        self.recorded_calls.append(
            {
                "query": query,
                "candidates": candidates,
                "top_k": top_k,
            }
        )
        limit = top_k if top_k is not None else len(candidates)
        reranked: list[RetrievedChunk] = []
        for c in candidates:
            if self.custom_scores is not None and c.chunk.id in self.custom_scores:
                new_score = self.custom_scores[c.chunk.id]
            else:
                new_score = c.score + self.score_offset
            reranked.append(
                c.model_copy(
                    update={
                        "score": new_score,
                        "rerank_score": new_score,
                    }
                )
            )

        # Sort descending by score, tie-break by chunk ID ascending
        reranked.sort(key=lambda item: (-item.score, item.chunk.id))
        return reranked[:limit]


# ==============================================================================
# Test Fixtures & Helper Factory
# ==============================================================================


def create_sample_chunk(
    chunk_id: str,
    doc_id: str = "doc-1",
    index: int = 0,
    parent_section_id: str | None = None,
    text: str = "Sample chunk text",
    heading_path: str = "Section > Sub",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=doc_id,
        chunk_hash=f"hash-{chunk_id}",
        text=text,
        heading_path=heading_path,
        index=index,
        parent_section_id=parent_section_id,
        token_count=20,
    )


def create_sample_section(
    section_id: str,
    doc_id: str = "doc-1",
    title: str = "Architecture",
    content: str = "Full architecture section body.",
) -> Section:
    return Section(
        id=section_id,
        document_id=doc_id,
        title=title,
        level=2,
        heading_path=f"Root > {title}",
        content=content,
        index=0,
        token_count=150,
    )


# ==============================================================================
# 1. End-to-End Hybrid Search & Reranking Flow
# ==============================================================================


def test_end_to_end_hybrid_search() -> None:
    """Verify query flows through dense ANN, sparse BM25, RRF, and mandatory reranker."""
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    sparse_index = InMemorySparseIndex()
    embedder = InMemoryEmbedder(dimension=4)
    reranker = InMemoryReranker()

    # Prepare chunks in SQLite
    chunk_a = create_sample_chunk("chunk-a", index=0, text="Raft leader election protocol")
    chunk_b = create_sample_chunk("chunk-b", index=1, text="Paxos distributed consensus")
    store.add_chunks([chunk_a, chunk_b])

    # Dense leg ranks: chunk-a (score 0.95), chunk-b (score 0.80)
    vector_store.results_to_return = [
        RetrievedChunk(chunk=chunk_a, score=0.95),
        RetrievedChunk(chunk=chunk_b, score=0.80),
    ]

    # Sparse leg ranks: chunk-b (score 12.0), chunk-a (score 8.0)
    sparse_index.results_to_return = [
        RetrievedChunk(chunk=chunk_b, score=12.0),
        RetrievedChunk(chunk=chunk_a, score=8.0),
    ]

    settings = Settings(fused_top_k=5, rrf_k=60)
    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
        settings=settings,
    )

    results = searcher.search(query="consensus algorithms", top_k=2)

    # 1. Verify query passed to embedder
    assert embedder.recorded_queries == ["consensus algorithms"]

    # 2. Verify query passed to vector store with candidate depth
    assert len(vector_store.recorded_calls) == 1
    assert vector_store.recorded_calls[0]["query_vector"] == [0.5, 0.5, 0.5, 0.5]
    assert vector_store.recorded_calls[0]["limit"] == 20

    # 3. Verify query passed to sparse index with candidate depth
    assert len(sparse_index.recorded_calls) == 1
    assert sparse_index.recorded_calls[0]["query"] == "consensus algorithms"
    assert sparse_index.recorded_calls[0]["limit"] == 20

    # 4. Verify candidates were passed into reranker
    assert len(reranker.recorded_calls) == 1
    assert reranker.recorded_calls[0]["query"] == "consensus algorithms"
    assert reranker.recorded_calls[0]["top_k"] == 2

    # 5. Verify RRF fusion merged both and reranker returned hydrated chunks
    assert len(results) == 2
    assert results[0].chunk is not None
    assert results[1].chunk is not None
    assert results[0].chunk.id == "chunk-a"
    assert results[1].chunk.id == "chunk-b"
    assert isinstance(results[0], SearchResult)
    assert results[0].score == pytest.approx(1.0 / 61 + 1.0 / 62)
    assert results[0].total_latency_ms >= 0.0


def test_reranker_reorders_fused_candidates_and_assigns_scores() -> None:
    """Verify cross-encoder reranker overrides initial RRF ordering and assigns final scores."""
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    sparse_index = InMemorySparseIndex()
    embedder = InMemoryEmbedder()

    c_leader = create_sample_chunk("c-leader", text="Leader election heartbeat")
    c_log = create_sample_chunk("c-log", text="Log compaction snapshot")
    store.add_chunks([c_leader, c_log])

    # Initial dense & sparse retrieval strongly favors c_leader over c_log
    vector_store.results_to_return = [
        RetrievedChunk(chunk=c_leader, score=0.99),
        RetrievedChunk(chunk=c_log, score=0.50),
    ]

    # Reranker assigns high score to c_log and low score to c_leader
    reranker = InMemoryReranker()
    reranker.custom_scores = {
        "c-leader": 0.15,
        "c-log": 0.94,
    }

    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
    )

    results = searcher.search(query="snapshot log", top_k=2)

    assert len(results) == 2
    # c-log was promoted to 1st position by reranker
    assert results[0].chunk is not None
    assert results[0].chunk.id == "c-log"
    assert results[0].score == 0.94
    assert results[0].chunk.rerank_score == 0.94

    # c-leader was demoted to 2nd position
    assert results[1].chunk is not None
    assert results[1].chunk.id == "c-leader"
    assert results[1].score == 0.15
    assert results[1].chunk.rerank_score == 0.15


# ==============================================================================
# 2. Small-to-Big Hydration & Parent Section Deduplication
# ==============================================================================


def test_small_to_big_hydration_enabled() -> None:
    """When expand_section=True, SearchResult.parent_section is populated with matching Section."""
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    sparse_index = InMemorySparseIndex()
    embedder = InMemoryEmbedder()
    reranker = InMemoryReranker()

    section = create_sample_section("sec-42", title="Raft Overview")
    chunk = create_sample_chunk("chunk-1", parent_section_id="sec-42")

    store.add_sections([section])
    store.add_chunks([chunk])

    vector_store.results_to_return = [RetrievedChunk(chunk=chunk, score=0.9)]

    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
    )

    results = searcher.search(query="raft", expand_section=True)

    assert len(results) == 1
    assert results[0].parent_section is not None
    assert results[0].parent_section.id == "sec-42"
    assert results[0].parent_section.title == "Raft Overview"
    assert results[0].chunk is not None
    assert results[0].chunk.parent_section == results[0].parent_section


def test_small_to_big_hydration_disabled() -> None:
    """When expand_section=False, parent_section is None even if chunk has parent_section_id."""
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    sparse_index = InMemorySparseIndex()
    embedder = InMemoryEmbedder()
    reranker = InMemoryReranker()

    section = create_sample_section("sec-42", title="Raft Overview")
    chunk = create_sample_chunk("chunk-1", parent_section_id="sec-42")

    store.add_sections([section])
    store.add_chunks([chunk])

    vector_store.results_to_return = [RetrievedChunk(chunk=chunk, score=0.9)]

    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
    )

    results = searcher.search(query="raft", expand_section=False)

    assert len(results) == 1
    assert results[0].parent_section is None


def test_section_caching_for_shared_parent_section() -> None:
    """When multiple chunks share the same parent section, fetch from SQLite only once."""
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    sparse_index = InMemorySparseIndex()
    embedder = InMemoryEmbedder()
    reranker = InMemoryReranker()

    shared_section = create_sample_section("sec-shared", title="Consensus Module")
    c1 = create_sample_chunk("c1", parent_section_id="sec-shared")
    c2 = create_sample_chunk("c2", parent_section_id="sec-shared")
    c3 = create_sample_chunk("c3", parent_section_id="sec-shared")

    store.add_sections([shared_section])
    store.add_chunks([c1, c2, c3])

    vector_store.results_to_return = [
        RetrievedChunk(chunk=c1, score=0.9),
        RetrievedChunk(chunk=c2, score=0.8),
        RetrievedChunk(chunk=c3, score=0.7),
    ]

    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
    )

    results = searcher.search(query="consensus", expand_section=True, top_k=3)

    assert len(results) == 3
    for r in results:
        assert r.parent_section is not None
        assert r.parent_section.id == "sec-shared"

    # SQLite store must have been queried exactly once for sec-shared due to section_cache
    assert store.recorded_section_fetches == ["sec-shared"]


def test_small_to_big_hydration_missing_section_or_none_parent() -> None:
    """When chunk has no parent_section_id or section is missing, parent_section is None."""
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    sparse_index = InMemorySparseIndex()
    embedder = InMemoryEmbedder()
    reranker = InMemoryReranker()

    chunk_without_parent = create_sample_chunk("c-no-parent", parent_section_id=None)
    chunk_with_missing_parent = create_sample_chunk("c-missing-parent", parent_section_id="ghost")

    store.add_chunks([chunk_without_parent, chunk_with_missing_parent])

    vector_store.results_to_return = [
        RetrievedChunk(chunk=chunk_without_parent, score=0.9),
        RetrievedChunk(chunk=chunk_with_missing_parent, score=0.8),
    ]

    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
    )

    results = searcher.search(query="test", expand_section=True)
    assert len(results) == 2
    assert results[0].parent_section is None
    assert results[1].parent_section is None


# ==============================================================================
# 3. Sliding Window Expansion
# ==============================================================================


def test_sliding_window_expansion_enabled() -> None:
    """When window_size=1, SearchResult.adjacent_chunks contains sibling chunks."""
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    sparse_index = InMemorySparseIndex()
    embedder = InMemoryEmbedder()
    reranker = InMemoryReranker()

    c0 = create_sample_chunk("chunk-0", index=0, text="preceding context")
    c1 = create_sample_chunk("chunk-1", index=1, text="target context")
    c2 = create_sample_chunk("chunk-2", index=2, text="succeeding context")
    store.add_chunks([c0, c1, c2])

    vector_store.results_to_return = [RetrievedChunk(chunk=c1, score=0.95)]

    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
    )

    results = searcher.search(query="target", window_size=1)

    assert len(results) == 1
    adjacent = results[0].adjacent_chunks
    assert len(adjacent) == 3
    assert [c.id for c in adjacent] == ["chunk-0", "chunk-1", "chunk-2"]


def test_sliding_window_expansion_disabled() -> None:
    """When window_size=0, adjacent_chunks is empty."""
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    sparse_index = InMemorySparseIndex()
    embedder = InMemoryEmbedder()
    reranker = InMemoryReranker()

    c0 = create_sample_chunk("chunk-0", index=0)
    c1 = create_sample_chunk("chunk-1", index=1)
    store.add_chunks([c0, c1])

    vector_store.results_to_return = [RetrievedChunk(chunk=c1, score=0.95)]

    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
    )

    results = searcher.search(query="target", window_size=0)

    assert len(results) == 1
    assert results[0].adjacent_chunks == []


# ==============================================================================
# 4. Payload Filter & Tags Handling
# ==============================================================================


def test_payload_filter_and_tags_properly_forwarded() -> None:
    """Ensure payload_filter and tags are passed to vector store search."""
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    sparse_index = InMemorySparseIndex()
    embedder = InMemoryEmbedder()
    reranker = InMemoryReranker()

    chunk = create_sample_chunk("chunk-1")
    store.add_chunks([chunk])
    vector_store.results_to_return = [RetrievedChunk(chunk=chunk, score=0.9)]

    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
    )

    custom_filter = {"author": "Alessandro", "category": "notes"}
    tags = ["distributed-systems", "raft"]

    searcher.search(
        query="leader election",
        payload_filter=custom_filter,
        tags=tags,
    )

    assert len(vector_store.recorded_calls) == 1
    call = vector_store.recorded_calls[0]
    assert call["tags"] == tags
    assert call["payload_filter"] is not None
    assert call["payload_filter"]["author"] == "Alessandro"
    assert call["payload_filter"]["tags"] == tags


def test_payload_filter_with_mock_without_tags_argument() -> None:
    """Verify fallback when VectorStore.search does not accept 'tags' keyword argument."""
    store = InMemoryDocumentStore()
    mock_vs = MagicMock(spec=VectorStore)

    chunk = create_sample_chunk("c1")
    store.add_chunks([chunk])
    mock_vs.search.return_value = [RetrievedChunk(chunk=chunk, score=0.9)]

    sparse_index = InMemorySparseIndex()
    embedder = InMemoryEmbedder()
    reranker = InMemoryReranker()

    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=mock_vs,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
    )

    searcher.search("query", tags=["important"])

    assert mock_vs.search.called
    _, kwargs = mock_vs.search.call_args
    assert "payload_filter" in kwargs
    assert kwargs["payload_filter"] == {"tags": ["important"]}


# ==============================================================================
# 5. Validation Guardrails & Edge Cases
# ==============================================================================


def test_reranker_is_mandatory_parameter() -> None:
    """StrataHybridSearcher requires reranker as a mandatory argument."""
    with pytest.raises(TypeError, match=r"missing 1 required positional argument: 'reranker'"):
        StrataHybridSearcher(  # type: ignore[call-arg]
            sqlite_store=InMemoryDocumentStore(),
            vector_store=InMemoryVectorStore(),
            sparse_index=InMemorySparseIndex(),
            embedder=InMemoryEmbedder(),
        )


@pytest.mark.parametrize("empty_query", ["", "   ", "\t\n"])
def test_empty_query_returns_empty_list(empty_query: str) -> None:
    """Empty or whitespace-only queries return [] without running search legs."""
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    sparse_index = InMemorySparseIndex()
    embedder = InMemoryEmbedder()
    reranker = InMemoryReranker()

    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
    )

    results = searcher.search(query=empty_query)
    assert results == []
    assert len(vector_store.recorded_calls) == 0
    assert len(sparse_index.recorded_calls) == 0
    assert len(embedder.recorded_queries) == 0
    assert len(reranker.recorded_calls) == 0


@pytest.mark.parametrize("invalid_limit", [0, -1, -10])
def test_invalid_limit_raises_value_error(invalid_limit: int) -> None:
    """Non-positive search limit raises ValueError."""
    searcher = StrataHybridSearcher(
        sqlite_store=InMemoryDocumentStore(),
        vector_store=InMemoryVectorStore(),
        sparse_index=InMemorySparseIndex(),
        embedder=InMemoryEmbedder(),
        reranker=InMemoryReranker(),
    )

    with pytest.raises(ValueError, match=r"Search limit must be at least 1"):
        searcher.search(query="valid query", top_k=invalid_limit)

    with pytest.raises(ValueError, match=r"Search limit must be at least 1"):
        searcher.search(query="valid query", limit=invalid_limit)


def test_invalid_window_size_raises_value_error() -> None:
    """Negative window_size raises ValueError."""
    searcher = StrataHybridSearcher(
        sqlite_store=InMemoryDocumentStore(),
        vector_store=InMemoryVectorStore(),
        sparse_index=InMemorySparseIndex(),
        embedder=InMemoryEmbedder(),
        reranker=InMemoryReranker(),
    )

    with pytest.raises(ValueError, match=r"window_size must be non-negative"):
        searcher.search(query="valid", window_size=-1)


def test_invalid_weights_raises_value_error() -> None:
    """Negative weights raise ValueError."""
    searcher = StrataHybridSearcher(
        sqlite_store=InMemoryDocumentStore(),
        vector_store=InMemoryVectorStore(),
        sparse_index=InMemorySparseIndex(),
        embedder=InMemoryEmbedder(),
        reranker=InMemoryReranker(),
    )

    with pytest.raises(ValueError, match=r"Weights must be non-negative"):
        searcher.search(query="valid", dense_weight=-0.1)

    with pytest.raises(ValueError, match=r"Weights must be non-negative"):
        searcher.search(query="valid", sparse_weight=-0.5)


def test_data_desync_missing_chunk_skipped_safely() -> None:
    """When an indexed chunk ID is missing in SQLite, skip it safely without failing."""
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    sparse_index = InMemorySparseIndex()
    embedder = InMemoryEmbedder()
    reranker = InMemoryReranker()

    existing_chunk = create_sample_chunk("existing-chunk")
    store.add_chunks([existing_chunk])

    orphaned_chunk = create_sample_chunk("orphaned-chunk")
    vector_store.results_to_return = [
        RetrievedChunk(chunk=orphaned_chunk, score=0.99),
        RetrievedChunk(chunk=existing_chunk, score=0.80),
    ]

    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
    )

    results = searcher.search(query="test")
    assert len(results) == 1
    assert results[0].chunk is not None
    assert results[0].chunk.id == "existing-chunk"


def test_default_limit_resolves_from_settings() -> None:
    """When top_k/limit is None, default resolves from settings.final_top_k."""
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    sparse_index = InMemorySparseIndex()
    embedder = InMemoryEmbedder()
    reranker = InMemoryReranker()

    for i in range(15):
        chunk = create_sample_chunk(f"c-{i}", index=i)
        store.add_chunks([chunk])
        vector_store.results_to_return.append(RetrievedChunk(chunk=chunk, score=1.0 - (i * 0.05)))

    custom_settings = Settings(final_top_k=7, fused_top_k=15)
    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
        settings=custom_settings,
    )

    results = searcher.search(query="test query")
    assert len(results) == 7


def test_embedder_failure_raises_model_inference_error() -> None:
    """Embedder runtime errors are wrapped in ModelInferenceError."""
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    sparse_index = InMemorySparseIndex()
    reranker = InMemoryReranker()

    mock_embedder = MagicMock(spec=Embedder)
    mock_embedder.embed_query.side_effect = RuntimeError("Tokenizer crash")
    del mock_embedder.embed_text

    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=mock_embedder,
        reranker=reranker,
    )

    with pytest.raises(ModelInferenceError, match=r"Embedding query failed"):
        searcher.search(query="failure test")


def test_reranker_failure_propagates_model_inference_error() -> None:
    """Cross-encoder reranking runtime errors propagate as ModelInferenceError."""
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    sparse_index = InMemorySparseIndex()
    embedder = InMemoryEmbedder()

    chunk = create_sample_chunk("c1")
    store.add_chunks([chunk])
    vector_store.results_to_return = [RetrievedChunk(chunk=chunk, score=0.9)]

    mock_reranker = MagicMock(spec=Reranker)
    mock_reranker.rerank.side_effect = ModelInferenceError("ONNX session evaluation failure")

    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=mock_reranker,
    )

    with pytest.raises(ModelInferenceError, match=r"ONNX session evaluation failure"):
        searcher.search(query="test query")


def test_hybrid_searcher_protocol_conformance() -> None:
    """StrataHybridSearcher must satisfy runtime HybridSearcher protocol check."""
    searcher = StrataHybridSearcher(
        sqlite_store=InMemoryDocumentStore(),
        vector_store=InMemoryVectorStore(),
        sparse_index=InMemorySparseIndex(),
        embedder=InMemoryEmbedder(),
        reranker=InMemoryReranker(),
    )
    assert isinstance(searcher, HybridSearcher)


def test_search_result_chunk_and_adjacent_chunks_types() -> None:
    """Verify distinction and RetrievedChunk types for chunk and adjacent_chunks."""
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    sparse_index = InMemorySparseIndex()
    embedder = InMemoryEmbedder()
    reranker = InMemoryReranker()

    c0 = create_sample_chunk("c-0", index=0, text="preceding")
    c1 = create_sample_chunk("c-1", index=1, text="target")
    c2 = create_sample_chunk("c-2", index=2, text="succeeding")
    store.add_chunks([c0, c1, c2])

    vector_store.results_to_return = [RetrievedChunk(chunk=c1, score=0.9)]

    searcher = StrataHybridSearcher(
        sqlite_store=store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
    )

    results = searcher.search(query="target", window_size=1)
    assert len(results) == 1
    hit = results[0]

    # Verify target chunk type and identity
    assert hit.chunk is not None
    assert isinstance(hit.chunk, RetrievedChunk)
    assert hit.chunk.id == "c-1"
    assert hit.chunk.chunk == c1

    # Verify adjacent chunks type and distinction
    assert isinstance(hit.adjacent_chunks, list)
    assert len(hit.adjacent_chunks) == 3
    for adj in hit.adjacent_chunks:
        assert isinstance(adj, RetrievedChunk)
    assert [adj.id for adj in hit.adjacent_chunks] == ["c-0", "c-1", "c-2"]
    assert [adj.chunk for adj in hit.adjacent_chunks] == [c0, c1, c2]

    # Verify distinction: target chunk vs adjacent chunks
    assert hit.chunk.id != hit.adjacent_chunks[0].id
    assert hit.chunk.score == pytest.approx(1.0 / (60 + 1))
