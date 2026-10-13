"""Unit test suite for Strata hybrid retrieval coordinator (StrataHybridSearcher).

Tests verify:
- _extract_rankings normalization across diverse input representations
- Protocol structural subtyping conformance (HybridSearcher)
- Input validation and guardrails (empty query, non-positive limits, negative weights/windows)
- Embedder method detection (embed_text, embed_query, embed_texts) and failure handling
- Vector store payload filtering and tag forwarding (including TypeError fallback)
- Sparse BM25 lexical leg integration
- Reciprocal Rank Fusion (RRF) candidate ranking and weighting
- SQLite canonical hydration (chunk hydration, document hydration, desync tolerance)
- Parent document title hydration on RetrievedChunk
- Cross-encoder reranking score assignment and candidate reordering
- Small-to-Big context expansion (Section hydration with deduplication caching)
- Sliding window sibling chunk expansion
- Latency measurement and SearchResult packaging
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
from strata.retrieval.hybrid_search import StrataHybridSearcher, _extract_rankings

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
# Helper Factories
# ==============================================================================


def create_sample_document(
    doc_id: str = "doc-1",
    title: str = "Distributed Systems Handbook",
    file_path: str = "/data/notes/dist_sys.md",
) -> Document:
    """Helper factory for test Document instances."""
    return Document(
        id=doc_id,
        title=title,
        file_path=file_path,
        raw_content="Comprehensive overview of distributed systems consensus.",
        doc_hash="a" * 64,
    )


def create_sample_chunk(
    chunk_id: str,
    doc_id: str = "doc-1",
    index: int = 0,
    parent_section_id: str | None = None,
    text: str = "Sample chunk text",
    heading_path: str = "Section > Sub",
) -> Chunk:
    """Helper factory for test Chunk instances."""
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
    """Helper factory for test Section instances."""
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


def setup_test_searcher(
    store: InMemoryDocumentStore | None = None,
    vector_store: InMemoryVectorStore | None = None,
    sparse_index: InMemorySparseIndex | None = None,
    embedder: InMemoryEmbedder | None = None,
    reranker: InMemoryReranker | None = None,
    settings: Settings | None = None,
    default_doc: bool = True,
) -> tuple[
    StrataHybridSearcher,
    InMemoryDocumentStore,
    InMemoryVectorStore,
    InMemorySparseIndex,
    InMemoryEmbedder,
    InMemoryReranker,
]:
    """Convenience setup factory returning initialized searcher and its mock adapters."""
    doc_store = store if store is not None else InMemoryDocumentStore()
    if default_doc and hasattr(doc_store, "documents") and "doc-1" not in doc_store.documents:
        doc_store.add_document(create_sample_document(doc_id="doc-1"))

    v_store = vector_store if vector_store is not None else InMemoryVectorStore()
    s_index = sparse_index if sparse_index is not None else InMemorySparseIndex()
    emb = embedder if embedder is not None else InMemoryEmbedder()
    rr = reranker if reranker is not None else InMemoryReranker()

    searcher = StrataHybridSearcher(
        sqlite_store=doc_store,
        vector_store=v_store,
        sparse_index=s_index,
        embedder=emb,
        reranker=rr,
        settings=settings,
    )
    return searcher, doc_store, v_store, s_index, emb, rr


# ==============================================================================
# 1. _extract_rankings Normalization Tests
# ==============================================================================


def test_extract_rankings_from_tuples() -> None:
    """Verify normalization of (id, score) 2-tuples."""
    raw = [("chunk-1", 0.95), ("chunk-2", 0.42)]
    rankings = _extract_rankings(raw)
    assert rankings == [("chunk-1", 0.95), ("chunk-2", 0.42)]


def test_extract_rankings_from_retrieved_chunks() -> None:
    """Verify normalization of objects with .chunk and .score (RetrievedChunk)."""
    c = create_sample_chunk("chunk-abc")
    rc = RetrievedChunk(chunk=c, score=0.88)
    rankings = _extract_rankings([rc])
    assert rankings == [("chunk-abc", 0.88)]


def test_extract_rankings_from_chunk_without_id_attribute() -> None:
    """Verify fallback to str(chunk) when item.chunk lacks an .id attribute."""

    class SimpleItem:
        chunk = "raw-chunk-identifier"
        score = 0.75

    rankings = _extract_rankings([SimpleItem()])
    assert rankings == [("raw-chunk-identifier", 0.75)]


def test_extract_rankings_from_objects_with_id_and_score() -> None:
    """Verify normalization of arbitrary objects with .id and .score."""

    class CustomHit:
        id = "hit-99"
        score = 0.65

    rankings = _extract_rankings([CustomHit()])
    assert rankings == [("hit-99", 0.65)]


def test_extract_rankings_from_objects_with_id_only() -> None:
    """Verify objects with .id but no .score default score to 1.0."""

    class IdOnlyHit:
        id = "hit-only-id"

    rankings = _extract_rankings([IdOnlyHit()])
    assert rankings == [("hit-only-id", 1.0)]


def test_extract_rankings_from_raw_strings() -> None:
    """Verify raw string entries default score to 1.0."""
    rankings = _extract_rankings(["chunk-str-1", "chunk-str-2"])
    assert rankings == [("chunk-str-1", 1.0), ("chunk-str-2", 1.0)]


def test_extract_rankings_from_dictionaries() -> None:
    """Verify dictionary entries with various key layouts (chunk_id, id, nested chunk)."""
    raw: list[Any] = [
        {"chunk_id": "dict-1", "score": 0.9},
        {"id": "dict-2", "score": 0.8},
        {"chunk": {"id": "dict-3"}, "score": 0.7},
        {"id": "dict-4"},  # Defaults score to 1.0
    ]
    rankings = _extract_rankings(raw)
    assert rankings == [
        ("dict-1", 0.9),
        ("dict-2", 0.8),
        ("dict-3", 0.7),
        ("dict-4", 1.0),
    ]


def test_extract_rankings_ignores_unrecognized_items() -> None:
    """Verify unsupported types or dictionaries without identifier are skipped."""
    raw: list[Any] = [123, None, {"other": "value"}, ["list", "with", "three", "items"]]
    rankings = _extract_rankings(raw)
    assert rankings == []


# ==============================================================================
# 2. Protocol Conformance & Initialization
# ==============================================================================


def test_hybrid_searcher_protocol_conformance() -> None:
    """StrataHybridSearcher must satisfy runtime HybridSearcher protocol check."""
    searcher, _, _, _, _, _ = setup_test_searcher()
    assert isinstance(searcher, HybridSearcher)


def test_reranker_is_mandatory_parameter() -> None:
    """StrataHybridSearcher requires reranker as a mandatory argument."""
    with pytest.raises(TypeError, match=r"missing 1 required positional argument: 'reranker'"):
        StrataHybridSearcher(  # type: ignore[call-arg]
            sqlite_store=InMemoryDocumentStore(),
            vector_store=InMemoryVectorStore(),
            sparse_index=InMemorySparseIndex(),
            embedder=InMemoryEmbedder(),
        )


# ==============================================================================
# 3. Validation Guardrails & Parameter Handling
# ==============================================================================


@pytest.mark.parametrize("empty_query", ["", "   ", "\t\n"])
def test_empty_or_whitespace_query_returns_empty_list(empty_query: str) -> None:
    """Empty or whitespace-only queries return [] without running any search legs."""
    searcher, _, v_store, s_index, emb, rr = setup_test_searcher()

    results = searcher.search(query=empty_query)
    assert results == []
    assert len(v_store.recorded_calls) == 0
    assert len(s_index.recorded_calls) == 0
    assert len(emb.recorded_queries) == 0
    assert len(rr.recorded_calls) == 0


@pytest.mark.parametrize("invalid_limit", [0, -1, -10])
def test_invalid_limit_raises_value_error(invalid_limit: int) -> None:
    """Non-positive search limit raises ValueError."""
    searcher, _, _, _, _, _ = setup_test_searcher()

    with pytest.raises(ValueError, match=r"Search limit must be at least 1"):
        searcher.search(query="valid query", top_k=invalid_limit)

    with pytest.raises(ValueError, match=r"Search limit must be at least 1"):
        searcher.search(query="valid query", limit=invalid_limit)


def test_negative_window_size_raises_value_error() -> None:
    """Negative window_size raises ValueError."""
    searcher, _, _, _, _, _ = setup_test_searcher()

    with pytest.raises(ValueError, match=r"window_size must be non-negative"):
        searcher.search(query="valid query", window_size=-1)


def test_negative_weights_raise_value_error() -> None:
    """Negative dense or sparse weights raise ValueError."""
    searcher, _, _, _, _, _ = setup_test_searcher()

    with pytest.raises(ValueError, match=r"Weights must be non-negative"):
        searcher.search(query="valid query", dense_weight=-0.1)

    with pytest.raises(ValueError, match=r"Weights must be non-negative"):
        searcher.search(query="valid query", sparse_weight=-0.5)


def test_default_limit_resolves_from_settings() -> None:
    """When top_k/limit is None, default resolves from settings.final_top_k."""
    searcher, doc_store, v_store, _, _, _ = setup_test_searcher(
        settings=Settings(final_top_k=4, fused_top_k=10)
    )

    for i in range(10):
        chunk = create_sample_chunk(f"c-{i}", index=i)
        doc_store.add_chunks([chunk])
        v_store.results_to_return.append(RetrievedChunk(chunk=chunk, score=1.0 - (i * 0.05)))

    results = searcher.search(query="test query")
    assert len(results) == 4


def test_limit_alias_takes_precedence_over_top_k() -> None:
    """When both limit and top_k are specified, limit parameter takes precedence."""
    searcher, doc_store, v_store, _, _, _ = setup_test_searcher()

    for i in range(5):
        chunk = create_sample_chunk(f"c-{i}", index=i)
        doc_store.add_chunks([chunk])
        v_store.results_to_return.append(RetrievedChunk(chunk=chunk, score=1.0 - (i * 0.1)))

    results = searcher.search(query="test", limit=2, top_k=5)
    assert len(results) == 2


# ==============================================================================
# 4. Dense Leg & Embedder Variations
# ==============================================================================


def test_embedder_embed_text_called_when_present() -> None:
    """When embedder has embed_text, it is called preferentially."""
    mock_emb = MagicMock()
    mock_emb.embed_text.return_value = [0.1, 0.2, 0.3, 0.4]

    searcher, doc_store, v_store, _, _, _ = setup_test_searcher(embedder=mock_emb)
    chunk = create_sample_chunk("c1")
    doc_store.add_chunks([chunk])
    v_store.results_to_return = [RetrievedChunk(chunk=chunk, score=0.9)]

    results = searcher.search(query="my query")
    assert len(results) == 1
    mock_emb.embed_text.assert_called_once_with("my query")


def test_embedder_embed_query_called_when_embed_text_absent() -> None:
    """When embedder lacks embed_text but has embed_query, embed_query is called."""
    mock_emb = MagicMock(spec=["embed_query", "embed_texts", "dimension"])
    mock_emb.embed_query.return_value = [0.2, 0.4, 0.6, 0.8]

    searcher, doc_store, v_store, _, _, _ = setup_test_searcher(embedder=mock_emb)
    chunk = create_sample_chunk("c1")
    doc_store.add_chunks([chunk])
    v_store.results_to_return = [RetrievedChunk(chunk=chunk, score=0.9)]

    results = searcher.search(query="my query")
    assert len(results) == 1
    mock_emb.embed_query.assert_called_once_with("my query")


def test_embedder_embed_texts_fallback_when_other_methods_absent() -> None:
    """When embedder only exposes batch embed_texts, it is called with single query list."""
    mock_emb = MagicMock(spec=["embed_texts", "dimension"])
    mock_emb.embed_texts.return_value = [[0.5, 0.5, 0.5, 0.5]]

    searcher, doc_store, v_store, _, _, _ = setup_test_searcher(embedder=mock_emb)
    chunk = create_sample_chunk("c1")
    doc_store.add_chunks([chunk])
    v_store.results_to_return = [RetrievedChunk(chunk=chunk, score=0.9)]

    results = searcher.search(query="batch query")
    assert len(results) == 1
    mock_emb.embed_texts.assert_called_once_with(["batch query"])


def test_embedder_without_methods_raises_model_inference_error() -> None:
    """Embedder without any query embedding method raises ModelInferenceError."""
    mock_emb = MagicMock(spec=["dimension"])

    searcher, _, _, _, _, _ = setup_test_searcher(embedder=mock_emb)

    with pytest.raises(
        ModelInferenceError, match=r"Embedder does not expose a query embedding method"
    ):
        searcher.search(query="query text")


def test_embedder_runtime_error_wrapped_in_model_inference_error() -> None:
    """Runtime exceptions raised by embedder are wrapped in ModelInferenceError."""
    mock_emb = MagicMock()
    mock_emb.embed_text.side_effect = RuntimeError("Tokenizer crash")

    searcher, _, _, _, _, _ = setup_test_searcher(embedder=mock_emb)

    with pytest.raises(ModelInferenceError, match=r"Embedding query failed: Tokenizer crash"):
        searcher.search(query="crash query")


def test_embedder_model_inference_error_propagated_without_wrapping() -> None:
    """ModelInferenceError raised by embedder is re-raised directly without double-wrapping."""
    mock_emb = MagicMock()
    mock_emb.embed_text.side_effect = ModelInferenceError("Direct embedding failure")

    searcher, _, _, _, _, _ = setup_test_searcher(embedder=mock_emb)

    with pytest.raises(ModelInferenceError, match=r"^Direct embedding failure$"):
        searcher.search(query="direct failure")


# ==============================================================================
# 5. Payload Filtering & Tags Handling
# ==============================================================================


def test_payload_filter_and_tags_forwarded() -> None:
    """Ensure payload_filter and tags are passed to vector store search."""
    searcher, doc_store, v_store, _, _, _ = setup_test_searcher()

    chunk = create_sample_chunk("chunk-1")
    doc_store.add_chunks([chunk])
    v_store.results_to_return = [RetrievedChunk(chunk=chunk, score=0.9)]

    custom_filter = {"author": "Alessandro", "category": "notes"}
    tags = ["distributed-systems", "raft"]

    searcher.search(
        query="leader election",
        payload_filter=custom_filter,
        tags=tags,
    )

    assert len(v_store.recorded_calls) == 1
    call = v_store.recorded_calls[0]
    assert call["tags"] == tags
    assert call["payload_filter"] is not None
    assert call["payload_filter"]["author"] == "Alessandro"
    assert call["payload_filter"]["tags"] == tags


def test_payload_filter_preserves_existing_tags_key() -> None:
    """If payload_filter already contains 'tags', it is preserved and not overwritten."""
    searcher, doc_store, v_store, _, _, _ = setup_test_searcher()

    chunk = create_sample_chunk("chunk-1")
    doc_store.add_chunks([chunk])
    v_store.results_to_return = [RetrievedChunk(chunk=chunk, score=0.9)]

    custom_filter = {"tags": ["original-tag"]}
    searcher.search(
        query="leader election",
        payload_filter=custom_filter,
        tags=["new-tag"],
    )

    assert len(v_store.recorded_calls) == 1
    call = v_store.recorded_calls[0]
    assert call["payload_filter"]["tags"] == ["original-tag"]


def test_vector_store_type_error_fallback_without_tags_argument() -> None:
    """Verify fallback when VectorStore.search raises TypeError when passed 'tags'."""
    doc_store = InMemoryDocumentStore()
    doc_store.add_document(create_sample_document(doc_id="doc-1"))
    chunk = create_sample_chunk("c1")
    doc_store.add_chunks([chunk])

    mock_vs = MagicMock()
    # First call with tags raises TypeError, second call without tags succeeds
    mock_vs.search.side_effect = [
        TypeError("search() got an unexpected keyword argument 'tags'"),
        [RetrievedChunk(chunk=chunk, score=0.9)],
    ]

    searcher, _, _, _, _, _ = setup_test_searcher(store=doc_store, vector_store=mock_vs)

    results = searcher.search("query", tags=["important"])
    assert len(results) == 1
    assert mock_vs.search.call_count == 2
    # Verify second call had no tags kwarg
    second_call_kwargs = mock_vs.search.call_args_list[1][1]
    assert "tags" not in second_call_kwargs


# ==============================================================================
# 6. Candidate Hydration & SQLite Desync Tolerance
# ==============================================================================


def test_missing_chunk_in_sqlite_skipped_safely() -> None:
    """When an indexed chunk ID is missing in SQLite, skip it safely without failing."""
    searcher, doc_store, v_store, _, _, _ = setup_test_searcher()

    existing_chunk = create_sample_chunk("existing-chunk")
    doc_store.add_chunks([existing_chunk])

    orphaned_chunk = create_sample_chunk("orphaned-chunk")
    v_store.results_to_return = [
        RetrievedChunk(chunk=orphaned_chunk, score=0.99),
        RetrievedChunk(chunk=existing_chunk, score=0.80),
    ]

    results = searcher.search(query="test")
    assert len(results) == 1
    assert results[0].chunk is not None
    assert results[0].chunk.id == "existing-chunk"


def test_missing_parent_document_in_sqlite_skipped_safely() -> None:
    """When chunk exists but its parent document is missing in SQLite, skip it safely."""
    # Initialize searcher without default document
    doc_store = InMemoryDocumentStore()
    searcher, _, v_store, _, _, _ = setup_test_searcher(store=doc_store, default_doc=False)

    # doc-valid exists, doc-ghost does not exist
    doc_store.add_document(create_sample_document(doc_id="doc-valid", title="Valid Document"))

    valid_chunk = create_sample_chunk("valid-chunk", doc_id="doc-valid")
    ghost_chunk = create_sample_chunk("ghost-chunk", doc_id="doc-ghost")
    doc_store.add_chunks([valid_chunk, ghost_chunk])

    v_store.results_to_return = [
        RetrievedChunk(chunk=ghost_chunk, score=0.99),
        RetrievedChunk(chunk=valid_chunk, score=0.85),
    ]

    results = searcher.search(query="test")
    assert len(results) == 1
    assert results[0].chunk is not None
    assert results[0].chunk.id == "valid-chunk"
    assert results[0].chunk.document_title == "Valid Document"


def test_document_title_hydrated_from_parent_document() -> None:
    """RetrievedChunk.document_title is hydrated from the parent Document."""
    doc_store = InMemoryDocumentStore()
    doc_store.add_document(
        create_sample_document(doc_id="doc-1", title="Database Internals (Vol 1)")
    )
    chunk = create_sample_chunk("c1", doc_id="doc-1")
    doc_store.add_chunks([chunk])

    searcher, _, v_store, _, _, _ = setup_test_searcher(store=doc_store, default_doc=False)
    v_store.results_to_return = [RetrievedChunk(chunk=chunk, score=0.9)]

    results = searcher.search(query="storage engine")
    assert len(results) == 1
    assert results[0].chunk is not None
    assert results[0].chunk.document_title == "Database Internals (Vol 1)"


def test_all_candidates_missing_returns_empty_list() -> None:
    """If all candidate chunks or parent documents are missing from SQLite, return []."""
    searcher, _, v_store, _, _, rr = setup_test_searcher(default_doc=False)

    # Return chunks that don't exist in doc_store
    missing_chunk = create_sample_chunk("ghost-chunk")
    v_store.results_to_return = [RetrievedChunk(chunk=missing_chunk, score=0.99)]

    results = searcher.search(query="missing test")
    assert results == []
    # Cross-encoder reranker must not be called when candidate list is empty
    assert len(rr.recorded_calls) == 0


def test_chunk_hydration_falls_back_to_get_chunk() -> None:
    """When DocumentStore lacks get_chunk_by_id, it falls back to get_chunk."""
    mock_store = MagicMock(spec=["get_chunk", "get_document", "get_adjacent_chunks"])
    chunk = create_sample_chunk("c1", doc_id="doc-1")
    doc = create_sample_document(doc_id="doc-1")
    mock_store.get_chunk.return_value = chunk
    mock_store.get_document.return_value = doc

    searcher, _, v_store, _, _, _ = setup_test_searcher(store=mock_store)
    v_store.results_to_return = [RetrievedChunk(chunk=chunk, score=0.9)]

    results = searcher.search("query")
    assert len(results) == 1
    mock_store.get_chunk.assert_called_once_with("c1")


# ==============================================================================
# 7. Cross-Encoder Re-Ranking & RRF Fusion
# ==============================================================================


def test_end_to_end_hybrid_search() -> None:
    """Verify query flows through dense ANN, sparse BM25, RRF, and mandatory reranker."""
    searcher, doc_store, v_store, s_index, emb, rr = setup_test_searcher(
        settings=Settings(fused_top_k=5, rrf_k=60)
    )

    chunk_a = create_sample_chunk("chunk-a", index=0, text="Raft leader election protocol")
    chunk_b = create_sample_chunk("chunk-b", index=1, text="Paxos distributed consensus")
    doc_store.add_chunks([chunk_a, chunk_b])

    # Dense leg: chunk-a (0.95), chunk-b (0.80)
    v_store.results_to_return = [
        RetrievedChunk(chunk=chunk_a, score=0.95),
        RetrievedChunk(chunk=chunk_b, score=0.80),
    ]

    # Sparse leg: chunk-b (12.0), chunk-a (8.0)
    s_index.results_to_return = [
        RetrievedChunk(chunk=chunk_b, score=12.0),
        RetrievedChunk(chunk=chunk_a, score=8.0),
    ]

    results = searcher.search(query="consensus algorithms", top_k=2)

    # 1. Verify query passed to embedder
    assert emb.recorded_queries == ["consensus algorithms"]

    # 2. Verify query passed to vector store with candidate depth
    assert len(v_store.recorded_calls) == 1
    assert v_store.recorded_calls[0]["query_vector"] == [0.5, 0.5, 0.5, 0.5]
    assert v_store.recorded_calls[0]["limit"] == 20

    # 3. Verify query passed to sparse index with candidate depth
    assert len(s_index.recorded_calls) == 1
    assert s_index.recorded_calls[0]["query"] == "consensus algorithms"
    assert s_index.recorded_calls[0]["limit"] == 20

    # 4. Verify candidates were passed into reranker
    assert len(rr.recorded_calls) == 1
    assert rr.recorded_calls[0]["query"] == "consensus algorithms"
    assert rr.recorded_calls[0]["top_k"] == 2

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
    searcher, doc_store, v_store, _, _, rr = setup_test_searcher()

    c_leader = create_sample_chunk("c-leader", text="Leader election heartbeat")
    c_log = create_sample_chunk("c-log", text="Log compaction snapshot")
    doc_store.add_chunks([c_leader, c_log])

    # Initial dense retrieval favors c_leader over c_log
    v_store.results_to_return = [
        RetrievedChunk(chunk=c_leader, score=0.99),
        RetrievedChunk(chunk=c_log, score=0.50),
    ]

    # Reranker assigns high score to c_log and low score to c_leader
    rr.custom_scores = {
        "c-leader": 0.15,
        "c-log": 0.94,
    }

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


def test_reranker_failure_propagates_model_inference_error() -> None:
    """Cross-encoder reranking runtime errors propagate as ModelInferenceError."""
    mock_rr = MagicMock(spec=Reranker)
    mock_rr.rerank.side_effect = ModelInferenceError("ONNX session evaluation failure")

    searcher, doc_store, v_store, _, _, _ = setup_test_searcher(reranker=mock_rr)

    chunk = create_sample_chunk("c1")
    doc_store.add_chunks([chunk])
    v_store.results_to_return = [RetrievedChunk(chunk=chunk, score=0.9)]

    with pytest.raises(ModelInferenceError, match=r"ONNX session evaluation failure"):
        searcher.search(query="test query")


# ==============================================================================
# 8. Small-to-Big Hydration & Parent Section Deduplication
# ==============================================================================


def test_small_to_big_hydration_enabled() -> None:
    """When expand_section=True, SearchResult.parent_section is populated with matching Section."""
    searcher, doc_store, v_store, _, _, _ = setup_test_searcher()

    section = create_sample_section("sec-42", title="Raft Overview")
    chunk = create_sample_chunk("chunk-1", parent_section_id="sec-42")

    doc_store.add_sections([section])
    doc_store.add_chunks([chunk])

    v_store.results_to_return = [RetrievedChunk(chunk=chunk, score=0.9)]

    results = searcher.search(query="raft", expand_section=True)

    assert len(results) == 1
    assert results[0].parent_section is not None
    assert results[0].parent_section.id == "sec-42"
    assert results[0].parent_section.title == "Raft Overview"
    assert results[0].chunk is not None
    assert results[0].chunk.parent_section == results[0].parent_section
    assert results[0].chunk.parent_context == section.content


def test_small_to_big_hydration_disabled() -> None:
    """When expand_section=False, parent_section is None even if chunk has parent_section_id."""
    searcher, doc_store, v_store, _, _, _ = setup_test_searcher()

    section = create_sample_section("sec-42", title="Raft Overview")
    chunk = create_sample_chunk("chunk-1", parent_section_id="sec-42")

    doc_store.add_sections([section])
    doc_store.add_chunks([chunk])

    v_store.results_to_return = [RetrievedChunk(chunk=chunk, score=0.9)]

    results = searcher.search(query="raft", expand_section=False)

    assert len(results) == 1
    assert results[0].parent_section is None


def test_section_caching_for_shared_parent_section() -> None:
    """When multiple chunks share the same parent section, fetch from SQLite only once."""
    searcher, doc_store, v_store, _, _, _ = setup_test_searcher()

    shared_section = create_sample_section("sec-shared", title="Consensus Module")
    c1 = create_sample_chunk("c1", parent_section_id="sec-shared")
    c2 = create_sample_chunk("c2", parent_section_id="sec-shared")
    c3 = create_sample_chunk("c3", parent_section_id="sec-shared")

    doc_store.add_sections([shared_section])
    doc_store.add_chunks([c1, c2, c3])

    v_store.results_to_return = [
        RetrievedChunk(chunk=c1, score=0.9),
        RetrievedChunk(chunk=c2, score=0.8),
        RetrievedChunk(chunk=c3, score=0.7),
    ]

    results = searcher.search(query="consensus", expand_section=True, top_k=3)

    assert len(results) == 3
    for r in results:
        assert r.parent_section is not None
        assert r.parent_section.id == "sec-shared"

    # SQLite store must have been queried exactly once for sec-shared due to section_cache
    assert doc_store.recorded_section_fetches == ["sec-shared"]


def test_small_to_big_missing_section_or_none_parent_handled_safely() -> None:
    """When chunk has no parent_section_id or section is missing, parent_section is None."""
    searcher, doc_store, v_store, _, _, _ = setup_test_searcher()

    chunk_without_parent = create_sample_chunk("c-no-parent", parent_section_id=None)
    chunk_with_missing_parent = create_sample_chunk("c-missing-parent", parent_section_id="ghost")

    doc_store.add_chunks([chunk_without_parent, chunk_with_missing_parent])

    v_store.results_to_return = [
        RetrievedChunk(chunk=chunk_without_parent, score=0.9),
        RetrievedChunk(chunk=chunk_with_missing_parent, score=0.8),
    ]

    results = searcher.search(query="test", expand_section=True)
    assert len(results) == 2
    assert results[0].parent_section is None
    assert results[1].parent_section is None


def test_section_hydration_falls_back_to_get_section() -> None:
    """When DocumentStore lacks get_section_by_id, it falls back to get_section."""
    mock_store = MagicMock(spec=["get_chunk", "get_document", "get_section"])
    chunk = create_sample_chunk("c1", doc_id="doc-1", parent_section_id="sec-1")
    doc = create_sample_document(doc_id="doc-1")
    sec = create_sample_section("sec-1", doc_id="doc-1")
    mock_store.get_chunk.return_value = chunk
    mock_store.get_document.return_value = doc
    mock_store.get_section.return_value = sec

    searcher, _, v_store, _, _, _ = setup_test_searcher(store=mock_store)
    v_store.results_to_return = [RetrievedChunk(chunk=chunk, score=0.9)]

    results = searcher.search("query", expand_section=True)
    assert len(results) == 1
    mock_store.get_section.assert_called_once_with("sec-1")
    assert results[0].parent_section == sec


# ==============================================================================
# 9. Sliding Window Expansion
# ==============================================================================


def test_sliding_window_expansion_enabled() -> None:
    """When window_size=1, SearchResult.adjacent_chunks contains sibling chunks."""
    searcher, doc_store, v_store, _, _, _ = setup_test_searcher()

    c0 = create_sample_chunk("chunk-0", index=0, text="preceding context")
    c1 = create_sample_chunk("chunk-1", index=1, text="target context")
    c2 = create_sample_chunk("chunk-2", index=2, text="succeeding context")
    doc_store.add_chunks([c0, c1, c2])

    v_store.results_to_return = [RetrievedChunk(chunk=c1, score=0.95)]

    results = searcher.search(query="target", window_size=1)

    assert len(results) == 1
    adjacent = results[0].adjacent_chunks
    assert len(adjacent) == 3
    assert [c.id for c in adjacent] == ["chunk-0", "chunk-1", "chunk-2"]
    for adj in adjacent:
        assert isinstance(adj, RetrievedChunk)
        assert adj.score == 0.0


def test_sliding_window_expansion_disabled() -> None:
    """When window_size=0, adjacent_chunks is empty."""
    searcher, doc_store, v_store, _, _, _ = setup_test_searcher()

    c0 = create_sample_chunk("chunk-0", index=0)
    c1 = create_sample_chunk("chunk-1", index=1)
    doc_store.add_chunks([c0, c1])

    v_store.results_to_return = [RetrievedChunk(chunk=c1, score=0.95)]

    results = searcher.search(query="target", window_size=0)

    assert len(results) == 1
    assert results[0].adjacent_chunks == []


# ==============================================================================
# 10. SearchResult Output & Latency Measurement
# ==============================================================================


def test_search_result_chunk_and_adjacent_chunks_types() -> None:
    """Verify distinction and RetrievedChunk types for chunk and adjacent_chunks."""
    searcher, doc_store, v_store, _, _, _ = setup_test_searcher()

    c0 = create_sample_chunk("c-0", index=0, text="preceding")
    c1 = create_sample_chunk("c-1", index=1, text="target")
    c2 = create_sample_chunk("c-2", index=2, text="succeeding")
    doc_store.add_chunks([c0, c1, c2])

    v_store.results_to_return = [RetrievedChunk(chunk=c1, score=0.9)]

    results = searcher.search(query="target", window_size=1)
    assert len(results) == 1
    hit = results[0]

    # Verify target chunk type and identity
    assert hit.chunk is not None
    assert isinstance(hit.chunk, RetrievedChunk)
    assert hit.chunk.id == "c-1"
    assert hit.chunk.chunk == c1
    assert hit.chunk.document_title == "Distributed Systems Handbook"

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
    assert hit.total_latency_ms >= 0.0
