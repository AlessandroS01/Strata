"""Hybrid retrieval coordinator for Strata air-gapped knowledge engine.

Orchestrates dense ANN search (Qdrant), sparse lexical search (BM25),
Reciprocal Rank Fusion (RRF), and two-stage SQLite hydration (Small-to-Big
and sliding window expansion) with sub-200ms latency. Zero LLM dependencies.
"""

import inspect
import logging
from collections.abc import Mapping
from typing import Any, cast

from strata.core.config import Settings, get_settings
from strata.core.exceptions import ModelInferenceError
from strata.core.models import Chunk, RetrievedChunk, SearchResult, Section
from strata.core.protocols import (
    DocumentStore,
    Embedder,
    HybridSearcher,
    SparseIndex,
    VectorStore,
)
from strata.retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

__all__ = ["SearchResult", "StrataHybridSearcher"]


def _extract_rankings(raw_results: list[Any]) -> list[tuple[str, float]]:
    """Normalize diverse search result representations into a list of (chunk_id, score) tuples.

    Handles RetrievedChunk/RetrievedContext models, 2-tuples, objects with id/score,
    and dictionary formats.
    """
    rankings: list[tuple[str, float]] = []
    for item in raw_results:
        if isinstance(item, tuple) and len(item) == 2:
            rankings.append((str(item[0]), float(item[1])))
        elif hasattr(item, "chunk") and hasattr(item, "score"):
            chunk_obj = item.chunk
            chunk_id = getattr(chunk_obj, "id", str(chunk_obj))
            score = float(item.score)
            rankings.append((str(chunk_id), score))
        elif hasattr(item, "id") and hasattr(item, "score"):
            rankings.append((str(item.id), float(item.score)))
        elif hasattr(item, "id"):
            rankings.append((str(item.id), 1.0))
        elif isinstance(item, str):
            rankings.append((item, 1.0))
        elif isinstance(item, dict):
            cid = item.get("chunk_id") or item.get("id")
            if cid is None and isinstance(item.get("chunk"), dict):
                cid = item["chunk"].get("id")
            score = float(item.get("score", 1.0))
            if cid is not None:
                rankings.append((str(cid), score))
    return rankings


class StrataHybridSearcher(HybridSearcher):
    """Decoupled hybrid retrieval coordinator fulfilling the HybridSearcher protocol.

    Executes parallel candidate retrieval over dense vector index and sparse BM25 index,
    merges rankings using Reciprocal Rank Fusion (RRF), and hydrates rich context from
    canonical SQLite storage.
    """

    def __init__(
        self,
        sqlite_store: DocumentStore,
        vector_store: VectorStore,
        sparse_index: SparseIndex,
        embedder: Embedder,
        settings: Settings | None = None,
    ) -> None:
        """Initialize hybrid searcher with injected storage and indexing adapters.

        Args:
            sqlite_store: Canonical SQLite document and chunk storage source of truth.
            vector_store: Embedded vector store adapter (Qdrant).
            sparse_index: BM25 lexical inverted index adapter.
            embedder: Local dense embedding model runtime.
            settings: Optional configuration settings. Defaults to core Settings singleton.
        """
        self.sqlite_store: DocumentStore = sqlite_store
        self.vector_store: VectorStore = vector_store
        self.sparse_index: SparseIndex = sparse_index
        self.embedder: Embedder = embedder
        self.settings: Settings = settings if settings is not None else get_settings()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        payload_filter: Mapping[str, object] | None = None,
        limit: int | None = None,
        tags: list[str] | None = None,
        expand_section: bool = False,
        window_size: int = 0,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[SearchResult]:
        """Execute decoupled hybrid search pipeline in sub-200ms without LLM.

        Args:
            query: User search query text.
            top_k: Optional maximum number of fused candidates to return. Defaults to
                configured fused_top_k in Settings.
            payload_filter: Optional arbitrary key-value metadata filter for vector search.
            limit: Alias for top_k to support generic limit parameterization.
            tags: Optional tags list to filter candidates during retrieval.
            expand_section: If True, hydrates parent Section from SQLite for Small-to-Big
                context expansion.
            window_size: Number of adjacent sibling chunks to fetch on each side (>= 0).
            dense_weight: Reciprocal rank fusion multiplier for dense vector results.
            sparse_weight: Reciprocal rank fusion multiplier for sparse lexical results.

        Returns:
            Ranked list of hydrated SearchResult objects. Returns empty list if query is empty.

        Raises:
            ValueError: If limit < 1, window_size < 0, or weights < 0.
            ModelInferenceError: If dense embedding generation fails.
        """
        # 1. Validation & Guardrails
        if not query or not query.strip():
            logger.debug("Empty query provided to hybrid search, returning empty list")
            return []

        resolved_limit = limit if limit is not None else top_k
        if resolved_limit is None:
            resolved_limit = self.settings.fused_top_k

        if resolved_limit < 1:
            raise ValueError(f"Search limit must be at least 1, got {resolved_limit}")

        if window_size < 0:
            raise ValueError(f"window_size must be non-negative, got {window_size}")

        if dense_weight < 0.0 or sparse_weight < 0.0:
            raise ValueError("Weights must be non-negative")

        candidate_depth = max(resolved_limit * 2, 20)

        # 2. Parallel Candidate Retrieval
        # 2a. Dense Leg: compute query embedding and search vector store
        try:
            if hasattr(self.embedder, "embed_text"):
                query_vector: list[float] = cast(Any, self.embedder).embed_text(query)
            elif hasattr(self.embedder, "embed_query"):
                query_vector = self.embedder.embed_query(query)
            elif hasattr(self.embedder, "embed_texts"):
                batch_vectors = self.embedder.embed_texts([query])
                query_vector = batch_vectors[0] if batch_vectors else []
            else:
                raise ModelInferenceError("Embedder does not expose a query embedding method")
        except Exception as exc:
            if isinstance(exc, ModelInferenceError):
                raise
            raise ModelInferenceError(f"Embedding query failed: {exc}") from exc

        # Construct and forward payload filter including tags
        effective_filter: dict[str, object] = (
            dict(payload_filter) if payload_filter is not None else {}
        )
        if tags is not None and "tags" not in effective_filter:
            effective_filter["tags"] = tags
        filter_arg: Mapping[str, object] | None = effective_filter if effective_filter else None

        sig = inspect.signature(self.vector_store.search)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

        search_kwargs: dict[str, Any] = {"limit": candidate_depth}
        if "payload_filter" in sig.parameters or has_var_kw:
            search_kwargs["payload_filter"] = filter_arg
        if tags is not None and ("tags" in sig.parameters or has_var_kw):
            search_kwargs["tags"] = tags

        try:
            dense_results_raw = self.vector_store.search(query_vector, **search_kwargs)
        except TypeError:
            search_kwargs.pop("tags", None)
            dense_results_raw = self.vector_store.search(query_vector, **search_kwargs)

        # 2b. Sparse Leg: query BM25 index
        sparse_results_raw = self.sparse_index.search(query, limit=candidate_depth)

        # 3. Reciprocal Rank Fusion
        dense_rankings = _extract_rankings(dense_results_raw)
        sparse_rankings = _extract_rankings(sparse_results_raw)

        fused_candidates = reciprocal_rank_fusion(
            rankings=[dense_rankings, sparse_rankings],
            k=self.settings.rrf_k,
            weights=[dense_weight, sparse_weight],
            limit=resolved_limit,
        )

        top_candidates = fused_candidates[:resolved_limit]

        # 4. Two-Stage Hydration from SQLite
        results: list[SearchResult] = []
        for chunk_id, fused_score in top_candidates:
            # Hydrate canonical chunk
            if hasattr(self.sqlite_store, "get_chunk_by_id"):
                chunk: Chunk | None = cast(Any, self.sqlite_store).get_chunk_by_id(chunk_id)
            else:
                chunk = self.sqlite_store.get_chunk(chunk_id)

            if chunk is None:
                # Data desync or orphaned chunk: skip safely
                logger.warning("Chunk id=%s not found in SQLite store; skipping", chunk_id)
                continue

            # Small-to-Big Expansion (expand_section=True)
            parent_section: Section | None = None
            if expand_section and chunk.parent_section_id:
                if hasattr(self.sqlite_store, "get_section_by_id"):
                    parent_section = cast(Any, self.sqlite_store).get_section_by_id(
                        chunk.parent_section_id
                    )
                else:
                    parent_section = self.sqlite_store.get_section(chunk.parent_section_id)

            # Hydrate central target chunk as RetrievedChunk
            target_retrieved = RetrievedChunk(
                chunk=chunk,
                score=fused_score,
                rrf_score=fused_score,
                parent_context=parent_section.content if parent_section else None,
                parent_section=parent_section,
            )

            # Window Expansion (window_size > 0)
            adjacent_raw_chunks: list[Chunk] = []
            if window_size > 0:
                adjacent_raw_chunks = self.sqlite_store.get_adjacent_chunks(
                    chunk_id, window=window_size
                )

            adjacent_retrieved: list[RetrievedChunk] = [
                RetrievedChunk(chunk=adj_c, score=0.0) for adj_c in adjacent_raw_chunks
            ]

            results.append(
                SearchResult(
                    chunk=target_retrieved,
                    score=fused_score,
                    parent_section=parent_section,
                    adjacent_chunks=adjacent_retrieved,
                )
            )

        return results
