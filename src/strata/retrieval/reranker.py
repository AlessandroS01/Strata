"""Cross-encoder reranking adapter utilizing FlashRank ONNX runtime.

Provides lightweight, CPU-friendly cross-attention scoring over candidate
passages with zero background daemons and zero external network calls.
Fulfills the Reranker protocol defined in strata.core.protocols.
"""

import logging
from typing import Any

from strata.core.config import Settings, get_settings
from strata.core.exceptions import ModelInferenceError
from strata.core.models import RetrievedChunk
from strata.core.protocols import Reranker

logger = logging.getLogger(__name__)

__all__ = ["FlashRankReranker"]


class FlashRankReranker(Reranker):
    """Local CPU-friendly cross-encoder reranker implementing the Reranker protocol.

    Utilizes FlashRank ONNX models to compute cross-attention relevance scores
    for candidate chunks. Models are lazy-loaded on the first invocation to ensure
    sub-millisecond application startup.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        model: Any = None,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize FlashRank reranker with optional settings and model override.

        Args:
            settings: Configuration settings containing reranker_model and final_top_k.
            model: Optional pre-initialized ranker instance (useful for unit testing).
            cache_dir: Optional filesystem cache directory for model storage.
        """
        self.settings: Settings = settings if settings is not None else get_settings()
        self._model: Any = model
        self._cache_dir: str = cache_dir or str(self.settings.data_dir / "models")

    def _get_model(self) -> Any:
        """Lazy-load the underlying FlashRank model on first inference invocation."""
        if self._model is None:
            try:
                from flashrank import Ranker

                logger.debug(
                    "Lazy-loading FlashRank reranker model: %s (cache: %s)",
                    self.settings.reranker_model,
                    self._cache_dir,
                )
                self._model = Ranker(
                    model_name=self.settings.reranker_model,
                    cache_dir=self._cache_dir,
                )
            except Exception as exc:
                raise ModelInferenceError(
                    f"Failed to initialize FlashRank reranker with model "
                    f"'{self.settings.reranker_model}': {exc}"
                ) from exc
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Re-rank candidate chunks using full cross-attention scoring.

        Args:
            query: User search query string.
            candidates: List of canonical RetrievedChunk instances from RRF fusion.
            top_k: Maximum number of reranked candidates to return. Defaults to
                final_top_k in Settings.

        Returns:
            Ranked list of RetrievedChunk models sorted by rerank score descending,
            with ties broken deterministically by chunk ID ascending.

        Raises:
            ValueError: If top_k is non-positive.
            ModelInferenceError: If model initialization or scoring fails.
        """
        # 1. Immediate guardrail checks: return empty on empty candidates or blank query
        if not candidates or not query or not query.strip():
            return []

        resolved_top_k = top_k if top_k is not None else self.settings.final_top_k
        if resolved_top_k <= 0:
            raise ValueError(f"top_k must be strictly positive (>= 1), got {resolved_top_k}")

        # 2. Execute FlashRank cross-attention scoring
        try:
            model = self._get_model()
            from flashrank import RerankRequest

            passages = [{"id": c.chunk.id, "text": c.chunk.text} for c in candidates]
            request = RerankRequest(query=query, passages=passages)
            raw_results = model.rerank(request)
        except Exception as exc:
            if isinstance(exc, ModelInferenceError):
                raise
            raise ModelInferenceError(f"FlashRank reranking inference failed: {exc}") from exc

        # 3. Map assigned cross-attention scores back to candidate IDs
        score_map: dict[str, float] = {}
        for item in raw_results:
            if isinstance(item, dict) and "id" in item and "score" in item:
                score_map[str(item["id"])] = float(item["score"])
            elif hasattr(item, "id") and hasattr(item, "score"):
                score_map[str(item.id)] = float(item.score)
            elif isinstance(item, tuple) and len(item) == 2:
                score_map[str(item[0])] = float(item[1])

        # 4. Construct updated RetrievedChunk domain objects with rerank score provenance
        reranked_chunks: list[RetrievedChunk] = []
        for candidate in candidates:
            new_score = score_map.get(candidate.chunk.id, candidate.score)
            reranked_chunks.append(
                candidate.model_copy(
                    update={
                        "score": new_score,
                        "rerank_score": new_score,
                    }
                )
            )

        # 5. Sort descending by relevance score, tie-break by chunk ID ascending
        sorted_chunks = sorted(
            reranked_chunks,
            key=lambda c: (-c.score, c.chunk.id),
        )

        return sorted_chunks[:resolved_top_k]
