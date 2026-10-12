"""Unit test suite for FlashRankReranker cross-encoder adapter.

Tests verify:
- Conformance to the Reranker runtime protocol
- Lazy-loading of model to ensure near-zero startup time
- Immediate empty list returns on blank query or empty candidates
- Input validation (positive top_k)
- Score mapping, descending sort, and deterministic tie-breaking
- Slicing to top_k limit with fallback to Settings.final_top_k
- Error wrapping of initialization and inference exceptions into ModelInferenceError
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strata.core.config import Settings
from strata.core.exceptions import ModelInferenceError
from strata.core.models import Chunk, RetrievedChunk
from strata.core.protocols import Reranker
from strata.retrieval.reranker import FlashRankReranker


def create_test_chunk(chunk_id: str, text: str = "Sample content") -> RetrievedChunk:
    """Helper factory for test RetrievedChunk instances."""
    return RetrievedChunk(
        chunk=Chunk(
            id=chunk_id,
            document_id="doc-1",
            chunk_hash=f"hash-{chunk_id}",
            text=text,
            heading_path="Root > Heading",
            index=0,
            token_count=10,
        ),
        score=0.1,
    )


class DummyFlashRankModel:
    """Mock FlashRank model returning predefined scores."""

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores: dict[str, float] = scores or {}
        self.recorded_requests: list[Any] = []

    def rerank(self, request: Any) -> list[dict[str, Any]]:
        self.recorded_requests.append(request)
        results: list[dict[str, Any]] = []
        for passage in request.passages:
            pid = passage["id"]
            score = self.scores.get(pid, 0.5)
            results.append({"id": pid, "text": passage["text"], "score": score})
        return results


# ==============================================================================
# 1. Protocol Conformance & Initialization
# ==============================================================================


def test_flashrank_reranker_protocol_conformance() -> None:
    """Verify FlashRankReranker satisfies the Reranker runtime protocol."""
    reranker = FlashRankReranker()
    assert isinstance(reranker, Reranker)
    assert Reranker in FlashRankReranker.__mro__


def test_lazy_loading_model_not_initialized_on_instantiation() -> None:
    """Underlying model instance must not be loaded during constructor execution."""
    reranker = FlashRankReranker()
    assert reranker._model is None


def test_lazy_loading_model_initialized_on_first_access() -> None:
    """Model is initialized on first invocation of _get_model and cached thereafter."""
    custom_settings = Settings(reranker_model="ms-marco-TinyBERT-L-2-v2")
    reranker = FlashRankReranker(settings=custom_settings)

    mock_ranker_cls = MagicMock()
    mock_instance = MagicMock()
    mock_ranker_cls.return_value = mock_instance

    with patch.dict("sys.modules", {"flashrank": MagicMock(Ranker=mock_ranker_cls)}):
        model1 = reranker._get_model()
        assert model1 is mock_instance
        mock_ranker_cls.assert_called_once_with(
            model_name="ms-marco-TinyBERT-L-2-v2",
            cache_dir=str(custom_settings.data_dir / "models"),
        )

        # Subsequent call returns cached instance without re-instantiation
        model2 = reranker._get_model()
        assert model2 is mock_instance
        assert mock_ranker_cls.call_count == 1


# ==============================================================================
# 2. Guardrails & Early Exits
# ==============================================================================


@pytest.mark.parametrize("empty_query", ["", "   ", "\t\n"])
def test_empty_query_returns_empty_list_without_loading_model(empty_query: str) -> None:
    """Blank or whitespace-only query returns [] immediately without initializing model."""
    reranker = FlashRankReranker()
    candidates = [create_test_chunk("c1")]

    results = reranker.rerank(query=empty_query, candidates=candidates)
    assert results == []
    assert reranker._model is None


def test_empty_candidates_returns_empty_list_without_loading_model() -> None:
    """Empty candidate list returns [] immediately without initializing model."""
    reranker = FlashRankReranker()

    results = reranker.rerank(query="search term", candidates=[])
    assert results == []
    assert reranker._model is None


@pytest.mark.parametrize("invalid_top_k", [0, -1, -10])
def test_invalid_top_k_raises_value_error(invalid_top_k: int) -> None:
    """Non-positive top_k raises ValueError."""
    dummy_model = DummyFlashRankModel()
    reranker = FlashRankReranker(model=dummy_model)
    candidates = [create_test_chunk("c1")]

    with pytest.raises(ValueError, match=r"top_k must be strictly positive"):
        reranker.rerank(query="valid query", candidates=candidates, top_k=invalid_top_k)


# ==============================================================================
# 3. Scoring, Ordering & Tie-Breaking
# ==============================================================================


def test_reranking_scores_and_descending_order() -> None:
    """Candidates are scored, updated with rerank_score, and ordered descending."""
    dummy_model = DummyFlashRankModel(
        scores={
            "c1": 0.25,
            "c2": 0.95,
            "c3": 0.60,
        }
    )
    reranker = FlashRankReranker(model=dummy_model)

    candidates = [
        create_test_chunk("c1", "Low relevance text"),
        create_test_chunk("c2", "High relevance text"),
        create_test_chunk("c3", "Medium relevance text"),
    ]

    results = reranker.rerank(query="machine learning", candidates=candidates, top_k=3)

    assert len(results) == 3
    assert [r.chunk.id for r in results] == ["c2", "c3", "c1"]
    assert results[0].score == 0.95
    assert results[0].rerank_score == 0.95
    assert results[1].score == 0.60
    assert results[1].rerank_score == 0.60
    assert results[2].score == 0.25
    assert results[2].rerank_score == 0.25


def test_deterministic_tie_breaking_by_chunk_id() -> None:
    """Candidates with identical scores are ordered deterministically by chunk ID ascending."""
    dummy_model = DummyFlashRankModel(
        scores={
            "chunk-z": 0.80,
            "chunk-a": 0.80,
            "chunk-m": 0.80,
        }
    )
    reranker = FlashRankReranker(model=dummy_model)

    candidates = [
        create_test_chunk("chunk-z"),
        create_test_chunk("chunk-a"),
        create_test_chunk("chunk-m"),
    ]

    results = reranker.rerank(query="test tie-break", candidates=candidates, top_k=3)

    assert [r.chunk.id for r in results] == ["chunk-a", "chunk-m", "chunk-z"]
    assert all(r.score == 0.80 for r in results)


def test_top_k_slicing_and_default_from_settings() -> None:
    """Results are sliced to explicit top_k, or fallback to settings.final_top_k."""
    dummy_model = DummyFlashRankModel(scores={f"c{i}": float(i) / 10.0 for i in range(10)})
    settings = Settings(final_top_k=3)
    reranker = FlashRankReranker(settings=settings, model=dummy_model)

    candidates = [create_test_chunk(f"c{i}") for i in range(10)]

    # 1. Fallback to settings.final_top_k (3)
    default_results = reranker.rerank(query="query", candidates=candidates)
    assert len(default_results) == 3
    assert [r.chunk.id for r in default_results] == ["c9", "c8", "c7"]

    # 2. Explicit top_k limit (2)
    custom_results = reranker.rerank(query="query", candidates=candidates, top_k=2)
    assert len(custom_results) == 2
    assert [r.chunk.id for r in custom_results] == ["c9", "c8"]


# ==============================================================================
# 4. Error Handling & Exception Wrapping
# ==============================================================================


def test_model_initialization_failure_raises_model_inference_error() -> None:
    """Failure during model lazy-loading is wrapped in ModelInferenceError."""
    reranker = FlashRankReranker()

    mock_ranker_cls = MagicMock(side_effect=RuntimeError("ONNX weights missing"))
    with patch.dict("sys.modules", {"flashrank": MagicMock(Ranker=mock_ranker_cls)}):
        candidates = [create_test_chunk("c1")]
        with pytest.raises(ModelInferenceError, match=r"Failed to initialize FlashRank reranker"):
            reranker.rerank(query="valid query", candidates=candidates)


def test_inference_scoring_failure_raises_model_inference_error() -> None:
    """Failure during scoring execution is wrapped in ModelInferenceError."""
    mock_model = MagicMock()
    mock_model.rerank.side_effect = RuntimeError("Inference execution crash")
    reranker = FlashRankReranker(model=mock_model)

    candidates = [create_test_chunk("c1")]
    with pytest.raises(ModelInferenceError, match=r"FlashRank reranking inference failed"):
        reranker.rerank(query="valid query", candidates=candidates)
