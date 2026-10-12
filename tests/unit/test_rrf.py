"""Unit tests for Reciprocal Rank Fusion (RRF) retrieval algorithm.

Tests verify mathematical correctness, rank-based deduplication, deterministic
tie-breaking, edge-case bounds, weight multipliers, and configuration defaults.
"""

import math
from unittest.mock import patch

import pytest

from strata.core.config import Settings
from strata.retrieval.rrf import reciprocal_rank_fusion

# -----------------------------------------------------------------------------
# 1. Edge Cases & Boundary Handling
# -----------------------------------------------------------------------------


def test_empty_rankings_returns_empty_list() -> None:
    """Empty list of rankings returns an empty candidate list."""
    assert reciprocal_rank_fusion([]) == []


def test_all_inner_lists_empty_returns_empty_list() -> None:
    """When all input ranking lists are empty, return empty list."""
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([[], [], []], weights=[1.0, 1.0, 1.0]) == []


def test_mixed_empty_and_populated_rankings() -> None:
    """Populated rankings merge gracefully when paired with empty candidate lists."""
    list1: list[tuple[str, float]] = [("chunk_a", 0.9), ("chunk_b", 0.7)]
    list2: list[tuple[str, float]] = []

    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert len(fused) == 2
    assert fused[0] == ("chunk_a", pytest.approx(1.0 / 61))
    assert fused[1] == ("chunk_b", pytest.approx(1.0 / 62))


@pytest.mark.parametrize("limit", [0, -1, -100])
def test_limit_zero_or_negative_returns_empty_list(limit: int) -> None:
    """A limit of zero or negative returns an empty list without error."""
    ranking: list[tuple[str, float]] = [("chunk_a", 1.0), ("chunk_b", 0.5)]
    assert reciprocal_rank_fusion([ranking], k=60, limit=limit) == []


def test_limit_none_returns_all_unique_candidates() -> None:
    """When limit is None, all unique candidates are returned."""
    list1: list[tuple[str, float]] = [("a", 1.0), ("b", 0.8)]
    list2: list[tuple[str, float]] = [("c", 1.0), ("d", 0.8)]

    fused = reciprocal_rank_fusion([list1, list2], k=60, limit=None)
    assert len(fused) == 4


def test_limit_larger_than_candidate_pool() -> None:
    """When limit exceeds candidate count, return all candidates without padding."""
    ranking: list[tuple[str, float]] = [("chunk_a", 1.0), ("chunk_b", 0.5)]
    fused = reciprocal_rank_fusion([ranking], k=60, limit=50)
    assert len(fused) == 2


def test_limit_truncates_to_top_k() -> None:
    """Limit truncates fused results to top candidates in score order."""
    ranking: list[tuple[str, float]] = [
        ("chunk_a", 10.0),
        ("chunk_b", 8.0),
        ("chunk_c", 6.0),
        ("chunk_d", 4.0),
    ]
    fused = reciprocal_rank_fusion([ranking], k=60, limit=2)
    assert len(fused) == 2
    assert fused[0][0] == "chunk_a"
    assert fused[1][0] == "chunk_b"


# -----------------------------------------------------------------------------
# 2. Mathematical Correctness & Single / Multi-List Fusion
# -----------------------------------------------------------------------------


def test_single_ranking_preserves_relative_order() -> None:
    """Single ranking list preserves its order while transforming scores to reciprocal ranks."""
    ranking: list[tuple[str, float]] = [
        ("c_first", 100.5),
        ("c_second", 50.2),
        ("c_third", 10.1),
    ]
    fused = reciprocal_rank_fusion([ranking], k=60)

    assert len(fused) == 3
    assert [cid for cid, _ in fused] == ["c_first", "c_second", "c_third"]

    expected_scores = [1.0 / (60 + 1), 1.0 / (60 + 2), 1.0 / (60 + 3)]
    for (_, actual_score), expected_score in zip(fused, expected_scores, strict=True):
        assert actual_score == pytest.approx(expected_score)


def test_multiple_rankings_reciprocal_sum() -> None:
    """Multi-list fusion sums reciprocal rank contributions across lists."""
    dense: list[tuple[str, float]] = [
        ("doc_1", 0.95),  # rank 1
        ("doc_2", 0.85),  # rank 2
        ("doc_3", 0.75),  # rank 3
    ]
    sparse: list[tuple[str, float]] = [
        ("doc_2", 15.0),  # rank 1
        ("doc_4", 12.0),  # rank 2
        ("doc_1", 10.0),  # rank 3
    ]

    # k = 60
    # doc_1: 1/61 + 1/63 = 0.01639344 + 0.01587302 = 0.03226646
    # doc_2: 1/62 + 1/61 = 0.01612903 + 0.01639344 = 0.03252247  -> doc_2 > doc_1
    # doc_3: 1/63 = 0.01587302
    # doc_4: 1/62 = 0.01612903                                   -> doc_4 > doc_3
    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    assert len(fused) == 4
    expected_order = ["doc_2", "doc_1", "doc_4", "doc_3"]
    assert [cid for cid, _ in fused] == expected_order

    scores_dict = dict(fused)
    assert scores_dict["doc_1"] == pytest.approx(1.0 / 61 + 1.0 / 63)
    assert scores_dict["doc_2"] == pytest.approx(1.0 / 62 + 1.0 / 61)
    assert scores_dict["doc_3"] == pytest.approx(1.0 / 63)
    assert scores_dict["doc_4"] == pytest.approx(1.0 / 62)


def test_disjoint_lists_fusion() -> None:
    """Candidates with zero overlap between lists fuse correctly with tie-breaking."""
    list1: list[tuple[str, float]] = [("apple", 0.9), ("banana", 0.8)]
    list2: list[tuple[str, float]] = [("cherry", 10.0), ("date", 8.0)]

    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert len(fused) == 4

    # Rank 1 in list1: apple (1/61), Rank 1 in list2: cherry (1/61) -> tie broken by chunk_id
    # Rank 2 in list1: banana (1/62), Rank 2 in list2: date (1/62) -> tie broken by chunk_id
    expected_ids = ["apple", "cherry", "banana", "date"]
    assert [cid for cid, _ in fused] == expected_ids


def test_floating_point_precision_is_float64() -> None:
    """Verify returned scores are native 64-bit Python floats."""
    ranking: list[tuple[str, float]] = [("doc_x", 0.5)]
    fused = reciprocal_rank_fusion([ranking], k=60)
    score = fused[0][1]
    assert isinstance(score, float)
    assert not math.isnan(score)
    assert not math.isinf(score)


# -----------------------------------------------------------------------------
# 3. Default Configuration & Parameter Overrides
# -----------------------------------------------------------------------------


def test_default_k_from_settings() -> None:
    """When k is None, the smoothing constant resolves from core Settings.rrf_k."""
    ranking: list[tuple[str, float]] = [("chunk_a", 1.0)]

    # Explicit None should resolve to Settings.rrf_k (default 60)
    fused_none = reciprocal_rank_fusion([ranking], k=None)
    # Default argument should also resolve to Settings.rrf_k
    fused_default = reciprocal_rank_fusion([ranking])

    assert fused_none[0][1] == pytest.approx(1.0 / (60 + 1))
    assert fused_default[0][1] == pytest.approx(1.0 / (60 + 1))


def test_custom_settings_k_resolution() -> None:
    """Smoothing constant dynamically respects patched settings.rrf_k."""
    custom_settings = Settings(rrf_k=40)
    ranking: list[tuple[str, float]] = [("chunk_a", 1.0)]

    with patch("strata.retrieval.rrf.get_settings", return_value=custom_settings):
        fused = reciprocal_rank_fusion([ranking])
        assert fused[0][1] == pytest.approx(1.0 / (40 + 1))


def test_explicit_k_overrides_settings() -> None:
    """Explicitly provided k takes precedence over settings."""
    ranking: list[tuple[str, float]] = [("chunk_a", 1.0)]
    fused_k10 = reciprocal_rank_fusion([ranking], k=10)
    assert fused_k10[0][1] == pytest.approx(1.0 / (10 + 1))

    fused_k100 = reciprocal_rank_fusion([ranking], k=100)
    assert fused_k100[0][1] == pytest.approx(1.0 / (100 + 1))


# -----------------------------------------------------------------------------
# 4. Weight Multipliers
# -----------------------------------------------------------------------------


def test_weighted_fusion_asymmetric() -> None:
    """Weights scale reciprocal rank contributions proportionally."""
    list1: list[tuple[str, float]] = [("c1", 1.0), ("c2", 0.5)]
    list2: list[tuple[str, float]] = [("c2", 10.0), ("c3", 5.0)]

    weights = [0.8, 0.2]
    # k = 60
    # c1: 0.8 / 61
    # c2: 0.8 / 62 + 0.2 / 61
    # c3: 0.2 / 62
    fused = reciprocal_rank_fusion([list1, list2], k=60, weights=weights)

    scores = dict(fused)
    assert scores["c1"] == pytest.approx(0.8 / 61)
    assert scores["c2"] == pytest.approx(0.8 / 62 + 0.2 / 61)
    assert scores["c3"] == pytest.approx(0.2 / 62)


def test_zero_weight_for_one_ranking() -> None:
    """Ranking with weight 0.0 contributes 0.0 to candidate scores."""
    list1: list[tuple[str, float]] = [("c1", 1.0)]
    list2: list[tuple[str, float]] = [("c2", 1.0)]

    fused = reciprocal_rank_fusion([list1, list2], k=60, weights=[1.0, 0.0])
    scores = dict(fused)
    assert scores["c1"] == pytest.approx(1.0 / 61)
    assert scores["c2"] == 0.0


# -----------------------------------------------------------------------------
# 5. Deduplication within a Single Ranking List
# -----------------------------------------------------------------------------


def test_deduplication_uses_highest_rank_and_ignores_subsequent() -> None:
    """Duplicate occurrences in a single list use highest rank (lowest ordinal r)."""
    # doc_a occurs at rank 1 and duplicate at position 3
    ranking: list[tuple[str, float]] = [
        ("doc_a", 0.95),  # 1st occurrence -> rank 1
        ("doc_b", 0.80),  # rank 2
        ("doc_a", 0.65),  # duplicate -> ignored
    ]
    fused = reciprocal_rank_fusion([ranking], k=60)

    assert len(fused) == 2
    assert fused[0] == ("doc_a", pytest.approx(1.0 / 61))
    assert fused[1] == ("doc_b", pytest.approx(1.0 / 62))


def test_consecutive_duplicates_at_head() -> None:
    """Consecutive duplicates at head of list do not displace subsequent item ranks."""
    ranking: list[tuple[str, float]] = [
        ("doc_a", 0.99),  # rank 1
        ("doc_a", 0.95),  # duplicate -> ignored
        ("doc_b", 0.50),  # rank 2
    ]
    fused = reciprocal_rank_fusion([ranking], k=60)

    assert len(fused) == 2
    assert fused[0] == ("doc_a", pytest.approx(1.0 / 61))
    assert fused[1] == ("doc_b", pytest.approx(1.0 / 62))


def test_multiple_duplicates_across_multiple_lists() -> None:
    """Duplicates within each list are pruned independently without affecting other lists."""
    list1: list[tuple[str, float]] = [("x", 1.0), ("x", 0.9), ("y", 0.8)]
    list2: list[tuple[str, float]] = [("y", 10.0), ("x", 8.0), ("x", 6.0)]

    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert len(fused) == 2

    # In list1: x has rank 1, y has rank 2
    # In list2: y has rank 1, x has rank 2
    # Total x: 1/61 + 1/62
    # Total y: 1/62 + 1/61 (tie: x vs y -> x < y)
    assert fused[0] == ("x", pytest.approx(1.0 / 61 + 1.0 / 62))
    assert fused[1] == ("y", pytest.approx(1.0 / 61 + 1.0 / 62))


# -----------------------------------------------------------------------------
# 6. Deterministic Tie-Breaking
# -----------------------------------------------------------------------------


def test_deterministic_tie_breaking_lexicographical() -> None:
    """Identical RRF scores are broken deterministically by chunk_id ascending."""
    # List of 4 items with identical score
    ranking: list[tuple[str, float]] = [("gamma", 1.0)]
    ranking2: list[tuple[str, float]] = [("alpha", 1.0)]
    ranking3: list[tuple[str, float]] = [("delta", 1.0)]
    ranking4: list[tuple[str, float]] = [("beta", 1.0)]

    # Each is rank 1 in its own list -> equal RRF score 1/61
    fused = reciprocal_rank_fusion([ranking, ranking2, ranking3, ranking4], k=60)

    assert [cid for cid, _ in fused] == ["alpha", "beta", "delta", "gamma"]


def test_tie_breaking_with_zero_scores() -> None:
    """Candidates with 0.0 scores break ties lexicographically."""
    list1: list[tuple[str, float]] = [("z_chunk", 1.0), ("a_chunk", 1.0)]
    fused = reciprocal_rank_fusion([list1], k=60, weights=[0.0])

    assert [cid for cid, _ in fused] == ["a_chunk", "z_chunk"]


# -----------------------------------------------------------------------------
# 7. Validation & Exception Handling
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_k", [0, -1, -60])
def test_invalid_k_raises_value_error(invalid_k: int) -> None:
    """Non-positive k raises ValueError."""
    ranking: list[tuple[str, float]] = [("chunk_1", 1.0)]
    with pytest.raises(ValueError, match="k must be strictly positive"):
        reciprocal_rank_fusion([ranking], k=invalid_k)


def test_weights_length_mismatch_raises_value_error() -> None:
    """When weights length does not match rankings length, raise ValueError."""
    rankings: list[list[tuple[str, float]]] = [[("c1", 1.0)], [("c2", 1.0)]]

    # Too many weights
    with pytest.raises(ValueError, match=r"Length of weights .* does not match"):
        reciprocal_rank_fusion(rankings, weights=[1.0, 1.0, 1.0])

    # Too few weights
    with pytest.raises(ValueError, match=r"Length of weights .* does not match"):
        reciprocal_rank_fusion(rankings, weights=[1.0])


@pytest.mark.parametrize("invalid_weights", [[-0.1, 1.0], [1.0, -1.0], [float("nan"), 1.0]])
def test_negative_or_nan_weights_raises_value_error(invalid_weights: list[float]) -> None:
    """Negative or NaN weights raise ValueError."""
    rankings: list[list[tuple[str, float]]] = [[("c1", 1.0)], [("c2", 1.0)]]
    with pytest.raises(ValueError, match="Weights must be non-negative"):
        reciprocal_rank_fusion(rankings, weights=invalid_weights)
