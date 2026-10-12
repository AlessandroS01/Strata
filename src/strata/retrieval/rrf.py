"""Reciprocal Rank Fusion (RRF) algorithm for hybrid search retrieval.

Merges multiple ranked candidate lists (e.g., dense semantic ANN search and
sparse BM25 lexical search) by evaluating positional ranks rather than raw scores.
"""

import math
from collections import defaultdict

from strata.core.config import get_settings


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int | None = None,
    weights: list[float] | None = None,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        rankings: A list of ranked candidate lists, where each list contains
            tuples of (chunk_id, raw_score) sorted in descending order of relevance.
        k: Smoothing constant to adjust rank discounting. If None, defaults
            to `settings.rrf_k` from `strata.core.config`.
        weights: Optional list of multiplier weights corresponding to each ranking list.
            If provided, must match len(rankings). Defaults to 1.0 for each list.
        limit: Optional maximum number of fused candidates to return. If None,
            returns all unique candidates.

    Returns:
        A list of (chunk_id, rrf_score) tuples sorted in descending order of rrf_score.
        Ties are broken deterministically by chunk_id in ascending lexicographical order.

    Raises:
        ValueError: If k <= 0, or if len(weights) != len(rankings) when weights is provided,
            or if any weight is negative.
    """
    resolved_k = get_settings().rrf_k if k is None else k
    if resolved_k <= 0:
        raise ValueError(
            f"Smoothing constant k must be strictly positive (k > 0), got {resolved_k}"
        )

    if weights is not None:
        if len(weights) != len(rankings):
            raise ValueError(
                f"Length of weights ({len(weights)}) does not match length of rankings ({len(rankings)})"
            )
        if any(math.isnan(w) or w < 0.0 for w in weights):
            raise ValueError(f"Weights must be non-negative, got {weights}")

    if limit is not None and limit <= 0:
        return []

    if not rankings or all(len(ranking) == 0 for ranking in rankings):
        return []

    # Map chunk_id to accumulated RRF score using native 64-bit float
    fused_scores: dict[str, float] = defaultdict(float)

    for list_idx, ranking in enumerate(rankings):
        weight = float(weights[list_idx]) if weights is not None else 1.0

        seen_in_list: set[str] = set()
        rank = 1
        for chunk_id, _score in ranking:
            if chunk_id in seen_in_list:
                continue
            seen_in_list.add(chunk_id)
            fused_scores[chunk_id] += weight / (resolved_k + rank)
            rank += 1

    # Sort primary by rrf_score descending (-score), secondary by chunk_id ascending
    sorted_candidates = sorted(
        fused_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    if limit is not None:
        return sorted_candidates[:limit]

    return sorted_candidates
