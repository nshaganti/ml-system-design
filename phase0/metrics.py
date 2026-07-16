"""
Phase 0 -- Ranking Metrics
============================
Pure, dependency-free ranking metrics over a ranked list of recommended item
IDs and a set of relevant (e.g. purchased) item IDs.

Kept separate from evaluate.py so they are trivially unit-testable and reusable
by every phase (Phase 2's ranker cares about NDCG/MAP, not just recall).

All functions assume BINARY relevance: an item is either relevant or it isn't.
"""

from __future__ import annotations

import math
from typing import Iterable


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k recommendations that are relevant."""
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items that appear in the top-k recommendations."""
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def dcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Discounted cumulative gain with binary gains."""
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            # rank i is 0-based -> discount by log2(i + 2)
            dcg += 1.0 / math.log2(i + 2)
    return dcg


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """
    Normalized DCG: DCG divided by the ideal DCG (all relevant items ranked
    first). Ranges 0..1; rewards putting relevant items nearer the top.
    """
    if not relevant:
        return 0.0
    dcg = dcg_at_k(recommended, relevant, k)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """
    Average precision @ k: mean of precision values taken at each rank where a
    relevant item is found. The building block of MAP (mean average precision).
    """
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)   # precision at this rank
    denom = min(len(relevant), k)
    return score / denom if denom > 0 else 0.0


def mean(values: Iterable[float]) -> float:
    """Small helper: mean of an iterable, 0.0 when empty."""
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
