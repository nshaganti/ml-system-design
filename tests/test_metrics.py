"""Tests for the pure ranking-metric functions in phase0/metrics.py."""

from __future__ import annotations

import math

from metrics import (
    precision_at_k,
    recall_at_k,
    dcg_at_k,
    ndcg_at_k,
    average_precision_at_k,
    mean,
)


REC = ["a", "b", "c", "d"]


def test_precision_at_k():
    # 2 of top-4 relevant -> 0.5
    assert precision_at_k(REC, {"a", "c"}, 4) == 0.5
    # only top-2 considered: "a" relevant, "b" not -> 0.5
    assert precision_at_k(REC, {"a", "c"}, 2) == 0.5
    assert precision_at_k(REC, set(), 4) == 0.0
    assert precision_at_k(REC, {"a"}, 0) == 0.0


def test_recall_at_k():
    # relevant {a, z}; only "a" retrievable -> 1/2
    assert recall_at_k(REC, {"a", "z"}, 4) == 0.5
    assert recall_at_k(REC, {"a", "b"}, 4) == 1.0
    assert recall_at_k(REC, set(), 4) == 0.0


def test_dcg_rewards_earlier_hits():
    early = dcg_at_k(["a", "x", "y"], {"a"}, 3)   # hit at rank 0
    late = dcg_at_k(["x", "y", "a"], {"a"}, 3)    # hit at rank 2
    assert early > late
    # rank 0 gain = 1/log2(2) = 1.0
    assert math.isclose(early, 1.0)


def test_ndcg_perfect_is_one():
    # All relevant items already at the top -> NDCG = 1.0
    assert math.isclose(ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3), 1.0)
    assert ndcg_at_k(REC, set(), 4) == 0.0


def test_ndcg_between_zero_and_one():
    val = ndcg_at_k(["x", "a", "y"], {"a"}, 3)
    assert 0.0 < val < 1.0


def test_average_precision():
    # relevant {a, c}: hit at rank1 (p=1/1) and rank3 (p=2/3); AP = (1 + 0.667)/2
    ap = average_precision_at_k(REC, {"a", "c"}, 4)
    assert math.isclose(ap, (1.0 + 2.0 / 3.0) / 2.0)
    assert average_precision_at_k(REC, set(), 4) == 0.0


def test_average_precision_perfect():
    assert math.isclose(average_precision_at_k(["a", "b", "z"], {"a", "b"}, 3), 1.0)


def test_mean_helper():
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([]) == 0.0
