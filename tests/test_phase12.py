"""
Tests for Phase 12 -- TwoStageRanker composition (pipeline.py).

We use tiny fakes for the three collaborators so the test proves the WIRING --
stage 1 retrieves, features are fetched, stage 2 reranks, the interface holds --
without training a model or touching the dataset. That's the unit under test:
the composition, not the two-tower or the LR (each already has its own tests).
"""

from __future__ import annotations

import polars as pl

from pipeline import TwoStageRanker


class FakeGenerator:
    """Stage-1 stand-in: returns a fixed candidate list, records the n it was asked."""

    def __init__(self, candidates):
        self._candidates = candidates
        self.catalog_size = 999
        self.asked_n = None

    def recommend(self, user_id, user_events, n=20):
        self.asked_n = n
        return list(self._candidates[:n])


class ReverseReranker:
    """Stage-2 stand-in: ranks candidates in REVERSE of stage-1 order."""

    def rank(self, features_df, item_col="item_id", n=20):
        return features_df[item_col].to_list()[::-1][:n]


class PassThroughStore:
    """Feature-store stand-in: returns the (user, item) rows unchanged."""

    def get_online_features_batch(self, entity_df):
        return entity_df


def _events():
    return pl.DataFrame(
        {"user_id": ["u1"], "item_id": ["x"], "timestamp_ms": [1]}
    ).with_columns(pl.col("timestamp_ms").cast(pl.Int64))


def test_two_stage_reranks_stage1_candidates():
    gen = FakeGenerator(["a", "b", "c", "d"])
    ranker = TwoStageRanker(gen, ReverseReranker(), PassThroughStore(), candidate_pool=4)
    out = ranker.recommend("u1", _events(), n=3)
    # ReverseReranker flips [a,b,c,d] -> [d,c,b,a]; top-3 = d,c,b
    assert out == ["d", "c", "b"]


def test_stage1_is_asked_for_the_candidate_pool_not_n():
    gen = FakeGenerator(["a", "b", "c", "d", "e"])
    ranker = TwoStageRanker(gen, ReverseReranker(), PassThroughStore(), candidate_pool=5)
    ranker.recommend("u1", _events(), n=2)
    assert gen.asked_n == 5          # retrieval widens; ranking narrows


def test_catalog_size_delegates_to_stage1():
    gen = FakeGenerator(["a"])
    ranker = TwoStageRanker(gen, ReverseReranker(), PassThroughStore())
    assert ranker.catalog_size == 999


def test_empty_stage1_yields_empty_output():
    gen = FakeGenerator([])
    ranker = TwoStageRanker(gen, ReverseReranker(), PassThroughStore())
    assert ranker.recommend("u1", _events(), n=5) == []


def test_backfill_when_reranker_returns_too_few():
    class DropReranker:
        def rank(self, features_df, item_col="item_id", n=20):
            return ["a"]                      # returns only one, regardless of n

    gen = FakeGenerator(["a", "b", "c", "d"])
    ranker = TwoStageRanker(gen, DropReranker(), PassThroughStore(), candidate_pool=4)
    out = ranker.recommend("u1", _events(), n=3)
    assert out[0] == "a"                      # reranker's stays first
    assert len(out) == 3                      # backfilled from stage-1 order
    assert out == ["a", "b", "c"]
