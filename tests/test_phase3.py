"""
Phase 3 tests: candidate generation + the recommendation service.

Focus areas: the two-stage flow wires together, business rules actually filter,
the service degrades gracefully instead of crashing, and Rule 29 feature logging
captures the served items.
"""

from __future__ import annotations

import polars as pl
import pytest

from feature_store import PointInTimeFeatureStore
from lr_ranker import LRRanker
from candidate_generator import PopularityCandidateGenerator
from service import RecommendationService, RecommendationRequest


def _events():
    rows = [
        ("u1", "strong",    "a", 10),
        ("u2", "medium", "b", 11),
        ("u1", "strong",    "c", 12),
        ("u3", "strong",    "a", 13),
        ("u2", "strong",    "a", 14),
        ("u3", "medium", "b", 15),
        ("u1", "strong",    "b", 16),
        ("u2", "strong",    "a", 17),  # gives 'a' a clear popularity lead
    ]
    return pl.DataFrame(
        {
            "user_id":      [r[0] for r in rows],
            "event_type":   [r[1] for r in rows],
            "item_id":      [r[2] for r in rows],
            "timestamp_ms": [r[3] for r in rows],
        }
    ).with_columns(pl.col("timestamp_ms").cast(pl.Int64))


def _store():
    return PointInTimeFeatureStore().fit(_events(), cutoff_timestamp_ms=1000)


# ------------------------------------------------------- candidate generator


def test_popularity_generator_returns_most_popular_first():
    store = _store()
    cg = PopularityCandidateGenerator(store._item_totals, pool_size=10)
    cands = cg.generate("anyuser", n=3)
    # 'a' has the most strong events -> first.
    assert cands[0] == "a"
    assert len(cands) == 3


def test_popularity_generator_is_user_agnostic():
    store = _store()
    cg = PopularityCandidateGenerator(store._item_totals)
    assert cg.generate("u1", n=5) == cg.generate("u2", n=5)


# ------------------------------------------------------------------ service


def _service(ranker=None, oos=None):
    store = _store()
    cg = PopularityCandidateGenerator(store._item_totals, pool_size=10)
    return RecommendationService(cg, store, ranker=ranker, ineligible_items=oos or set())


def test_service_returns_requested_count():
    svc = _service()
    resp = svc.recommend(RecommendationRequest(user_id="u1", n=2))
    assert len(resp.items) == 2
    assert resp.request_id  # a uuid was assigned


def test_service_records_all_stage_latencies():
    svc = _service()
    resp = svc.recommend(RecommendationRequest(user_id="u1", n=2))
    for stage in ["candidate_generation", "feature_fetch", "ranking",
                  "business_rules", "feature_log"]:
        assert stage in resp.latency_ms
    assert resp.total_latency_ms >= 0


def test_business_rules_filter_out_of_stock():
    svc = _service(oos={"a"})
    resp = svc.recommend(RecommendationRequest(user_id="u3", n=10))
    assert "a" not in resp.items


def test_business_rules_filter_already_seen():
    svc = _service()
    resp = svc.recommend(RecommendationRequest(user_id="u1", n=10), already_seen={"b"})
    assert "b" not in resp.items


def test_service_ranker_reorders():
    # Fit a real ranker so ranking runs (not fallback).
    store = _store()
    training = store.get_historical_features(
        pl.DataFrame({
            "user_id": ["u1", "u2", "u3", "u1"],
            "item_id": ["a", "a", "b", "c"],
            "timestamp_ms": [10, 14, 15, 12],
        }).with_columns(pl.col("timestamp_ms").cast(pl.Int64))
    ).with_columns(pl.Series("label", [1, 1, 0, 0]))
    ranker = LRRanker().fit(training)
    cg = PopularityCandidateGenerator(store._item_totals, pool_size=10)
    svc = RecommendationService(cg, store, ranker=ranker)
    resp = svc.recommend(RecommendationRequest(user_id="u1", n=3))
    assert resp.fallback_used is False
    assert resp.model_version == "lr_ranker_v1"


def test_service_degrades_gracefully_on_ranker_error():
    class BrokenRanker:
        def rank(self, *a, **k):
            raise RuntimeError("boom")

    svc = _service(ranker=BrokenRanker())
    resp = svc.recommend(RecommendationRequest(user_id="u1", n=3))
    # No exception; fell back to candidate order.
    assert resp.fallback_used is True
    assert resp.model_version == "fallback"
    assert len(resp.items) == 3


def test_service_logs_inference_features_for_served_items():
    svc = _service()
    resp = svc.recommend(RecommendationRequest(user_id="u1", n=2))
    logged_items = {row["item_id"] for row in svc.feature_log}
    # Every served item appears in the feature log (Rule 29).
    assert set(resp.items).issubset(logged_items)
    # Logged rows carry the request_id and the feature values.
    for row in svc.feature_log:
        assert row["request_id"] == resp.request_id
        assert "item_pop" in row["features"]
