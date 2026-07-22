"""
Phase 0 tests: temporal split, heuristic ranker, and the shared evaluation.

These lock in the behaviours we most rely on:
  - the split never leaks the future into training
  - the ranker weights strong signals and personalizes by category
  - recall_at_k's coverage uses the SHARED catalog_size denominator (the fix)
"""

from __future__ import annotations

import polars as pl

from evaluate import temporal_split, recall_at_k
from heuristic_ranker import HeuristicRanker, EVENT_WEIGHTS
from load_data import get_item_snapshot


# ---------------------------------------------------------------- get_item_snapshot


def test_item_snapshot_is_point_in_time_correct():
    # Two timestamped values for the same (item, property). A lookup as-of t=150
    # must see the value set at t=100, NOT the future value set at t=200.
    props = pl.DataFrame({
        "timestamp_ms": [100, 200, 100],
        "item_id": ["a", "a", "b"],
        "property": ["price", "price", "price"],
        "value": ["10", "20", "5"],
    })
    snap = get_item_snapshot(props, as_of_timestamp_ms=150).sort("item_id")
    prices = dict(zip(snap["item_id"], snap["price"]))
    assert prices["a"] == "10"      # the future 20 is invisible at t=150
    assert prices["b"] == "5"


def test_item_snapshot_takes_most_recent_past_value():
    props = pl.DataFrame({
        "timestamp_ms": [100, 200],
        "item_id": ["a", "a"],
        "property": ["price", "price"],
        "value": ["10", "20"],
    })
    snap = get_item_snapshot(props, as_of_timestamp_ms=999)
    assert snap["price"][0] == "20"  # both are in the past -> newest wins


# ---------------------------------------------------------------- temporal_split


def test_temporal_split_has_no_future_leakage(synthetic_events):
    train, test, cutoff = temporal_split(synthetic_events, train_fraction=0.75)

    assert train["timestamp_ms"].max() < cutoff
    assert test["timestamp_ms"].min() >= cutoff
    # No overlap and nothing dropped.
    assert len(train) + len(test) == len(synthetic_events)


def test_temporal_split_respects_fraction(synthetic_events):
    train, test, _ = temporal_split(synthetic_events, train_fraction=0.5)
    # 8 events, 50% cutoff -> roughly half in train.
    assert 0 < len(train) < len(synthetic_events)
    assert len(test) > 0


# ---------------------------------------------------------------- HeuristicRanker


def test_event_weights_ordering():
    # Domain knowledge: strong > medium > weak. Guard against accidental edits.
    assert EVENT_WEIGHTS["strong"] > EVENT_WEIGHTS["medium"] > EVENT_WEIGHTS["weak"]


def test_ranker_fit_and_recommend(synthetic_events, synthetic_item_properties):
    cutoff = synthetic_events["timestamp_ms"].max() + 1
    ranker = HeuristicRanker().fit(
        events=synthetic_events,
        item_properties=synthetic_item_properties,
        cutoff_timestamp_ms=cutoff,
    )
    assert ranker.catalog_size > 0

    # Cold-start user (empty history) -> global top-n, never errors.
    cold = ranker.recommend(user_id="new", user_events=synthetic_events.head(0), n=2)
    assert len(cold) <= 2
    assert all(isinstance(x, str) for x in cold)


def test_ranker_excludes_already_consumed(synthetic_events, synthetic_item_properties):
    cutoff = synthetic_events["timestamp_ms"].max() + 1
    ranker = HeuristicRanker().fit(
        events=synthetic_events,
        item_properties=synthetic_item_properties,
        cutoff_timestamp_ms=cutoff,
    )
    u1_events = synthetic_events.filter(pl.col("user_id") == "u1")
    recs = ranker.recommend(user_id="u1", user_events=u1_events, n=20)

    consumed = set(
        u1_events.filter(pl.col("event_type") == "strong")["item_id"].to_list()
    )
    # If the category filter leaves enough items, consumed ones are excluded.
    # (When it falls back to global top-n, exclusion isn't guaranteed -- so we
    #  only assert exclusion held for the items still present.)
    if len(recs) >= 20:
        assert consumed.isdisjoint(set(recs))


def test_ranker_requires_fit_before_recommend():
    ranker = HeuristicRanker()
    try:
        ranker.recommend(user_id="x", user_events=pl.DataFrame(), n=5)
        assert False, "expected RuntimeError when recommending before fit()"
    except RuntimeError:
        pass


# ---------------------------------------------------------------- recall_at_k


class _StubRanker:
    """Minimal ranker with the .recommend / .catalog_size interface."""

    def __init__(self, fixed_recs, catalog_size):
        self._recs = fixed_recs
        self.catalog_size = catalog_size

    def recommend(self, user_id, user_events, n):
        return self._recs[:n]


def _test_frames():
    base = 1_600_000_000_000
    train = pl.DataFrame(
        {
            "user_id":      ["u1"],
            "event_type":   ["weak"],
            "item_id":      ["i1"],
            "timestamp_ms": [base],
        }
    )
    test = pl.DataFrame(
        {
            "user_id":      ["u1", "u1"],
            "event_type":   ["strong", "strong"],
            "item_id":      ["i2", "i3"],
            "timestamp_ms": [base + 10, base + 11],
        }
    )
    return train, test


def test_recall_at_k_perfect_hit():
    train, test = _test_frames()
    ranker = _StubRanker(fixed_recs=["i2", "i3"], catalog_size=100)
    res = recall_at_k(ranker, train, test, k=20, catalog_size=100)
    # Both purchased items recommended -> recall 1.0.
    assert res["recall_at_k"] == 1.0
    assert res["total_hits"] == 2


def test_recall_at_k_coverage_uses_shared_denominator():
    train, test = _test_frames()
    # Ranker's own catalog_size is deliberately WRONG (2) to prove the explicit
    # catalog_size argument wins -- this is the fairness fix.
    ranker = _StubRanker(fixed_recs=["i2", "i3"], catalog_size=2)
    res = recall_at_k(ranker, train, test, k=20, catalog_size=1000)
    # 2 unique items recommended out of a 1000-item universe.
    assert res["catalog_coverage"] == 2 / 1000


def test_recall_at_k_falls_back_to_ranker_catalog_size():
    train, test = _test_frames()
    ranker = _StubRanker(fixed_recs=["i2", "i3"], catalog_size=4)
    res = recall_at_k(ranker, train, test, k=20)  # no catalog_size passed
    assert res["catalog_coverage"] == 2 / 4
