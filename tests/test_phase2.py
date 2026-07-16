"""
Phase 2 tests: point-in-time feature store + LR ranker.

The feature-store tests are the important ones -- they lock in leakage-free
point-in-time semantics (Rule 29) and prove the skewed path is inflated.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from feature_store import PointInTimeFeatureStore, FEATURE_COLUMNS
from lr_ranker import LRRanker, to_matrix


def _events():
    """
    Item 'x' gets strong events at t=10, 20, 30 (from three different users).
    Item 'y' gets one strong event at t=15.
    """
    rows = [
        ("u1", "purchase",    "x", 10),
        ("u2", "add_to_cart", "y", 15),
        ("u2", "purchase",    "x", 20),
        ("u3", "purchase",    "x", 30),
        # a view that must NOT count toward popularity
        ("u4", "impression",  "x", 5),
    ]
    return pl.DataFrame(
        {
            "user_id":      [r[0] for r in rows],
            "event_type":   [r[1] for r in rows],
            "item_id":      [r[2] for r in rows],
            "timestamp_ms": [r[3] for r in rows],
        }
    ).with_columns(pl.col("timestamp_ms").cast(pl.Int64))


# ------------------------------------------------------------- point-in-time


def test_point_in_time_counts_prior_events_only():
    store = PointInTimeFeatureStore().fit(_events(), cutoff_timestamp_ms=1000)

    # Query item 'x' at each of its event timestamps: item_pop must equal the
    # number of strong events STRICTLY BEFORE that time (leakage-free).
    entity = pl.DataFrame(
        {
            "user_id":      ["u1", "u2", "u3"],
            "item_id":      ["x", "x", "x"],
            "timestamp_ms": [10, 20, 30],
        }
    ).with_columns(pl.col("timestamp_ms").cast(pl.Int64))

    out = store.get_historical_features(entity).sort("timestamp_ms")
    assert out["item_pop"].to_list() == [0, 1, 2]


def test_point_in_time_before_any_event_is_zero():
    store = PointInTimeFeatureStore().fit(_events(), cutoff_timestamp_ms=1000)
    entity = pl.DataFrame(
        {"user_id": ["u1"], "item_id": ["x"], "timestamp_ms": [1]}
    ).with_columns(pl.col("timestamp_ms").cast(pl.Int64))
    out = store.get_historical_features(entity)
    assert out["item_pop"].to_list() == [0]
    assert out["user_pop"].to_list() == [0]


def test_views_do_not_count_as_popularity():
    # Item x has a view at t=5; at t=10 (its first strong event) item_pop is 0.
    store = PointInTimeFeatureStore().fit(_events(), cutoff_timestamp_ms=1000)
    entity = pl.DataFrame(
        {"user_id": ["u1"], "item_id": ["x"], "timestamp_ms": [10]}
    ).with_columns(pl.col("timestamp_ms").cast(pl.Int64))
    assert store.get_historical_features(entity)["item_pop"].to_list() == [0]


# -------------------------------------------------------------------- skew


def test_skewed_features_are_inflated_vs_point_in_time():
    store = PointInTimeFeatureStore().fit(_events(), cutoff_timestamp_ms=1000)
    entity = pl.DataFrame(
        {"user_id": ["u1"], "item_id": ["x"], "timestamp_ms": [10]}
    ).with_columns(pl.col("timestamp_ms").cast(pl.Int64))

    correct = store.get_historical_features(entity)["item_pop"][0]
    skewed = store.get_skewed_features(entity)["item_pop"][0]
    # point-in-time at t=10 = 0 prior; skewed = full total (3) -> leakage.
    assert correct == 0
    assert skewed == 3
    assert skewed > correct


# ------------------------------------------------------------------ online


def test_online_features_return_cutoff_totals():
    store = PointInTimeFeatureStore().fit(_events(), cutoff_timestamp_ms=1000)
    feats = store.get_online_features(user_id="u2", item_id="x")
    assert feats["item_pop"] == 3     # x has 3 strong events total
    assert feats["user_pop"] == 2     # u2 had 2 strong events (y@15, x@20)


def test_online_features_unknown_entity_is_zero():
    store = PointInTimeFeatureStore().fit(_events(), cutoff_timestamp_ms=1000)
    feats = store.get_online_features(user_id="ghost", item_id="nope")
    assert feats == {"item_pop": 0, "user_pop": 0}


def test_feature_store_requires_fit():
    store = PointInTimeFeatureStore()
    with pytest.raises(RuntimeError):
        store.get_online_features("u", "i")


# -------------------------------------------------------------- LR ranker


def _labelled_df():
    # Clear signal: high item_pop/user_pop -> positive.
    return pl.DataFrame(
        {
            "item_id":  [f"i{i}" for i in range(8)],
            "item_pop": [100, 90, 80, 70, 1, 2, 3, 0],
            "user_pop": [50, 40, 30, 20, 0, 1, 0, 1],
            "label":    [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )


def test_to_matrix_applies_log1p():
    df = pl.DataFrame({"item_pop": [0.0, np.e - 1], "user_pop": [0.0, 0.0]})
    X = to_matrix(df, ["item_pop", "user_pop"])
    # log1p(0) = 0 ; log1p(e-1) = 1
    assert np.isclose(X[0, 0], 0.0)
    assert np.isclose(X[1, 0], 1.0)


def test_lr_ranker_learns_and_ranks():
    ranker = LRRanker().fit(_labelled_df())
    # Higher-popularity items should rank above low-popularity ones.
    top = ranker.rank(_labelled_df(), item_col="item_id", n=4)
    assert set(top) == {"i0", "i1", "i2", "i3"}


def test_lr_ranker_needs_both_classes():
    df = pl.DataFrame({"item_id": ["a", "b"], "item_pop": [1, 2], "user_pop": [0, 0], "label": [1, 1]})
    with pytest.raises(ValueError):
        LRRanker().fit(df)


def test_lr_ranker_rank_before_fit_raises():
    with pytest.raises(RuntimeError):
        LRRanker().rank(_labelled_df())
