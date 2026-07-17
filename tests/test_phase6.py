"""
Phase 6 tests: the streaming feature store.

What matters: streamed deltas add correctly on top of batch features, the online
interface stays identical to the batch store (so the service works unchanged),
session tracking feeds the freshness rule, and only STRONG events move the
popularity counters.
"""

from __future__ import annotations

import polars as pl

from feature_store import PointInTimeFeatureStore
from streaming_store import StreamingFeatureStore


def _events():
    rows = [
        ("u1", "purchase",    "a", 10),
        ("u2", "add_to_cart", "b", 11),
        ("u3", "purchase",    "a", 12),
        ("u1", "purchase",    "b", 13),
    ]
    return pl.DataFrame(
        {
            "user_id":      [r[0] for r in rows],
            "event_type":   [r[1] for r in rows],
            "item_id":      [r[2] for r in rows],
            "timestamp_ms": [r[3] for r in rows],
        }
    ).with_columns(pl.col("timestamp_ms").cast(pl.Int64))


def _item_props():
    rows = [("a", "categoryid", "CAT_A", 1), ("b", "categoryid", "CAT_B", 1)]
    return pl.DataFrame(
        {
            "item_id":      [r[0] for r in rows],
            "property":     [r[1] for r in rows],
            "value":        [r[2] for r in rows],
            "timestamp_ms": [r[3] for r in rows],
        }
    ).with_columns(pl.col("timestamp_ms").cast(pl.Int64))


def _batch():
    return PointInTimeFeatureStore().fit(
        _events(), cutoff_timestamp_ms=1000, item_properties=_item_props()
    )


# ---------------------------------------------------------------- deltas


def test_streamed_event_bumps_features():
    sfs = StreamingFeatureStore(_batch())
    before = sfs.get_online_features("u1", "a")
    sfs.ingest("u1", "a", "purchase")
    after = sfs.get_online_features("u1", "a")
    assert after["item_pop"] == before["item_pop"] + 1
    assert after["user_pop"] == before["user_pop"] + 1
    assert after["user_cat_affinity"] == before["user_cat_affinity"] + 1


def test_before_ingest_matches_batch():
    batch = _batch()
    sfs = StreamingFeatureStore(batch)
    # With no streamed events, the streaming store must equal the batch store.
    assert sfs.get_online_features("u1", "a") == batch.get_online_features("u1", "a")


def test_only_strong_events_move_popularity():
    sfs = StreamingFeatureStore(_batch())
    before = sfs.get_online_features("u1", "a")
    sfs.ingest("u1", "a", "view")   # weak event -> no popularity change
    after = sfs.get_online_features("u1", "a")
    assert after["item_pop"] == before["item_pop"]
    assert after["user_pop"] == before["user_pop"]
    # ...but it IS tracked for session dedup.
    assert "a" in sfs.session_seen("u1")


def test_cross_feature_only_bumps_matching_category():
    sfs = StreamingFeatureStore(_batch())
    sfs.ingest("u1", "a", "purchase")   # item a is CAT_A
    # affinity to CAT_A (item a) went up; affinity to CAT_B (item b) did not.
    assert sfs.get_online_features("u1", "a")["user_cat_affinity"] >= 1
    b_before = _batch().get_online_features("u1", "b")["user_cat_affinity"]
    assert sfs.get_online_features("u1", "b")["user_cat_affinity"] == b_before


def test_session_seen_tracks_all_items():
    sfs = StreamingFeatureStore(_batch())
    sfs.ingest("u1", "a", "purchase")
    sfs.ingest("u1", "b", "view")
    assert sfs.session_seen("u1") == {"a", "b"}
    assert sfs.session_seen("nobody") == set()


# ------------------------------------------------------ batch interface parity


def test_batch_interface_parity():
    sfs = StreamingFeatureStore(_batch())
    entity = pl.DataFrame({"user_id": ["u1", "u2"], "item_id": ["a", "b"]})
    out = sfs.get_online_features_batch(entity)
    # Same columns as the batch store -> the service consumes it unchanged.
    for col in ["item_pop", "user_pop", "user_cat_affinity"]:
        assert col in out.columns
    assert out.height == 2


def test_batch_read_reflects_deltas():
    sfs = StreamingFeatureStore(_batch())
    entity = pl.DataFrame({"user_id": ["u1"], "item_id": ["a"]})
    before = sfs.get_online_features_batch(entity)["item_pop"][0]
    sfs.ingest("u1", "a", "purchase")
    after = sfs.get_online_features_batch(entity)["item_pop"][0]
    assert after == before + 1


def test_ingest_frame_replays_in_order():
    sfs = StreamingFeatureStore(_batch())
    sfs.ingest_frame(_events())
    # 3 strong events on the stream touched item a twice, b once (purchase).
    assert sfs.events_ingested == 4
    assert sfs.get_online_features("u1", "a")["item_pop"] >= 2
