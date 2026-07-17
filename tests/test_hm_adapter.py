"""
H&M adapter tests -- prove the port maps to our canonical schema correctly,
using the synthetic generator (no 3.5 GB download needed). These lock in the
contract every phase relies on: canonical columns, purchase-only events, and a
"categoryid" property for the feature store.
"""

from __future__ import annotations

import polars as pl

from data_sources import hm


def _make(tmp_path):
    hm.generate_synthetic(tmp_path, n_customers=120, n_articles=60, n_days=20, seed=1)
    return tmp_path


def test_synthetic_generator_writes_three_csvs(tmp_path):
    _make(tmp_path)
    for f in ["transactions_train.csv", "articles.csv", "customers.csv"]:
        assert (tmp_path / f).exists()


def test_events_have_canonical_schema(tmp_path):
    events = hm.load_events(_make(tmp_path))
    assert set(events.columns) == {"timestamp_ms", "user_id", "event_type", "item_id"}
    assert events["timestamp_ms"].dtype == pl.Int64
    assert events["user_id"].dtype == pl.Utf8
    assert events["item_id"].dtype == pl.Utf8


def test_events_are_purchase_only(tmp_path):
    events = hm.load_events(_make(tmp_path))
    assert events["event_type"].unique().to_list() == ["purchase"]


def test_events_sorted_and_non_null(tmp_path):
    events = hm.load_events(_make(tmp_path))
    ts = events["timestamp_ms"].to_list()
    assert ts == sorted(ts)
    for c in ["timestamp_ms", "user_id", "item_id"]:
        assert events[c].null_count() == 0


def test_item_properties_expose_categoryid(tmp_path):
    props = hm.load_item_properties(_make(tmp_path))
    assert set(props.columns) == {"timestamp_ms", "item_id", "property", "value"}
    # The feature store's cross feature depends on a "categoryid" property.
    assert "categoryid" in props["property"].unique().to_list()
    # Static metadata is stamped at time 0 (known before every event).
    assert props["timestamp_ms"].max() == 0


def test_categoryid_covers_items(tmp_path):
    tmp = _make(tmp_path)
    events = hm.load_events(tmp)
    props = hm.load_item_properties(tmp)
    cats = props.filter(pl.col("property") == "categoryid")
    # Every article in transactions should have a category (articles is the catalog).
    covered = set(cats["item_id"].to_list())
    purchased = set(events["item_id"].to_list())
    assert purchased.issubset(covered)


def test_hm_max_rows_env_caps_events(tmp_path, monkeypatch):
    tmp = _make(tmp_path)
    full = hm.load_events(tmp)
    monkeypatch.setenv("HM_MAX_ROWS", "50")
    capped = hm.load_events(tmp)
    assert len(capped) == 50
    # Cap keeps the most RECENT rows -> max timestamp preserved.
    assert capped["timestamp_ms"].max() == full["timestamp_ms"].max()


def test_dispatcher_selects_hm(tmp_path, monkeypatch):
    _make(tmp_path)
    monkeypatch.setenv("DATASET", "hm")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import importlib
    import load_data
    importlib.reload(load_data)
    try:
        events = load_data.load_events()
        assert events["event_type"].unique().to_list() == ["purchase"]
    finally:
        monkeypatch.delenv("DATASET", raising=False)
        monkeypatch.delenv("DATA_DIR", raising=False)
        importlib.reload(load_data)
