"""
Phase 0 -- Data Loading & Validation (dataset dispatcher)
==========================================================
Loads a dataset and maps it to our canonical event schema:
    events:          [timestamp_ms, user_id, event_type, item_id]
    item_properties: [timestamp_ms, item_id, property, value]

The whole pipeline is dataset-agnostic: every phase calls load_events() /
load_item_properties() and never knows which dataset is underneath. Pick the
dataset with the DATASET environment variable:

    DATASET=retailrocket   (default)  -> data/events.csv + item_properties_*.csv
    DATASET=hm                        -> data/transactions_train.csv + articles.csv

    # examples
    cd phase0 && python run.py                 # retail rocket
    DATASET=hm python run.py                    # H&M (once CSVs are in data/)
    DATASET=hm HM_MAX_ROWS=2000000 python run.py  # H&M, capped for small machines

Adding a new dataset = drop a module in data_sources/ with load_events() and
load_item_properties(), then register it in _SOURCES below. No phase changes.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from data_sources import retailrocket, hm

# Data directory. Override with the DATA_DIR env var (handy for pointing at a
# synthetic/sample dataset without touching the real data/ folder).
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent / "data")))

_SOURCES = {
    "retailrocket": retailrocket,
    "hm": hm,
}


def _active_source():
    name = os.environ.get("DATASET", "retailrocket").lower()
    if name not in _SOURCES:
        raise ValueError(
            f"Unknown DATASET={name!r}. Options: {sorted(_SOURCES)}."
        )
    return name, _SOURCES[name]


def load_events(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    _, source = _active_source()
    return source.load_events(data_dir)


def load_item_properties(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    _, source = _active_source()
    return source.load_item_properties(data_dir)


def get_item_snapshot(item_properties: pl.DataFrame, as_of_timestamp_ms: int) -> pl.DataFrame:
    """
    Point-in-time correct item property lookup: for each (item_id, property),
    the most recent value <= as_of_timestamp_ms. (Illustrative -- the Phase 2
    feature store enforces this automatically.)
    """
    return (
        item_properties
        .filter(pl.col("timestamp_ms") <= as_of_timestamp_ms)
        .sort("timestamp_ms", descending=True)
        .unique(subset=["item_id", "property"], keep="first")
        .pivot(on="property", index="item_id", values="value")
    )


def summarize(events: pl.DataFrame, item_props: pl.DataFrame) -> None:
    name, _ = _active_source()
    ts_min, ts_max = events["timestamp_ms"].min(), events["timestamp_ms"].max()
    span_days = (ts_max - ts_min) / (1000 * 86400)

    print("=" * 50)
    print(f"EVENT DATA  (dataset={name})")
    print(f"  Total events      : {len(events):,}")
    print(f"  Unique users      : {events['user_id'].n_unique():,}")
    print(f"  Unique items      : {events['item_id'].n_unique():,}")
    print(f"  Time span         : {span_days:.1f} days")
    print()
    print("  Event type breakdown:")
    counts = events.group_by("event_type").len().sort("len", descending=True)
    for row in counts.iter_rows(named=True):
        pct = row["len"] / len(events) * 100
        print(f"    {row['event_type']:<15} {row['len']:>10,}  ({pct:.1f}%)")
    print()
    print("ITEM PROPERTIES")
    print(f"  Total property records : {len(item_props):,}")
    print(f"  Unique items           : {item_props['item_id'].n_unique():,}")
    print(f"  Properties seen        : {item_props['property'].n_unique():,}")
    print("=" * 50)


if __name__ == "__main__":
    events = load_events()
    item_props = load_item_properties()
    summarize(events, item_props)
