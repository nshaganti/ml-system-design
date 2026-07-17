"""
data_sources.retailrocket -- Retail Rocket loader
===================================================
The original dataset. Kept behind the same interface as every other data source
so the dispatcher (load_data.py) can swap datasets without any phase code
changing.

Download: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
Drop into data/: events.csv, item_properties_part1.csv, item_properties_part2.csv
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from signals import WEAK, MEDIUM, STRONG

# Retail Rocket native events -> canonical signal taxonomy (see phase0/signals.py)
EVENT_TYPE_MAP = {
    "view":        WEAK,     # exposure
    "addtocart":   MEDIUM,   # engagement
    "transaction": STRONG,   # target action
}
KNOWN_EVENT_TYPES = {WEAK, MEDIUM, STRONG}


def load_events(data_dir: Path) -> pl.DataFrame:
    """events.csv -> canonical [timestamp_ms, user_id, event_type, item_id]."""
    path = data_dir / "events.csv"
    _assert_file_exists(path, "retailrocket")

    df = (
        pl.read_csv(path)
        .rename({
            "timestamp":  "timestamp_ms",
            "visitorid":  "user_id",
            "event":      "event_type",
            "itemid":     "item_id",
        })
        .with_columns([
            pl.col("event_type").replace(EVENT_TYPE_MAP),
            pl.col("user_id").cast(pl.Utf8),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("timestamp_ms").cast(pl.Int64),
        ])
        .drop("transactionid")
        .sort("timestamp_ms")
    )
    _validate_events(df, KNOWN_EVENT_TYPES)
    return df


def load_item_properties(data_dir: Path) -> pl.DataFrame:
    """
    Combine both versioned item_properties CSVs -> long
    [timestamp_ms, item_id, property, value]. The per-timestamp versioning here
    is what makes Retail Rocket good for illustrating skew.
    """
    paths = [
        data_dir / "item_properties_part1.csv",
        data_dir / "item_properties_part2.csv",
    ]
    for p in paths:
        _assert_file_exists(p, "retailrocket")

    return (
        pl.concat([pl.read_csv(p) for p in paths])
        .rename({"timestamp": "timestamp_ms", "itemid": "item_id"})
        .with_columns([
            pl.col("item_id").cast(pl.Utf8),
            pl.col("timestamp_ms").cast(pl.Int64),
        ])
        .sort("timestamp_ms")
    )


def _assert_file_exists(path: Path, dataset: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing: {path}\n"
            "Download: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset\n"
            "Place all 3 CSVs in the data/ directory."
        )


def _validate_events(df: pl.DataFrame, known: set[str]) -> None:
    unknown = set(df["event_type"].unique().to_list()) - known
    if unknown:
        raise ValueError(f"Unknown event types found: {unknown}")
    nulls = {c: df[c].null_count() for c in ["user_id", "item_id", "timestamp_ms"]}
    if any(v > 0 for v in nulls.values()):
        raise ValueError(f"Null values in critical columns: {nulls}")
    print(f"[load_data] (retailrocket) Loaded {len(df):,} events. Schema OK.")
