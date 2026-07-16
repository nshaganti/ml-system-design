"""
Phase 0 — Data Loading & Validation
=====================================
Loads the Retail Rocket dataset and maps it to our canonical event schema
(from the design doc). Run this first to verify your data download is clean.

Dataset download: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
Drop the 3 CSVs into: ../data/
  - events.csv
  - item_properties_part1.csv
  - item_properties_part2.csv
"""

from pathlib import Path
import polars as pl

DATA_DIR = Path(__file__).parent.parent / "data"

# Retail Rocket event types -> our canonical schema
EVENT_TYPE_MAP = {
    "view":       "impression",   # user saw the item
    "addtocart":  "add_to_cart",
    "transaction": "purchase",
}


def load_events(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    """
    Load events.csv and normalize to our canonical schema.

    Raw schema:  timestamp (ms), visitorid, event, itemid, transactionid
    Output:      timestamp_ms, user_id, event_type, item_id
    """
    path = data_dir / "events.csv"
    _assert_file_exists(path)

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
            # Cast to consistent types
            pl.col("user_id").cast(pl.Utf8),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("timestamp_ms").cast(pl.Int64),
        ])
        .drop("transactionid")
        .sort("timestamp_ms")
    )

    _validate_events(df)
    return df


def load_item_properties(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    """
    Load and combine both item_properties CSVs.
    Item properties are versioned (each row = property value at a point in time).
    This is key for demonstrating training-serving skew in Phase 2.

    Raw schema:  timestamp (ms), itemid, property, value
    """
    paths = [
        data_dir / "item_properties_part1.csv",
        data_dir / "item_properties_part2.csv",
    ]
    for p in paths:
        _assert_file_exists(p)

    df = (
        pl.concat([pl.read_csv(p) for p in paths])
        .rename({
            "timestamp": "timestamp_ms",
            "itemid":    "item_id",
        })
        .with_columns([
            pl.col("item_id").cast(pl.Utf8),
            pl.col("timestamp_ms").cast(pl.Int64),
        ])
        .sort("timestamp_ms")
    )
    return df


def get_item_snapshot(
    item_properties: pl.DataFrame,
    as_of_timestamp_ms: int,
) -> pl.DataFrame:
    """
    Get item property values AS OF a specific timestamp.
    For each (item_id, property), return the most recent value <= as_of_timestamp_ms.

    This is point-in-time correct feature lookup — the same concept the feature
    store (Phase 2) enforces automatically. Here we do it manually to illustrate
    what skew looks like when you DON'T do this.
    """
    return (
        item_properties
        .filter(pl.col("timestamp_ms") <= as_of_timestamp_ms)
        .sort("timestamp_ms", descending=True)
        .unique(subset=["item_id", "property"], keep="first")
        .pivot(on="property", index="item_id", values="value")
    )


def summarize(events: pl.DataFrame, item_props: pl.DataFrame) -> None:
    """Print a quick sanity check of the loaded data."""
    ts_min = events["timestamp_ms"].min()
    ts_max = events["timestamp_ms"].max()
    span_days = (ts_max - ts_min) / (1000 * 86400)

    print("=" * 50)
    print("EVENT DATA")
    print(f"  Total events      : {len(events):,}")
    print(f"  Unique users      : {events['user_id'].n_unique():,}")
    print(f"  Unique items      : {events['item_id'].n_unique():,}")
    print(f"  Time span         : {span_days:.1f} days")
    print(f"  Date range        : {ts_min} -> {ts_max} (ms)")
    print()
    print("  Event type breakdown:")
    counts = events.group_by("event_type").len().sort("len", descending=True)
    for row in counts.iter_rows(named=True):
        pct = row["len"] / len(events) * 100
        print(f"    {row['event_type']:<15} {row['len']:>8,}  ({pct:.1f}%)")
    print()
    print("ITEM PROPERTIES")
    print(f"  Total property records : {len(item_props):,}")
    print(f"  Unique items           : {item_props['item_id'].n_unique():,}")
    print(f"  Properties seen        : {item_props['property'].n_unique():,}")
    print("=" * 50)


def _assert_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing: {path}\n"
            "Download from: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset\n"
            "Place all 3 CSVs in the data/ directory."
        )


def _validate_events(df: pl.DataFrame) -> None:
    """Fail loudly on data quality problems (Rule 10: no silent failures)."""
    known_types = set(EVENT_TYPE_MAP.values())
    found_types = set(df["event_type"].unique().to_list())
    unknown = found_types - known_types
    if unknown:
        raise ValueError(f"Unknown event types found: {unknown}")

    null_counts = {col: df[col].null_count() for col in ["user_id", "item_id", "timestamp_ms"]}
    if any(v > 0 for v in null_counts.values()):
        raise ValueError(f"Null values in critical columns: {null_counts}")

    print(f"[load_data] Loaded {len(df):,} events. Schema OK.")


if __name__ == "__main__":
    events = load_events()
    item_props = load_item_properties()
    summarize(events, item_props)
