"""
data_sources.kuairand -- the KuaiRand-Pure dataset (real logs)
==============================================================
KuaiRand is a Kuaishou short-video recommendation dataset. What makes it special
-- and why this whole repo is built on it -- is that it ships TWO logs:

  * log_standard_*  : interactions served by the PRODUCTION recommender (biased)
  * log_random_*    : interactions served by a UNIFORM-RANDOM policy (unbiased)

The random log is the gold: because we KNOW the logging policy (uniform), we know
each impression's propensity, which lets us do honest off-policy evaluation
(Part II / Phase 8). The standard log is the everyday biased data every classic
recsys phase (Part I) trains and evaluates on.

Native schema (per interaction row):
    user_id, video_id, time_ms, is_click, is_like, is_follow, is_comment,
    is_forward, is_hate, long_view, play_time_ms, duration_ms, ..., is_rand, tab

We normalize every row into the canonical WEAK/MEDIUM/STRONG signal taxonomy
(see phase0/signals.py):

    STRONG  a target action    long_view OR like OR follow OR forward OR comment
    MEDIUM  an engagement       is_click == 1 (but no strong action)
    WEAK    a mere exposure     shown, not clicked

Item category comes from video_features_basic.tag (a comma-separated list; we take
the first tag as the primary categoryid).
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from signals import WEAK, MEDIUM, STRONG

KNOWN_EVENT_TYPES = {WEAK, MEDIUM, STRONG}

# Columns that, when set, mark a STRONG (target-action) signal.
_STRONG_FLAGS = ["long_view", "is_like", "is_follow", "is_forward", "is_comment"]

_STANDARD_LOGS = [
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
]
_RANDOM_LOG = "log_random_4_22_to_5_08_pure.csv"
_VIDEO_FEATURES = "video_features_basic_pure.csv"


def _kuairand_dir(data_dir: Path | None) -> Path:
    """Resolve the folder holding the KuaiRand CSVs."""
    base = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent / "data"
    # The GitHub archive nests the CSVs under KuaiRand-Pure/data/.
    for candidate in (base / "KuaiRand-Pure" / "data", base / "KuaiRand-Pure", base):
        if (candidate / _RANDOM_LOG).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find KuaiRand CSVs (looked for {_RANDOM_LOG!r} under {base}). "
        "Download KuaiRand-Pure and place it in data/KuaiRand-Pure/."
    )


def _to_signals(df: pl.DataFrame) -> pl.DataFrame:
    """Map native interaction flags -> canonical events with a signal level."""
    strong_expr = pl.any_horizontal([pl.col(c) == 1 for c in _STRONG_FLAGS])
    return (
        df.with_columns(
            pl.when(strong_expr)
            .then(pl.lit(STRONG))
            .when(pl.col("is_click") == 1)
            .then(pl.lit(MEDIUM))
            .otherwise(pl.lit(WEAK))
            .alias("event_type")
        )
        .select(
            pl.col("time_ms").cast(pl.Int64).alias("timestamp_ms"),
            pl.col("user_id").cast(pl.Utf8),
            pl.col("event_type"),
            pl.col("video_id").cast(pl.Utf8).alias("item_id"),
        )
        .sort("timestamp_ms")
    )


def _read_log(path: Path, max_rows: int | None) -> pl.DataFrame:
    cols = ["user_id", "video_id", "time_ms", "is_click", *_STRONG_FLAGS]
    df = pl.read_csv(path, columns=cols, n_rows=max_rows)
    return df


def _max_rows() -> int | None:
    v = os.environ.get("KUAIRAND_MAX_ROWS")
    return int(v) if v else None


def load_events(data_dir: Path | None = None) -> pl.DataFrame:
    """Canonical events from the STANDARD (biased, production-policy) logs."""
    d = _kuairand_dir(data_dir)
    frames = [_read_log(d / name, _max_rows()) for name in _STANDARD_LOGS]
    events = _to_signals(pl.concat(frames))
    counts = events["event_type"].value_counts()
    print(f"[load_data] (kuairand) {len(events):,} standard-log events "
          f"(WEAK/MEDIUM/STRONG). Signal mix: {dict(zip(counts['event_type'], counts['count']))}")
    return events


def load_item_properties(data_dir: Path | None = None) -> pl.DataFrame:
    """Canonical item properties: categoryid from the first video tag."""
    d = _kuairand_dir(data_dir)
    vids = pl.read_csv(d / _VIDEO_FEATURES, columns=["video_id", "tag"])
    props = (
        vids.with_columns(
            # tag can be a comma-separated list like "20,1,3"; take the first.
            pl.col("tag").cast(pl.Utf8).str.split(",").list.first().alias("value")
        )
        .drop_nulls("value")
        .select(
            pl.lit(0).cast(pl.Int64).alias("timestamp_ms"),
            pl.col("video_id").cast(pl.Utf8).alias("item_id"),
            pl.lit("categoryid").alias("property"),
            pl.col("value"),
        )
    )
    return props


def load_random_log(data_dir: Path | None = None) -> pl.DataFrame:
    """
    The UNIFORM-RANDOM exposure log -- Part II's off-policy-evaluation gold.

    Returns canonical events PLUS a `propensity` column: because the logging
    policy is uniform-random over the item space, every shown item had the same
    probability of being shown, p = 1 / (number of distinct items exposed).
    That known propensity is what makes IPS / SNIPS / doubly-robust estimators
    unbiased here.
    """
    d = _kuairand_dir(data_dir)
    raw = _read_log(d / _RANDOM_LOG, _max_rows())
    events = _to_signals(raw)
    n_items = events["item_id"].n_unique()
    propensity = 1.0 / n_items
    events = events.with_columns(pl.lit(propensity).alias("propensity"))
    print(f"[load_data] (kuairand) {len(events):,} random-log events "
          f"(uniform propensity = 1/{n_items} = {propensity:.2e}).")
    return events
