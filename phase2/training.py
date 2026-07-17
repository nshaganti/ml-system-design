"""
Phase 2 -- Shared Training-Set Construction
=============================================
Builds a labelled (user, item, features, label) table for the ranker, with
point-in-time-correct features. Extracted here so Phases 2, 3, and 4 all build
training data the SAME way (DRY -- one definition of "what a training row is").

  positives = positive-signal events (engagement or stronger), label 1
  negatives = a random item at the same (user, timestamp), label 0
  features  = store.get_historical_features(...)  <- point-in-time correct
"""

from __future__ import annotations

import numpy as np
import polars as pl

from signals import POSITIVE_SIGNALS


def build_labelled_features(
    train_events: pl.DataFrame,
    store,                       # PointInTimeFeatureStore (already fitted)
    rng: np.random.Generator,
    max_positives: int = 100_000,
    seed: int = 42,
) -> pl.DataFrame:
    strong = train_events.filter(pl.col("event_type").is_in(list(POSITIVE_SIGNALS)))
    if len(strong) > max_positives:
        strong = strong.sample(max_positives, seed=seed)
    all_items = train_events["item_id"].unique().to_list()

    pos = strong.select(["user_id", "item_id", "timestamp_ms"]).with_columns(
        pl.lit(1).alias("label")
    )
    neg_items = rng.choice(np.array(all_items), size=len(pos))
    neg = pos.select(["user_id", "timestamp_ms"]).with_columns(
        pl.Series("item_id", neg_items).cast(pl.Utf8),
        pl.lit(0).alias("label"),
    ).select(["user_id", "item_id", "timestamp_ms", "label"])

    return store.get_historical_features(pl.concat([pos, neg]))
