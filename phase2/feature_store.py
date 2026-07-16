"""
Phase 2 -- Point-in-Time Feature Store
========================================
The single most under-taught concept in production ML (design doc, Phase 2).

A feature store provides two guarantees:

  1. Point-in-time correctness at TRAINING time -- when you join features to a
     historical event, you get the feature value AS IT EXISTED at the moment of
     that event, never a value computed from data that arrived later.
  2. Low-latency lookup at SERVING time -- the latest feature values.

This module implements a minimal version of both over polars, plus a
deliberately-broken `get_skewed_features()` so you can SEE training-serving
skew: the bug where training joins today's feature values onto month-old events,
making offline eval look great and production quietly degrade (Rules 29-37).

Features computed (both point-in-time correct):
  - item_pop : number of strong (cart/purchase) events on the item, before T
  - user_pop : number of strong events by the user, before T

polars `join_asof(strategy="backward")` is exactly the right primitive: for each
event at time T it grabs the most recent timeline row with timestamp <= T.
"""

from __future__ import annotations

import polars as pl

# What counts as a "strong" interaction for popularity features. Kept local so
# Phase 2 stays decoupled from Phase 1's training code.
STRONG_EVENT_TYPES = ["add_to_cart", "purchase"]


class PointInTimeFeatureStore:
    """
    Minimal point-in-time feature store.

    fit() builds per-item and per-user event timelines (cumulative counts over
    time). Those timelines are then queried three ways:
      - get_historical_features : point-in-time correct (the RIGHT way)
      - get_online_features     : latest values as of the cutoff (serving)
      - get_skewed_features     : the WRONG way, for demonstrating skew
    """

    def __init__(self) -> None:
        self._item_timeline: pl.DataFrame | None = None
        self._user_timeline: pl.DataFrame | None = None
        self._item_totals: pl.DataFrame | None = None
        self._user_totals: pl.DataFrame | None = None
        self._cutoff_ms: int | None = None
        self._fitted = False

    # ------------------------------------------------------------------ fit

    def fit(self, events: pl.DataFrame, cutoff_timestamp_ms: int) -> "PointInTimeFeatureStore":
        """
        Build feature timelines from strong events strictly BEFORE the cutoff.
        The store only knows the past -- exactly like a real system at the moment
        it generates training data.
        """
        strong = events.filter(
            pl.col("event_type").is_in(STRONG_EVENT_TYPES)
            & (pl.col("timestamp_ms") < cutoff_timestamp_ms)
        )

        self._item_timeline = self._build_timeline(strong, key="item_id", feat="item_pop")
        self._user_timeline = self._build_timeline(strong, key="user_id", feat="user_pop")

        # "As of cutoff" totals for online serving = full count per entity.
        self._item_totals = strong.group_by("item_id").agg(pl.len().alias("item_pop"))
        self._user_totals = strong.group_by("user_id").agg(pl.len().alias("user_pop"))

        self._cutoff_ms = cutoff_timestamp_ms
        self._fitted = True
        print(
            f"[feature_store] Fitted on {len(strong):,} strong events | "
            f"{len(self._item_totals):,} items | {len(self._user_totals):,} users"
        )
        return self

    @staticmethod
    def _build_timeline(strong: pl.DataFrame, key: str, feat: str) -> pl.DataFrame:
        """
        Timeline of PRIOR cumulative counts per entity.

        For the k-th (0-based) event of an entity in time order, the prior count
        is k -- i.e. how many strong events that entity had BEFORE this one.
        A backward as-of join against this timeline therefore yields the count of
        events strictly before the query timestamp (no leakage of the current
        event or any future event).
        """
        return (
            strong
            .select([key, "timestamp_ms"])
            .sort("timestamp_ms")
            .with_columns(pl.int_range(0, pl.len()).over(key).alias(feat))
            .select(["timestamp_ms", key, feat])
        )

    # --------------------------------------------------- historical (correct)

    def get_historical_features(self, entity_df: pl.DataFrame) -> pl.DataFrame:
        """
        Point-in-time correct join (Rule 29). entity_df needs user_id, item_id,
        and timestamp_ms. Returns entity_df + item_pop + user_pop as they existed
        at each event's timestamp. Entities with no prior events get 0.
        """
        self._require_fitted()
        base = entity_df.sort("timestamp_ms")

        out = base.join_asof(
            self._item_timeline.sort("timestamp_ms"),
            on="timestamp_ms",
            by="item_id",
            strategy="backward",
        )
        out = out.join_asof(
            self._user_timeline.sort("timestamp_ms"),
            on="timestamp_ms",
            by="user_id",
            strategy="backward",
        )
        return out.with_columns(
            pl.col("item_pop").fill_null(0),
            pl.col("user_pop").fill_null(0),
        )

    # ------------------------------------------------------- online (serving)

    def get_online_features(self, user_id: str, item_id: str) -> dict:
        """Latest feature values as of the cutoff -- what serving would fetch."""
        self._require_fitted()
        item_pop = self._lookup(self._item_totals, "item_id", item_id, "item_pop")
        user_pop = self._lookup(self._user_totals, "user_id", user_id, "user_pop")
        return {"item_pop": item_pop, "user_pop": user_pop}

    def get_online_features_batch(self, entity_df: pl.DataFrame) -> pl.DataFrame:
        """Vectorized online lookup for many (user_id, item_id) rows."""
        self._require_fitted()
        return (
            entity_df
            .join(self._item_totals, on="item_id", how="left")
            .join(self._user_totals, on="user_id", how="left")
            .with_columns(
                pl.col("item_pop").fill_null(0),
                pl.col("user_pop").fill_null(0),
            )
        )

    # ----------------------------------------------------- skewed (the bug)

    def get_skewed_features(self, entity_df: pl.DataFrame) -> pl.DataFrame:
        """
        The ANTI-PATTERN (Rules 29-32). Joins the CUTOFF-time totals onto every
        historical event, ignoring when each event happened. A month-old event
        gets today's popularity count -- the model trains on a feature
        distribution it will never see at serving time. Offline metrics look
        great; production silently degrades.

        Provided so run.py can quantify the skew vs get_historical_features.
        """
        self._require_fitted()
        return (
            entity_df
            .join(self._item_totals, on="item_id", how="left")
            .join(self._user_totals, on="user_id", how="left")
            .with_columns(
                pl.col("item_pop").fill_null(0),
                pl.col("user_pop").fill_null(0),
            )
        )

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _lookup(totals: pl.DataFrame, key: str, value: str, feat: str) -> int:
        row = totals.filter(pl.col(key) == value)
        return int(row[feat][0]) if len(row) else 0

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before querying the feature store.")


FEATURE_COLUMNS = ["item_pop", "user_pop"]
