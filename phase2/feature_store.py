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

Features computed (all point-in-time correct):
  - item_pop          : # strong (cart/purchase) events on the item, before T
  - user_pop          : # strong events by the user, before T
  - user_cat_affinity : # of the user's prior strong events in the SAME category
                        as the candidate item (a user x item CROSS feature, Rule 20)

Why the cross feature matters: item_pop and user_pop are each constant across one
axis (item_pop is the same for every user; user_pop is the same for every item),
so a ranker built only on them cannot PERSONALIZE -- per user it just reproduces
popularity order. user_cat_affinity varies per (user, item) pair, which is what
lets the ranker actually tailor results.

polars `join_asof(strategy="backward")` is exactly the right primitive: for each
event at time T it grabs the most recent timeline row with timestamp <= T.
"""

from __future__ import annotations

import warnings

import polars as pl

from signals import POSITIVE_SIGNALS

# polars emits an informational warning on join_asof when a `by` group is given
# (it can't verify sortedness cheaply). We always sort on timestamp_ms right
# before each join, so this is safe to silence for clean output.
warnings.filterwarnings(
    "ignore", message="Sortedness of columns cannot be checked"
)

# What counts as a positive interaction for popularity features. Sourced from the
# signal taxonomy (Rule 7) so Phase 2 stays domain-neutral.
STRONG_EVENT_TYPES = list(POSITIVE_SIGNALS)

FEATURE_COLUMNS = ["item_pop", "user_pop", "user_cat_affinity"]


class PointInTimeFeatureStore:
    """
    Minimal point-in-time feature store.

    fit() builds per-item, per-user, and per-(user, category) event timelines
    (cumulative counts over time). Those timelines are queried three ways:
      - get_historical_features : point-in-time correct (the RIGHT way)
      - get_online_features     : latest values as of the cutoff (serving)
      - get_skewed_features     : the WRONG way, for demonstrating skew
    """

    def __init__(self) -> None:
        self._item_timeline: pl.DataFrame | None = None
        self._user_timeline: pl.DataFrame | None = None
        self._user_cat_timeline: pl.DataFrame | None = None
        self._item_totals: pl.DataFrame | None = None
        self._user_totals: pl.DataFrame | None = None
        self._user_cat_totals: pl.DataFrame | None = None
        self._item_category: pl.DataFrame | None = None
        self._cutoff_ms: int | None = None
        self._fitted = False

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        events: pl.DataFrame,
        cutoff_timestamp_ms: int,
        item_properties: pl.DataFrame | None = None,
    ) -> "PointInTimeFeatureStore":
        """
        Build feature timelines from strong events strictly BEFORE the cutoff.
        The store only knows the past -- exactly like a real system at the moment
        it generates training data.

        item_properties (optional): the Retail Rocket item-property log. When
        provided, enables the user_cat_affinity cross feature by mapping each item
        to its category. When None, user_cat_affinity is present but always 0.
        """
        strong = events.filter(
            pl.col("event_type").is_in(STRONG_EVENT_TYPES)
            & (pl.col("timestamp_ms") < cutoff_timestamp_ms)
        )

        # Single-axis popularity timelines
        self._item_timeline = self._build_timeline(strong, ["item_id"], "item_pop")
        self._user_timeline = self._build_timeline(strong, ["user_id"], "user_pop")
        self._item_totals = strong.group_by("item_id").agg(pl.len().alias("item_pop"))
        self._user_totals = strong.group_by("user_id").agg(pl.len().alias("user_pop"))

        # Item -> category map (most recent categoryid before the cutoff)
        self._item_category = self._build_item_category(item_properties, cutoff_timestamp_ms)

        # Cross-feature timeline: per (user, category) cumulative prior counts
        strong_cat = strong.join(self._item_category, on="item_id", how="inner")
        self._user_cat_timeline = self._build_timeline(
            strong_cat, ["user_id", "category_id"], "user_cat_affinity"
        )
        self._user_cat_totals = (
            strong_cat.group_by(["user_id", "category_id"])
            .agg(pl.len().alias("user_cat_affinity"))
        )

        self._cutoff_ms = cutoff_timestamp_ms
        self._fitted = True
        print(
            f"[feature_store] Fitted on {len(strong):,} strong events | "
            f"{len(self._item_totals):,} items | {len(self._user_totals):,} users | "
            f"{self._item_category.height:,} items w/ category"
        )
        return self

    @staticmethod
    def _build_timeline(strong: pl.DataFrame, keys: list[str], feat: str) -> pl.DataFrame:
        """
        Timeline of PRIOR cumulative counts per entity (grouped by `keys`).

        For the k-th (0-based) event of an entity in time order, the prior count
        is k -- i.e. how many strong events that entity had BEFORE this one.
        A backward as-of join against this timeline therefore yields the count of
        events strictly before the query timestamp (no leakage of the current
        event or any future event).
        """
        return (
            strong
            .select(keys + ["timestamp_ms"])
            .sort("timestamp_ms")
            .with_columns(pl.int_range(0, pl.len()).over(keys).alias(feat))
            .select(["timestamp_ms"] + keys + [feat])
        )

    @staticmethod
    def _build_item_category(
        item_properties: pl.DataFrame | None,
        cutoff_ms: int,
    ) -> pl.DataFrame:
        """Most-recent categoryid per item, as of just before the cutoff."""
        empty = pl.DataFrame(
            {"item_id": [], "category_id": []},
            schema={"item_id": pl.Utf8, "category_id": pl.Utf8},
        )
        if item_properties is None:
            return empty
        cat = item_properties.filter(
            (pl.col("property") == "categoryid")
            & (pl.col("timestamp_ms") < cutoff_ms)
        )
        if cat.height == 0:
            return empty
        return (
            cat.sort("timestamp_ms", descending=True)
            .unique(subset=["item_id"], keep="first")
            .select(["item_id", pl.col("value").alias("category_id")])
        )

    # --------------------------------------------------- historical (correct)

    def get_historical_features(self, entity_df: pl.DataFrame) -> pl.DataFrame:
        """
        Point-in-time correct join (Rule 29). entity_df needs user_id, item_id,
        and timestamp_ms. Returns entity_df + item_pop + user_pop +
        user_cat_affinity as they existed at each event's timestamp. Entities with
        no prior events get 0.
        """
        self._require_fitted()

        # Attach each candidate item's category first (needed for the cross join),
        # then re-sort because a regular join may reorder rows.
        base = (
            entity_df.sort("timestamp_ms")
            .join(self._item_category, on="item_id", how="left")
            .sort("timestamp_ms")
        )

        base = base.join_asof(
            self._item_timeline.sort("timestamp_ms"),
            on="timestamp_ms", by="item_id", strategy="backward",
        )
        base = base.join_asof(
            self._user_timeline.sort("timestamp_ms"),
            on="timestamp_ms", by="user_id", strategy="backward",
        )
        base = base.join_asof(
            self._user_cat_timeline.sort("timestamp_ms"),
            on="timestamp_ms", by=["user_id", "category_id"], strategy="backward",
        )
        return base.with_columns(
            pl.col("item_pop").fill_null(0),
            pl.col("user_pop").fill_null(0),
            pl.col("user_cat_affinity").fill_null(0),
        )

    # ------------------------------------------------------- online (serving)

    def get_online_features(self, user_id: str, item_id: str) -> dict:
        """Latest feature values as of the cutoff -- what serving would fetch."""
        self._require_fitted()
        item_pop = self._lookup(self._item_totals, ["item_id"], [item_id], "item_pop")
        user_pop = self._lookup(self._user_totals, ["user_id"], [user_id], "user_pop")
        category = self._lookup_str(self._item_category, item_id)
        affinity = 0
        if category is not None:
            affinity = self._lookup(
                self._user_cat_totals, ["user_id", "category_id"],
                [user_id, category], "user_cat_affinity",
            )
        return {"item_pop": item_pop, "user_pop": user_pop, "user_cat_affinity": affinity}

    def get_online_features_batch(self, entity_df: pl.DataFrame) -> pl.DataFrame:
        """Vectorized online lookup for many (user_id, item_id) rows."""
        self._require_fitted()
        return (
            entity_df
            .join(self._item_category, on="item_id", how="left")
            .join(self._item_totals, on="item_id", how="left")
            .join(self._user_totals, on="user_id", how="left")
            .join(self._user_cat_totals, on=["user_id", "category_id"], how="left")
            .with_columns(
                pl.col("item_pop").fill_null(0),
                pl.col("user_pop").fill_null(0),
                pl.col("user_cat_affinity").fill_null(0),
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
        # The skewed path is exactly the online (cutoff-total) join applied to
        # historical rows -- which is precisely the bug.
        return self.get_online_features_batch(entity_df)

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _lookup(totals: pl.DataFrame, keys: list[str], values: list[str], feat: str) -> int:
        pred = pl.lit(True)
        for k, v in zip(keys, values):
            pred = pred & (pl.col(k) == v)
        row = totals.filter(pred)
        return int(row[feat][0]) if len(row) else 0

    @staticmethod
    def _lookup_str(mapping: pl.DataFrame, item_id: str) -> str | None:
        row = mapping.filter(pl.col("item_id") == item_id)
        return row["category_id"][0] if len(row) else None

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before querying the feature store.")
