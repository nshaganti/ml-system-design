"""
Phase 0 — Heuristic Ranker
============================
Implements the popularity-based + category-affinity ranker from the design doc.
No ML. Ships fast, collects data, establishes the baseline we'll beat in Phase 1.

Google's Rule 1: Launch without ML.
Google's Rule 7: Encode domain knowledge as features, not discarded alternatives.
"""

import polars as pl
from dataclasses import dataclass, field


# Event weights for the popularity score.
# Purchases signal stronger intent than add-to-carts, which signal more than views.
EVENT_WEIGHTS = {
    "purchase":    3.0,
    "add_to_cart": 2.0,
    "impression":  1.0,
}

# How far back to look when computing item popularity.
POPULARITY_WINDOW_DAYS = 7


@dataclass
class HeuristicRanker:
    """
    Popularity-based ranker with optional category affinity personalization.

    Two components:
    1. Global popularity: weighted sum of recent interactions per item.
    2. Category filter: if user has history, restrict candidates to their top categories.

    This is deliberately simple. Its purpose is to ship, collect data, and provide
    a strong baseline for Phase 1 to beat.
    """

    # Built from training events — populated by fit()
    item_scores: pl.DataFrame = field(default_factory=pl.DataFrame)
    item_categories: pl.DataFrame = field(default_factory=pl.DataFrame)
    _fitted: bool = False

    def fit(
        self,
        events: pl.DataFrame,
        item_properties: pl.DataFrame,
        cutoff_timestamp_ms: int,
    ) -> "HeuristicRanker":
        """
        Compute item popularity scores using events in the window:
            [cutoff - POPULARITY_WINDOW_DAYS, cutoff]

        cutoff_timestamp_ms:
            Everything *before* this timestamp is "training data."
            This mirrors what a real system does: the model only knows about
            the past, never the future. Critical for preventing leakage (Rule 33).
        """
        window_start_ms = cutoff_timestamp_ms - (POPULARITY_WINDOW_DAYS * 86400 * 1000)

        train_events = events.filter(
            (pl.col("timestamp_ms") >= window_start_ms)
            & (pl.col("timestamp_ms") < cutoff_timestamp_ms)
        )

        if len(train_events) == 0:
            raise ValueError("No training events in the specified window. Check your cutoff timestamp.")

        # Weighted popularity score per item
        # Rule 7: this heuristic encodes real domain knowledge —
        # purchases matter more than views.
        self.item_scores = (
            train_events
            .with_columns(
                pl.col("event_type")
                  .replace(EVENT_WEIGHTS)
                  .cast(pl.Float64)
                  .alias("weight")
            )
            .group_by("item_id")
            .agg(pl.col("weight").sum().alias("popularity_score"))
            .sort("popularity_score", descending=True)
        )

        # Extract categoryid from item properties (point-in-time correct)
        category_props = item_properties.filter(
            (pl.col("property") == "categoryid")
            & (pl.col("timestamp_ms") < cutoff_timestamp_ms)
        )
        # For each item, take the most recent categoryid before cutoff
        self.item_categories = (
            category_props
            .sort("timestamp_ms", descending=True)
            .unique(subset=["item_id"], keep="first")
            .select(["item_id", "value"])
            .rename({"value": "category_id"})
        )

        self._fitted = True
        print(
            f"[HeuristicRanker] Fitted on {len(train_events):,} events | "
            f"{self.catalog_size:,} scored items | "
            f"{self.item_categories['item_id'].n_unique():,} items with categories"
        )
        return self

    @property
    def catalog_size(self) -> int:
        """Number of items the ranker can recommend. Used by evaluate.py for coverage."""
        return self.item_scores["item_id"].n_unique() if self._fitted else 0

    def recommend(
        self,
        user_id: str,
        user_events: pl.DataFrame,
        n: int = 20,
    ) -> list[str]:
        """
        Return top-n item IDs for a user.

        user_events: this user's historical events BEFORE the cutoff.
                     Pass an empty DataFrame for cold-start users.

        Strategy:
          - If user has history: filter to items in their top categories, rank by popularity.
          - If cold-start: return global top-n (no category filter).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before recommend().")

        ranked = self.item_scores

        # Category affinity: restrict to user's preferred categories
        if len(user_events) > 0:
            user_top_categories = self._get_user_top_categories(user_events, top_k=3)
            if user_top_categories:
                items_in_cats = self.item_categories.filter(
                    pl.col("category_id").is_in(user_top_categories)
                )
                ranked = ranked.join(items_in_cats.select("item_id"), on="item_id", how="inner")

            # Exclude items the user has already purchased
            already_bought = (
                user_events
                .filter(pl.col("event_type") == "purchase")
                ["item_id"]
                .unique()
                .to_list()
            )
            if already_bought:
                ranked = ranked.filter(~pl.col("item_id").is_in(already_bought))

        # If category filtering left us with fewer than n items, fall back to global top-n
        if len(ranked) < n:
            ranked = self.item_scores

        return ranked.head(n)["item_id"].to_list()

    def _get_user_top_categories(
        self,
        user_events: pl.DataFrame,
        top_k: int = 3,
    ) -> list[str]:
        """
        Infer top-k category preferences from a user's history.
        Purchases count more than views (same weighting as popularity score).
        """
        weighted_events = user_events.with_columns(
            pl.col("event_type").replace(EVENT_WEIGHTS).cast(pl.Float64).alias("weight")
        )

        items_with_cats = weighted_events.join(
            self.item_categories, on="item_id", how="inner"
        )

        if len(items_with_cats) == 0:
            return []

        top_cats = (
            items_with_cats
            .group_by("category_id")
            .agg(pl.col("weight").sum().alias("cat_score"))
            .sort("cat_score", descending=True)
            .head(top_k)
            ["category_id"]
            .to_list()
        )
        return top_cats
