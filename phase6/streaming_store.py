"""
Phase 6 -- Near-Real-Time Freshness (Rule 8)
==============================================
> "A user who just bought running shoes should not see more running shoes
>  immediately after."

The batch feature store (Phase 2) is frozen at the training cutoff. That's fine
for a daily-retrained model -- but it means a user's behavior in the last 5
minutes is invisible to their next recommendation. The design doc's answer is
NOT online learning (retraining on the stream is fragile); it's simpler:

    stream fresh FEATURES to the online store, and keep the same model.

    User buys item -> Kafka -> Flink updates Redis (~30s lag) -> next request
    sees the updated features. No retrain required.

This module implements that idea. `StreamingFeatureStore` wraps the frozen batch
store and layers incremental, O(1) feature updates on top of it. Because it
exposes the SAME `get_online_features_batch` interface as the batch store, the
Phase 3 `RecommendationService` consumes it with zero changes -- the whole point
of programming to an interface (Phase 3).

It also tracks per-session interactions so the serving layer can apply the
"don't re-show what you just interacted with" freshness rule.
"""

from __future__ import annotations

from collections import defaultdict

import polars as pl

STRONG_EVENT_TYPES = ("add_to_cart", "purchase")


class StreamingFeatureStore:
    """
    Frozen batch features + a live delta layer updated by the event stream.

    Each ingested strong event bumps three counters in O(1):
      item_pop[item], user_pop[user], user_cat_affinity[(user, item's category)]

    Reads return batch value + streamed delta -- fresh features, no retrain.
    """

    def __init__(self, batch_store):
        self.batch = batch_store
        self._item_delta: dict[str, int] = defaultdict(int)
        self._user_delta: dict[str, int] = defaultdict(int)
        self._user_cat_delta: dict[tuple[str, str], int] = defaultdict(int)
        self._session_items: dict[str, set[str]] = defaultdict(set)
        self.events_ingested = 0

    # ---------------------------------------------------------- ingest

    def ingest(self, user_id: str, item_id: str, event_type: str) -> None:
        """Apply one streamed event. This is the Flink job, in miniature."""
        self._session_items[user_id].add(item_id)
        if event_type in STRONG_EVENT_TYPES:
            self._item_delta[item_id] += 1
            self._user_delta[user_id] += 1
            cat = self._category_of(item_id)
            if cat is not None:
                self._user_cat_delta[(user_id, cat)] += 1
        self.events_ingested += 1

    def ingest_frame(self, events: pl.DataFrame) -> None:
        """Convenience: replay a batch of events through ingest() in order."""
        for r in events.sort("timestamp_ms").iter_rows(named=True):
            self.ingest(r["user_id"], r["item_id"], r["event_type"])

    def session_seen(self, user_id: str) -> set[str]:
        """Items this user has touched in-session (for the freshness rule)."""
        return set(self._session_items.get(user_id, set()))

    # ----------------------------------------------------------- reads

    def get_online_features_batch(self, entity_df: pl.DataFrame) -> pl.DataFrame:
        """Same interface as the batch store, but with streamed deltas added."""
        base = self.batch.get_online_features_batch(entity_df)
        users = base["user_id"].to_list()
        items = base["item_id"].to_list()
        cats = base["category_id"].to_list()

        item_extra = [self._item_delta.get(i, 0) for i in items]
        user_extra = [self._user_delta.get(u, 0) for u in users]
        cat_extra = [
            self._user_cat_delta.get((u, c), 0) if c is not None else 0
            for u, c in zip(users, cats)
        ]
        return base.with_columns(
            (pl.col("item_pop") + pl.Series(item_extra)).alias("item_pop"),
            (pl.col("user_pop") + pl.Series(user_extra)).alias("user_pop"),
            (pl.col("user_cat_affinity") + pl.Series(cat_extra)).alias("user_cat_affinity"),
        )

    def get_online_features(self, user_id: str, item_id: str) -> dict:
        base = self.batch.get_online_features(user_id, item_id)
        cat = self._category_of(item_id)
        base["item_pop"] += self._item_delta.get(item_id, 0)
        base["user_pop"] += self._user_delta.get(user_id, 0)
        if cat is not None:
            base["user_cat_affinity"] += self._user_cat_delta.get((user_id, cat), 0)
        return base

    # --------------------------------------------------------- helpers

    def _category_of(self, item_id: str) -> str | None:
        return self.batch._lookup_str(self.batch._item_category, item_id)
