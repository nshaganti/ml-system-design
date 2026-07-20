"""
Phase 7 -- Session-Based Co-Visitation (the community-standard approach)
=========================================================================
Why this phase exists: the recsys community's classic sweet spot for interaction
logs is SESSION-BASED next-item prediction, and the consistently
strongest-yet-simplest method on this kind of data is **co-visitation** (a.k.a.
item-kNN over session co-occurrence):

    "people who interacted with X in a session also interacted with Y"

Phases 0-6 modelled a user's *whole* history (mean-pooled) and predicted next
*target action* over the full catalog. That is a different, harder task than the
one the literature benchmarks. On KuaiRand the session signal is real but modest
(co-visitation beats popularity by ~61%), because the small catalog makes
popularity a strong session baseline. This module measures it honestly.

Performance notes (this is pure Python over millions of events, so it matters):
  - fit() uses a SLIDING WINDOW, not all-pairs -- co-visitation quality comes from
    *nearby* items, and it turns O(session_len^2) into O(session_len * window).
  - after fit we precompute each item's TOP-M neighbors and a global popularity
    ranking, so recommend() is O(context * M) with no per-call sorting of huge
    Counters.
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict

import polars as pl

SESSION_GAP_MS = 30 * 60 * 1000     # 30 minutes of inactivity ends a session
MAX_SESSION_LEN = 30                # guard against bot/scraper mega-sessions
WINDOW = 5                          # co-visit items within +/- WINDOW positions
TOP_M_NEIGHBORS = 100               # keep only each item's strongest neighbors


def sessionize(events: pl.DataFrame, gap_ms: int = SESSION_GAP_MS) -> pl.DataFrame:
    """
    Add a `session_id` column. A new session starts at each user's first event or
    after a gap of > gap_ms since their previous event.
    """
    df = events.sort(["user_id", "timestamp_ms"])
    df = df.with_columns(
        (pl.col("timestamp_ms").diff().over("user_id") > gap_ms)
        .fill_null(True)          # first event per user starts a session
        .alias("_new_session")
    )
    return df.with_columns(pl.col("_new_session").cum_sum().alias("session_id"))


def session_item_lists(sessions: pl.DataFrame) -> list[list[str]]:
    """Time-ordered item list per session (input must already be sessionized)."""
    grouped = (
        sessions.sort("timestamp_ms")
        .group_by("session_id")
        .agg(pl.col("item_id").alias("items"))
    )
    return [row["items"] for row in grouped.iter_rows(named=True)]


class CoVisitationRecommender:
    """
    Item-item co-visitation scorer with a sliding window and top-M pruning.

    fit(): count windowed co-occurrence per session, then prune to each item's
    top-M neighbors. recommend(): sum the (pruned) neighbor vectors of the current
    session's items and return the highest-scoring other items.
    """

    def __init__(self, window: int = WINDOW, max_session_len: int = MAX_SESSION_LEN,
                 top_m: int = TOP_M_NEIGHBORS):
        self.window = window
        self.max_session_len = max_session_len
        self.top_m = top_m
        self._covis: dict[str, dict[str, int]] = {}
        self._popularity: Counter = Counter()
        self._pop_ranked: list[str] = []
        self.n_items = 0

    def fit(self, events: pl.DataFrame) -> "CoVisitationRecommender":
        raw: dict[str, Counter] = defaultdict(Counter)
        sessions = sessionize(events)
        for items in session_item_lists(sessions):
            seen = list(dict.fromkeys(items))[: self.max_session_len]
            self._popularity.update(seen)
            n = len(seen)
            for i, a in enumerate(seen):
                # Only pair with items within the forward window; symmetric update.
                for j in range(i + 1, min(i + 1 + self.window, n)):
                    b = seen[j]
                    if a == b:
                        continue
                    raw[a][b] += 1
                    raw[b][a] += 1

        # Prune to top-M neighbors per item (dict for O(1) lookups at serve time).
        self._covis = {
            item: dict(counter.most_common(self.top_m))
            for item, counter in raw.items()
        }
        self._pop_ranked = [it for it, _ in self._popularity.most_common()]
        self.n_items = len(self._popularity)
        print(f"[covisitation] Fitted: {self.n_items:,} items, "
              f"{sum(len(c) for c in self._covis.values()):,} pruned edges "
              f"(window={self.window}, top_m={self.top_m})")
        return self

    def recommend(self, session_items: list[str], n: int = 20,
                  exclude: set[str] | None = None) -> list[str]:
        exclude = set(exclude or set()) | set(session_items)
        scores: dict[str, int] = defaultdict(int)
        for item in session_items:
            for neighbor, w in self._covis.get(item, {}).items():
                if neighbor not in exclude:
                    scores[neighbor] += w
        if scores:
            ranked = heapq.nlargest(n, scores.items(), key=lambda kv: kv[1])
            return [it for it, _ in ranked]
        # Cold session (no co-vis signal) -> back off to global popularity.
        return self.popular(n, exclude=exclude)

    def popular(self, n: int, exclude: set[str] | None = None) -> list[str]:
        """Global most-popular items -- the baseline to beat under session eval."""
        exclude = exclude or set()
        out = []
        for it in self._pop_ranked:
            if it not in exclude:
                out.append(it)
                if len(out) >= n:
                    break
        return out
