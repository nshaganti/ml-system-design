"""
Phase 3 -- Candidate Generation (Stage 1 at serving time)
===========================================================
Turns the whole catalog into a small set of plausible candidates, fast.

The service depends on the CandidateGenerator *interface*, not a concrete class.
That's the whole point of the two-stage architecture: Stage 1 can evolve
independently. Today we ship a popularity generator (always available, no model
to train); tomorrow you drop in the Phase 1 two-tower ANN behind the same
`.generate()` method and nothing downstream changes.

    class TwoTowerCandidateGenerator:
        def generate(self, user_id, n): ...   # same signature -> plug-and-play
"""

from __future__ import annotations

from typing import Protocol

import polars as pl


class CandidateGenerator(Protocol):
    """Anything that can turn a user into a candidate list is a generator."""

    def generate(self, user_id: str, n: int) -> list[str]:
        ...


class PopularityCandidateGenerator:
    """
    Global most-popular items as of the cutoff. O(1) per request (precomputed
    list), always available, no cold-start problem. A perfectly reasonable Stage
    1 -- and the honest floor the two-tower must beat to justify its complexity.
    """

    def __init__(self, item_totals: pl.DataFrame, pool_size: int = 500):
        # item_totals: columns [item_id, item_pop]  (from the Phase 2 store)
        self._pool = (
            item_totals.sort("item_pop", descending=True)
            .head(pool_size)["item_id"]
            .to_list()
        )
        self.pool_size = len(self._pool)

    def generate(self, user_id: str, n: int = 500) -> list[str]:
        # Popularity is user-agnostic, so every user gets the same pool. The
        # ranker (Stage 2) is what personalizes it.
        return self._pool[:n]
