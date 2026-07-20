"""
Shared session-eval protocol (used by Phase 7 and Phase 10).
============================================================
Extracted so the co-visitation phase and the GRU4Rec phase grade themselves with
the EXACT same leave-one-out procedure -- same session definition, same test-case
construction, same reciprocal-rank. If the protocol lived in one phase's run.py,
the other couldn't reuse it without importing a module literally named `run`
(which collides with its own entry point). One obvious home instead (Zen).
"""

from __future__ import annotations

import polars as pl

from covisitation import sessionize, session_item_lists


def reciprocal_rank(recommended: list[str], target: str, k: int) -> float:
    """1/rank of the target within the top-k, else 0 (the MRR building block)."""
    for i, item in enumerate(recommended[:k], start=1):
        if item == target:
            return 1.0 / i
    return 0.0


def build_test_cases(test_events: pl.DataFrame) -> list[tuple[list[str], str]]:
    """
    Leave-one-out cases: (context_items, target_item) per multi-item session.
    Hide the last item of each session with >= 2 distinct items; the earlier items
    are the context. Deduplicated, time-order preserved.
    """
    sessions = sessionize(test_events)
    cases = []
    for items in session_item_lists(sessions):
        seen = list(dict.fromkeys(items))   # de-dup, keep order
        if len(seen) >= 2:
            cases.append((seen[:-1], seen[-1]))
    return cases
