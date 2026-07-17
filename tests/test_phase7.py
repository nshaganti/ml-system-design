"""
Phase 7 tests: sessionization + co-visitation.

The load-bearing logic is the 30-minute session boundary and the co-occurrence
scoring -- get those wrong and the whole benchmark is meaningless.
"""

from __future__ import annotations

import polars as pl

from covisitation import (
    sessionize,
    session_item_lists,
    CoVisitationRecommender,
    SESSION_GAP_MS,
)


def _df(rows):
    return pl.DataFrame(
        {
            "user_id":      [r[0] for r in rows],
            "item_id":      [r[1] for r in rows],
            "event_type":   ["view"] * len(rows),
            "timestamp_ms": [r[2] for r in rows],
        }
    ).with_columns(pl.col("timestamp_ms").cast(pl.Int64))


# ------------------------------------------------------------ sessionize


def test_gap_splits_sessions():
    # Two events close together, then one after a >30min gap -> 2 sessions.
    rows = [("u1", "a", 0), ("u1", "b", 1000), ("u1", "c", SESSION_GAP_MS + 2000)]
    s = sessionize(_df(rows))
    ids = s.sort("timestamp_ms")["session_id"].to_list()
    assert ids[0] == ids[1]        # a, b same session
    assert ids[2] != ids[1]        # c is a new session


def test_different_users_get_different_sessions():
    rows = [("u1", "a", 0), ("u2", "b", 500)]
    s = sessionize(_df(rows))
    assert s["session_id"].n_unique() == 2


def test_session_item_lists_are_time_ordered():
    rows = [("u1", "a", 0), ("u1", "b", 100), ("u1", "c", 200)]
    lists = session_item_lists(sessionize(_df(rows)))
    assert lists == [["a", "b", "c"]]


# --------------------------------------------------------- co-visitation


def test_covisitation_learns_pairs():
    # Two sessions both contain (a, b); a-b should co-occur strongly.
    rows = [
        ("u1", "a", 0), ("u1", "b", 100),
        ("u2", "a", 0), ("u2", "b", 100),
        ("u3", "a", 0), ("u3", "c", 100),
    ]
    rec = CoVisitationRecommender().fit(_df(rows))
    # Given a session with 'a', 'b' co-occurred twice, 'c' once -> b ranks first.
    recs = rec.recommend(["a"], n=2)
    assert recs[0] == "b"
    assert "a" not in recs           # context item excluded


def test_covisitation_backs_off_to_popularity():
    rows = [("u1", "a", 0), ("u1", "b", 100), ("u2", "a", 0), ("u2", "b", 100)]
    rec = CoVisitationRecommender().fit(_df(rows))
    # An item with no co-vis history -> fall back to popularity, not crash.
    recs = rec.recommend(["zzz_unknown"], n=5)
    assert isinstance(recs, list)
    assert "zzz_unknown" not in recs


def test_covisitation_excludes_session_and_extra():
    rows = [("u1", "a", 0), ("u1", "b", 100), ("u1", "c", 200)] * 1
    rec = CoVisitationRecommender().fit(_df(rows))
    recs = rec.recommend(["a"], n=10, exclude={"b"})
    assert "a" not in recs and "b" not in recs


def test_popular_returns_most_frequent():
    rows = [("u1", "a", 0), ("u2", "a", 0), ("u3", "a", 0), ("u4", "b", 0)]
    rec = CoVisitationRecommender().fit(_df(rows))
    assert rec.popular(n=1) == ["a"]


def test_recommend_respects_n():
    rows = [("u1", "a", 0), ("u1", "b", 100), ("u1", "c", 200), ("u1", "d", 300)]
    rec = CoVisitationRecommender().fit(_df(rows))
    assert len(rec.recommend(["a"], n=2)) <= 2
