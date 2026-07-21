"""
Tests for Phase 13 -- TwoTowerScorer (scorer.py).

Uses a hand-built 3-item embedding matrix and a fake vocab so the similarity
maths is checkable by hand -- no model training. Proves: user vector = mean of
history embeddings, score = dot product, cold user / OOV item -> 0.0 (neutral),
and the serving `augment` hook adds a tt_score column in candidate order.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from scorer import TwoTowerScorer, TT_SCORE_COL


class FakeVocab:
    def __init__(self, mapping):
        self._m = mapping                      # item_id -> idx (or absent = OOV)

    def encode(self, item_id):
        return self._m.get(item_id)


def _scorer():
    # 3 items in a 2-D embedding space, easy to reason about.
    emb = np.array([[1.0, 0.0],   # i0
                    [0.0, 1.0],   # i1
                    [1.0, 1.0]])  # i2
    vocab = FakeVocab({"i0": 0, "i1": 1, "i2": 2})
    return TwoTowerScorer(emb, vocab)


def test_user_vector_is_mean_of_history_embeddings():
    s = _scorer()
    v = s.user_vector_from_idx([0, 1])         # mean of (1,0) and (0,1)
    assert np.allclose(v, [0.5, 0.5])


def test_score_is_dot_product():
    s = _scorer()
    v = s.user_vector_from_idx([0])            # user vector = (1,0)
    assert s.score(v, "i0") == 1.0             # dot with (1,0)
    assert s.score(v, "i1") == 0.0             # dot with (0,1)
    assert s.score(v, "i2") == 1.0             # dot with (1,1)


def test_cold_user_and_oov_item_score_zero():
    s = _scorer()
    assert s.score(None, "i0") == 0.0          # cold user
    v = s.user_vector_from_idx([2])
    assert s.score(v, "unknown") == 0.0        # OOV item


def test_score_training_rows_uses_per_user_vectors():
    s = _scorer()
    user_vecs = {"u1": np.array([1.0, 0.0]), "u2": np.array([0.0, 1.0])}
    df = pl.DataFrame({"user_id": ["u1", "u2", "u3"], "item_id": ["i2", "i2", "i2"]})
    scores = s.score_training_rows(df, user_vecs)
    # u1.(1,1)=1, u2.(1,1)=1, u3 missing -> 0
    assert list(scores) == [1.0, 1.0, 0.0]


def test_augment_adds_tt_score_in_candidate_order():
    s = _scorer()
    events = pl.DataFrame({"item_id": ["i0"], "timestamp_ms": [1]})   # user vec (1,0)
    feats = pl.DataFrame({"item_id": ["i1", "i2", "i0"], "user_id": ["u"] * 3})
    out = s.augment("u", events, feats)
    assert TT_SCORE_COL in out.columns
    assert out[TT_SCORE_COL].to_list() == [0.0, 1.0, 1.0]            # dot with (1,0)
