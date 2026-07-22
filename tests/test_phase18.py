"""
Tests for Phase 18 -- SASRec model core (sasrec.py).

Light and fast: no training, no data load. We check the model's *contract* -- output
shape, PAD handling, causal masking, and that recommend() honors exclusions -- which
is what the eval harness relies on. Whether attention actually wins is an empirical
question answered by run.py, not a unit test.
"""

from __future__ import annotations

import torch

from sasrec import SASRec, PAD_IDX


def _model(vocab=10, max_len=6):
    torch.manual_seed(0)
    return SASRec(vocab_size=vocab, max_len=max_len, embedding_dim=16,
                  num_blocks=2, num_heads=2, dropout=0.0)


def test_forward_shape_is_vocab_plus_pad():
    m = _model(vocab=10, max_len=6)
    x = torch.randint(1, 11, (4, 6))         # batch of 4, len 6, real items
    out = m(x)
    assert out.shape == (4, 11)              # vocab_size + 1 (PAD row)


def test_pad_embedding_is_frozen_zero():
    m = _model()
    assert torch.allclose(m.item_emb.weight[PAD_IDX], torch.zeros(16))


def test_forward_is_finite_with_left_padding():
    m = _model(vocab=10, max_len=6)
    # left-padded: PADs first, one real item last (the common cold-ish case)
    x = torch.tensor([[0, 0, 0, 0, 0, 3]])
    out = m(x)
    assert torch.isfinite(out).all()         # padding must not produce NaN/inf


def test_recommend_excludes_pad_and_history():
    m = _model(vocab=10, max_len=6)
    seq = [0, 0, 0, 1, 2, 3]
    recs = m.recommend(seq, k=5, exclude={1, 2, 3})
    assert PAD_IDX not in recs
    assert not ({1, 2, 3} & set(recs))       # excluded history never recommended
    assert len(recs) == 5
    assert all(1 <= i <= 10 for i in recs)


def test_causal_mask_last_position_ignores_future_only():
    # Changing an EARLIER item should affect the last-position prediction; the last
    # position legitimately attends to all earlier items (that's the point).
    m = _model(vocab=10, max_len=4)
    m.eval()
    a = m(torch.tensor([[0, 5, 6, 7]]))
    b = m(torch.tensor([[0, 8, 6, 7]]))      # changed an earlier item
    assert not torch.allclose(a, b)          # earlier context matters


def test_deterministic_in_eval_mode():
    m = _model()
    m.eval()
    x = torch.tensor([[0, 0, 1, 2, 3, 4]])
    assert torch.allclose(m(x), m(x))        # no dropout noise at eval
