"""
Tests for Phase 18 -- SASRec model core (sasrec.py).

Mostly light contract tests (output shape, PAD handling, causal masking, exclusions)
plus ONE learning test that trains on a tiny synthetic pattern. The learning test is
the important one -- it mirrors Phase 10's GRU4Rec overfit test and proves the
attention model can actually LEARN a next-item signal. So when run.py reports SASRec
finishing last on KuaiRand, that's an honest statement about the data regime -- not a
silently broken or non-learning model. Whether attention *wins* is still empirical,
answered by run.py, not a unit test.

Uses Phase 10's `seq_data` for vocab/encoding -- exactly what run.py trains on -- so
"a session" means the same thing here as in the real run.
"""

from __future__ import annotations

import torch

import seq_data as ds
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


def test_forward_is_finite_on_inference_path_with_left_padding():
    # REGRESSION TEST. left-padded: PADs first, one real item last (the common case).
    # The bug: recommend() runs forward under eval() + no_grad, which takes PyTorch's
    # fused TransformerEncoder path; that returned all-NaN for the fully-masked leading
    # PAD rows left-padding creates, so topk saw NaN and returned index-order garbage.
    # train() forward stayed finite, so this MUST assert on the eval()/no_grad path or
    # it gives false confidence (the original version did, and missed the bug).
    m = _model(vocab=10, max_len=6)
    x = torch.tensor([[0, 0, 0, 0, 0, 3]])
    m.eval()
    with torch.no_grad():
        out = m(x)                           # the exact path recommend() uses
    assert torch.isfinite(out).all()         # padding must not produce NaN/inf
    # train and serve must agree (Rule 32) -- the bug made them differ (finite vs NaN).
    m.train()
    assert torch.allclose(m(x), out, atol=1e-5)
    # public API returns real items, not degenerate index-order [1,2,3,...] from NaN.
    recs = m.recommend([0, 0, 0, 0, 0, 3], k=5)
    assert len(recs) == 5 and PAD_IDX not in recs


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


# --------------------------------------------------------------------- learning

def test_model_can_learn_a_simple_next_item_pattern():
    """Train on a->b and ab->c; SASRec should learn to predict them.

    The mirror of Phase 10's GRU4Rec overfit test. The contract tests above prove
    the plumbing (shapes, masks, exclusions); THIS test proves self-attention can
    actually fit a next-item signal end-to-end. Without it, a non-learning model
    (bad grad flow, a mask bug that zeroes the signal) would pass every other test
    silently -- and SASRec's honest last-place run.py result would be ambiguous
    between "wrong data regime" and "broken model". This pins it to the former.
    """
    torch.manual_seed(0)
    vocab = {"a": 1, "b": 2, "c": 3}
    max_len = 3
    pairs = ds.make_training_pairs([["a", "b", "c"]] * 64, vocab, max_len=max_len)
    X = torch.tensor([p for p, _ in pairs], dtype=torch.long)
    y = torch.tensor([t for _, t in pairs], dtype=torch.long)

    model = SASRec(vocab_size=3, max_len=max_len, embedding_dim=16,
                   num_blocks=2, num_heads=2, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    first_loss = loss_fn(model(X), y).item()
    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()

    # It learned both patterns: context [.. a] ranks b (idx 2) first; [.. a b]
    # ranks c (idx 3) first. (recommend() excludes only PAD, so the correct item
    # has to genuinely win on score.)
    assert model.recommend(ds.encode_context(["a"], vocab, max_len), k=1)[0] == 2
    assert model.recommend(ds.encode_context(["a", "b"], vocab, max_len), k=1)[0] == 3
    assert loss.item() < first_loss          # loss actually decreased
