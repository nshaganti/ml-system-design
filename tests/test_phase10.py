"""
Tests for Phase 10 -- GRU4Rec sequence model core (model.py, dataset.py).

Pure and dataset-free. The overfit test is the important one: it proves the model
can actually learn a next-item pattern, so a flat result in run.py would point at
the data/regime, not a broken model.
"""

from __future__ import annotations

import torch

import seq_data as ds
from model import GRU4Rec, PAD_IDX


# ------------------------------------------------------------------ dataset

def test_build_vocab_is_one_based_and_reserves_pad():
    lists = [["a", "b", "a"], ["c"]]
    vocab = ds.build_vocab(lists)
    assert set(vocab.values()) == {1, 2, 3}      # 0 reserved for PAD
    assert PAD_IDX == 0
    assert PAD_IDX not in vocab.values()


def test_build_vocab_min_count_filters():
    lists = [["a", "a", "b"]]                     # a:2, b:1
    vocab = ds.build_vocab(lists, min_count=2)
    assert "a" in vocab and "b" not in vocab


def test_encode_context_left_pads_and_drops_oov():
    vocab = {"a": 1, "b": 2}
    enc = ds.encode_context(["a", "zzz", "b"], vocab, max_len=4)
    assert enc == [0, 0, 1, 2]                    # OOV dropped, left-padded


def test_encode_context_truncates_to_most_recent():
    vocab = {"a": 1, "b": 2, "c": 3}
    enc = ds.encode_context(["a", "b", "c"], vocab, max_len=2)
    assert enc == [2, 3]                          # keeps the last two


def test_make_training_pairs_generates_prefix_targets():
    vocab = {"a": 1, "b": 2, "c": 3}
    pairs = ds.make_training_pairs([["a", "b", "c"]], vocab, max_len=3)
    # (a -> b) and (a b -> c)
    assert pairs[0] == ([0, 0, 1], 2)
    assert pairs[1] == ([0, 1, 2], 3)


def test_make_training_pairs_respects_cap():
    vocab = {"a": 1, "b": 2, "c": 3, "d": 4}
    pairs = ds.make_training_pairs([["a", "b", "c", "d"]], vocab, max_len=4, max_pairs=2)
    assert len(pairs) == 2


# ------------------------------------------------------------------ model

def test_forward_shape_is_batch_by_vocab_plus_one():
    model = GRU4Rec(vocab_size=5, embedding_dim=8, hidden_dim=8)
    out = model(torch.tensor([[0, 1, 2], [0, 0, 3]], dtype=torch.long))
    assert out.shape == (2, 6)                    # vocab_size + 1 (PAD row)


def test_recommend_excludes_pad_and_excluded_items():
    model = GRU4Rec(vocab_size=5, embedding_dim=8, hidden_dim=8)
    recs = model.recommend([0, 0, 1], k=5, exclude={2})
    assert PAD_IDX not in recs
    assert 2 not in recs
    assert len(recs) == len(set(recs))            # no dupes


def test_model_can_overfit_a_simple_next_item_pattern():
    """Train on 1->2 and 2->3; the model should learn to predict them."""
    torch.manual_seed(0)
    vocab = {"a": 1, "b": 2, "c": 3}
    pairs = ds.make_training_pairs([["a", "b", "c"]] * 64, vocab, max_len=3)
    X = torch.tensor([p for p, _ in pairs], dtype=torch.long)
    y = torch.tensor([t for _, t in pairs], dtype=torch.long)

    model = GRU4Rec(vocab_size=3, embedding_dim=16, hidden_dim=16, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()

    # After overfitting, context [.. a] should rank b (idx 2) first.
    top = model.recommend(ds.encode_context(["a"], vocab, 3), k=1)
    assert top[0] == 2
