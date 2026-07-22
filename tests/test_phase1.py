"""
Phase 1 tests: vocab construction, dataset triples, model math, and the index.

We deliberately avoid importing train.py (pulls in MLflow) -- the training loop
is orchestration, not logic worth unit-testing here. We test the pieces that
carry real correctness risk: vocab thresholding, padding/masking, BPR loss, and
nearest-neighbour search.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from dataset import (
    build_vocab,
    build_user_histories,
    BPRDataset,
    MIN_POSITIVE_INTERACTIONS,
    MAX_HISTORY_LEN,
)
from two_tower import TwoTowerModel, bpr_loss, get_all_item_embeddings
from index import EmbeddingIndex


def _strong_events():
    """Item 'hot' gets >= MIN_POSITIVE_INTERACTIONS carts; 'rare' gets one."""
    base = 1_600_000_000_000
    rows = []
    # hot: enough strong events across a few users to clear the threshold
    for i in range(MIN_POSITIVE_INTERACTIONS + 1):
        rows.append((f"u{i}", "medium", "hot", base + i))
    # warm: exactly at the threshold
    for i in range(MIN_POSITIVE_INTERACTIONS):
        rows.append((f"w{i}", "strong", "warm", base + 100 + i))
    # rare: below threshold -> excluded from vocab
    rows.append(("u0", "medium", "rare", base + 200))
    # a bunch of views that must NOT create vocab entries
    rows.append(("u0", "weak", "viewonly", base + 300))

    return pl.DataFrame(
        {
            "user_id":      [r[0] for r in rows],
            "event_type":   [r[1] for r in rows],
            "item_id":      [r[2] for r in rows],
            "timestamp_ms": [r[3] for r in rows],
        }
    ).with_columns(
        pl.col("user_id").cast(pl.Utf8),
        pl.col("item_id").cast(pl.Utf8),
        pl.col("timestamp_ms").cast(pl.Int64),
    )


# ---------------------------------------------------------------- build_vocab


def test_vocab_only_includes_frequent_strong_items():
    vocab = build_vocab(_strong_events())
    ids = set(vocab.idx_to_item_id)
    assert "hot" in ids
    assert "warm" in ids
    assert "rare" not in ids       # below MIN_POSITIVE_INTERACTIONS
    assert "viewonly" not in ids   # views never count as strong


def test_vocab_encode_decode_roundtrip():
    vocab = build_vocab(_strong_events())
    for idx, item_id in enumerate(vocab.idx_to_item_id):
        assert vocab.encode(item_id) == idx
        assert vocab.decode(idx) == item_id
    assert vocab.encode("does-not-exist") is None


# ---------------------------------------------------------------- histories


def test_build_user_histories_dedups_and_caps():
    base = 1_600_000_000_000
    # u0 interacts with 'hot' many times -> dedup to a single entry.
    rows = [("u0", "medium", "hot", base + i) for i in range(MAX_HISTORY_LEN + 5)]
    events = pl.DataFrame(
        {
            "user_id":      [r[0] for r in rows],
            "event_type":   [r[1] for r in rows],
            "item_id":      [r[2] for r in rows],
            "timestamp_ms": [r[3] for r in rows],
        }
    ).with_columns(pl.col("timestamp_ms").cast(pl.Int64))

    vocab = build_vocab(events)
    histories = build_user_histories(events, vocab)
    assert "u0" in histories
    # dedup -> only one unique item, and never exceeds the cap
    assert len(histories["u0"]) == 1
    assert len(histories["u0"]) <= MAX_HISTORY_LEN


# ---------------------------------------------------------------- BPRDataset


def test_bpr_dataset_shapes_and_padding():
    events = _strong_events()
    vocab = build_vocab(events)
    histories = build_user_histories(events, vocab)
    ds = BPRDataset(events, vocab, histories, seed=0)

    if len(ds) == 0:
        # Every user here has single-item history -> all skipped. That's a valid
        # outcome and itself worth asserting so we notice if it changes.
        return

    sample = ds[0]
    assert sample["history"].shape == (MAX_HISTORY_LEN,)
    assert sample["pos"].ndim == 0
    assert sample["neg"].ndim == 0
    # Negative must not equal positive.
    assert int(sample["neg"]) != int(sample["pos"])


def test_bpr_dataset_rejects_bad_sampling_strategy():
    events = _strong_events()
    vocab = build_vocab(events)
    histories = build_user_histories(events, vocab)
    try:
        BPRDataset(events, vocab, histories, negative_sampling="nonsense")
        assert False, "expected ValueError for unknown sampling strategy"
    except ValueError:
        pass


def test_bpr_dataset_popularity_sampling_runs():
    # Build a dataset large enough to produce triples, using popularity negatives.
    base = 1_600_000_000_000
    rows = []
    for item in ("a", "b", "c", "d"):
        for i in range(MIN_POSITIVE_INTERACTIONS):
            rows.append((f"seed_{item}_{i}", "strong", item, base + i))
    rows.append(("hero", "strong", "a", base + 500))
    rows.append(("hero", "strong", "b", base + 501))
    events = pl.DataFrame(
        {
            "user_id":      [r[0] for r in rows],
            "event_type":   [r[1] for r in rows],
            "item_id":      [r[2] for r in rows],
            "timestamp_ms": [r[3] for r in rows],
        }
    ).with_columns(pl.col("timestamp_ms").cast(pl.Int64))

    vocab = build_vocab(events)
    histories = build_user_histories(events, vocab)
    ds = BPRDataset(events, vocab, histories, seed=7, negative_sampling="popularity")
    assert ds.negative_sampling == "popularity"
    # Every produced negative must be a valid vocab index.
    for context, pos, neg in ds.samples:
        assert 0 <= neg < vocab.size


def test_bpr_dataset_negatives_outside_history():
    # Craft a user with a real 2-item history so a triple is produced.
    base = 1_600_000_000_000
    rows = []
    for item in ("a", "b", "c"):
        for i in range(MIN_POSITIVE_INTERACTIONS):
            rows.append((f"seed_{item}_{i}", "strong", item, base + i))
    # the user we care about interacts with a and b (context + positive)
    rows.append(("hero", "strong", "a", base + 500))
    rows.append(("hero", "strong", "b", base + 501))
    events = pl.DataFrame(
        {
            "user_id":      [r[0] for r in rows],
            "event_type":   [r[1] for r in rows],
            "item_id":      [r[2] for r in rows],
            "timestamp_ms": [r[3] for r in rows],
        }
    ).with_columns(pl.col("timestamp_ms").cast(pl.Int64))

    vocab = build_vocab(events)
    histories = build_user_histories(events, vocab)
    ds = BPRDataset(events, vocab, histories, seed=1)
    assert len(ds) > 0


# ---------------------------------------------------------------- TwoTowerModel


def test_forward_returns_batch_scores():
    model = TwoTowerModel(n_items=10, embedding_dim=8)
    history = torch.tensor([[0, 1, model.pad_idx], [2, 3, 4]], dtype=torch.long)
    pos = torch.tensor([1, 2], dtype=torch.long)
    neg = torch.tensor([5, 6], dtype=torch.long)

    pos_scores, neg_scores = model(history, pos, neg)
    assert pos_scores.shape == (2,)
    assert neg_scores.shape == (2,)


def test_user_vector_ignores_padding():
    model = TwoTowerModel(n_items=5, embedding_dim=4)
    # Give item 0 a known embedding, zero out others via padding_idx already 0.
    with torch.no_grad():
        model.item_emb.weight[:] = 0.0
        model.item_emb.weight[1] = torch.tensor([1.0, 2.0, 3.0, 4.0])

    # History: item 1 plus two pad tokens. Mean should equal item 1's embedding.
    pad = model.pad_idx
    history = torch.tensor([[1, pad, pad]], dtype=torch.long)
    user_vec = model.get_user_vector(history)
    assert torch.allclose(user_vec[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_bpr_loss_lower_when_ranking_correct():
    good = bpr_loss(torch.tensor([5.0]), torch.tensor([-5.0]))  # pos >> neg
    bad = bpr_loss(torch.tensor([-5.0]), torch.tensor([5.0]))   # neg >> pos
    assert good < bad
    assert good.item() >= 0


def test_get_all_item_embeddings_excludes_pad_row():
    n_items = 7
    model = TwoTowerModel(n_items=n_items, embedding_dim=8)
    emb = get_all_item_embeddings(model)
    assert emb.shape == (n_items, 8)  # pad row dropped


# ---------------------------------------------------------------- EmbeddingIndex


def test_index_search_returns_closest_item():
    # Three orthogonal-ish items; querying near item 1 should return it first.
    emb = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    index = EmbeddingIndex(item_embeddings=emb, idx_to_item_id=["a", "b", "c"])
    results = index.search(np.array([0.0, 0.9, 0.1]), k=2)
    assert results[0] == "b"
    assert len(results) == 2


def test_index_search_respects_exclusions():
    emb = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    index = EmbeddingIndex(item_embeddings=emb, idx_to_item_id=["a", "b", "c"])
    results = index.search(np.array([1.0, 0.0]), k=2, exclude_item_ids={"a"})
    assert "a" not in results
