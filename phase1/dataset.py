"""
Phase 1 — Training Dataset
============================
Builds BPR (Bayesian Personalized Ranking) training triples from events.

Key design decisions (all flow from the design doc):

1. Positives = cart + purchase ONLY.
   Views are 96.7% of events and carry weak signal — using them as positives
   would teach the model to replicate a popularity ranker, not beat one.

2. User = mean-pool of history item embeddings (no user ID table).
   This makes the model generalize to users it has never seen (cold-start).
   Absent from training? No embedding → mean of empty set → fallback to Phase 0.
   This is how YouTube DNN and most production two-tower models work.

3. Items with < MIN_ITEM_INTERACTIONS are excluded from the vocab.
   Embeddings for items with 1-2 interactions can't be meaningfully learned.
   These tail items stay in the Phase 0 popularity fallback.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from signals import POSITIVE_SIGNALS

MIN_STRONG_INTERACTIONS = 3   # item must appear in >=N positive events to enter vocab
MAX_HISTORY_LEN         = 50  # cap user history to most-recent N items (memory + speed)
POSITIVE_SIGNAL_TYPES   = set(POSITIVE_SIGNALS)  # engagement-or-stronger = a positive

# Exponent for popularity-based negative sampling (word2vec's 0.75 trick).
# Sampling negatives ∝ count**0.75 gives HARDER negatives than uniform:
# popular items the user did NOT interact with are strong "why-not-this" signals,
# which sharpens the ranking far more than random tail items.
NEG_SAMPLING_EXPONENT = 0.75


@dataclass
class ItemVocab:
    """
    Maps item_id (string) <-> contiguous integer index for the embedding table.
    Items not in vocab were seen too rarely to train a useful embedding.
    """
    item_id_to_idx: dict[str, int]
    idx_to_item_id: list[str]

    @property
    def size(self) -> int:
        return len(self.idx_to_item_id)

    def encode(self, item_id: str) -> int | None:
        return self.item_id_to_idx.get(item_id)

    def decode(self, idx: int) -> str:
        return self.idx_to_item_id[idx]


def build_vocab(train_events: pl.DataFrame) -> ItemVocab:
    """
    Build item vocabulary from STRONG events (cart + purchase) only.

    Why strong events only?
    If we include all events, items with many views but zero purchases enter the
    vocab with no real training signal from positives. Their embeddings stay
    near-random after training. Searching 78K near-random embeddings is worse
    than a popularity ranker — this is exactly what happened in our first run.

    By restricting to items with ≥ MIN_STRONG_INTERACTIONS strong events, every
    item in the vocab has meaningful gradient signal and the search index is
    high-quality. Items outside the vocab fall back to Phase 0.
    """
    strong_events = train_events.filter(
        pl.col("event_type").is_in(list(POSITIVE_SIGNAL_TYPES))
    )

    item_counts = (
        strong_events
        .group_by("item_id")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") >= MIN_STRONG_INTERACTIONS)
        .sort("count", descending=True)
    )

    items = item_counts["item_id"].to_list()
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(items)}

    print(
        f"[dataset] Vocab: {len(items):,} items from strong events "
        f"(min {MIN_STRONG_INTERACTIONS} cart/purchase events each; "
        f"total unique items in training: {train_events['item_id'].n_unique():,})"
    )
    return ItemVocab(item_id_to_idx=item_id_to_idx, idx_to_item_id=items)


def build_user_histories(
    train_events: pl.DataFrame,
    vocab: ItemVocab,
) -> dict[str, list[int]]:
    """
    For each user: list of item indices (in vocab) from their training history,
    sorted by timestamp (most recent last), capped at MAX_HISTORY_LEN.

    These are the user's "context" — mean-pooled at inference time to form the
    user vector. Excludes items not in vocab (too rare to embed meaningfully).
    """
    histories: dict[str, list[int]] = {}

    grouped = (
        train_events
        .sort("timestamp_ms")
        .filter(pl.col("item_id").is_in(list(vocab.item_id_to_idx.keys())))
        .group_by("user_id")
        .agg(pl.col("item_id").alias("items"))
    )

    for row in grouped.iter_rows(named=True):
        items = row["items"]
        idxs  = [vocab.encode(i) for i in items if vocab.encode(i) is not None]
        # Deduplicate while preserving order (user buys same item twice: count once)
        seen  = set()
        idxs  = [x for x in idxs if not (x in seen or seen.add(x))]
        histories[row["user_id"]] = idxs[-MAX_HISTORY_LEN:]

    print(f"[dataset] Built histories for {len(histories):,} users")
    return histories


class BPRDataset(Dataset):
    """
    BPR training triples: (user_history, positive_item, negative_item)

    For each user with at least one cart/purchase event:
      - positive = each cart/purchase item (in vocab)
      - user_context = ALL other in-vocab items in user's history (excluding the positive)
      - negative = uniformly sampled item NOT in user's full history

    History is padded to MAX_HISTORY_LEN with PAD_IDX = vocab.size.
    The model zeroes out pad positions before mean-pooling.
    """

    def __init__(
        self,
        train_events: pl.DataFrame,
        vocab: ItemVocab,
        user_histories: dict[str, list[int]],
        seed: int = 42,
        negative_sampling: str = "popularity",
    ):
        self.vocab = vocab
        self.pad_idx = vocab.size  # out-of-vocab sentinel — model embedding table has size+1 rows

        if negative_sampling not in {"uniform", "popularity"}:
            raise ValueError(f"negative_sampling must be 'uniform' or 'popularity', got {negative_sampling!r}")
        self.negative_sampling = negative_sampling

        random.seed(seed)
        self._rng = np.random.default_rng(seed)

        # Build positives: (user_id, positive_item_idx)
        strong_events = train_events.filter(
            pl.col("event_type").is_in(list(POSITIVE_SIGNAL_TYPES))
            & pl.col("item_id").is_in(list(vocab.item_id_to_idx.keys()))
        )

        self.samples: list[tuple[list[int], int, int]] = []

        all_item_indices = list(range(vocab.size))

        # Precompute the popularity distribution over vocab indices (count**0.75).
        self._neg_probs = self._build_popularity_probs(strong_events, vocab)
        # A refillable buffer of pre-sampled negatives (vectorized draws are far
        # faster than one np.random.choice call per triple).
        self._neg_buffer: list[int] = []

        users_skipped = 0
        for row in strong_events.iter_rows(named=True):
            user_id     = row["user_id"]
            pos_idx     = vocab.encode(row["item_id"])
            if pos_idx is None:
                continue

            full_history = set(user_histories.get(user_id, []))
            # Context = history minus the positive (avoid trivial self-prediction)
            context = [i for i in user_histories.get(user_id, []) if i != pos_idx]

            if not context:
                # User only ever interacted with this one item — skip (no context to learn from)
                users_skipped += 1
                continue

            # Negative: item NOT in user's history (uniform or popularity-weighted)
            neg_idx = self._sample_negative(full_history, all_item_indices)

            self.samples.append((context, pos_idx, neg_idx))

        print(
            f"[dataset] {len(self.samples):,} training triples "
            f"(neg sampling: {self.negative_sampling}; "
            f"{users_skipped:,} samples skipped — single-item history)"
        )

    def _build_popularity_probs(
        self,
        strong_events: pl.DataFrame,
        vocab: ItemVocab,
    ) -> np.ndarray | None:
        """Probability of drawing each vocab index as a negative (count**0.75)."""
        if self.negative_sampling != "popularity":
            return None
        counts = np.zeros(vocab.size, dtype=np.float64)
        agg = (
            strong_events
            .group_by("item_id")
            .agg(pl.len().alias("count"))
        )
        for r in agg.iter_rows(named=True):
            idx = vocab.encode(r["item_id"])
            if idx is not None:
                counts[idx] = r["count"]
        weights = np.power(counts, NEG_SAMPLING_EXPONENT)
        total = weights.sum()
        if total <= 0:
            return None  # degenerate -> fall back to uniform
        return weights / total

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        context, pos_idx, neg_idx = self.samples[idx]

        # Pad history to MAX_HISTORY_LEN
        padded = context[-MAX_HISTORY_LEN:]  # most-recent N items
        pad_len = MAX_HISTORY_LEN - len(padded)
        padded  = padded + [self.pad_idx] * pad_len

        return {
            "history": torch.tensor(padded,   dtype=torch.long),
            "pos":     torch.tensor(pos_idx,  dtype=torch.long),
            "neg":     torch.tensor(neg_idx,  dtype=torch.long),
        }

    def _sample_negative(self, user_item_set: set[int], all_items: list[int]) -> int:
        """
        Draw a negative item not in the user's history.

        - uniform: any vocab item with equal probability.
        - popularity: items proportional to count**0.75 (harder negatives).
          Drawn from a refillable buffer for speed.
        """
        if self.negative_sampling == "uniform" or self._neg_probs is None:
            while True:
                neg = random.choice(all_items)
                if neg not in user_item_set:
                    return neg

        # popularity-weighted, buffered
        while True:
            if not self._neg_buffer:
                draws = self._rng.choice(
                    len(self._neg_probs), size=8192, p=self._neg_probs
                )
                self._neg_buffer = draws.tolist()
            neg = self._neg_buffer.pop()
            if neg not in user_item_set:
                return neg
