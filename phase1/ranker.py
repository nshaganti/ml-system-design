"""
Phase 1 — Two-Tower Ranker
=============================
Wraps the trained model + index into the SAME interface as phase0/HeuristicRanker.

This is the key integration point: because the interface is identical, we can
pass TwoTowerRanker directly into phase0/evaluate.py's recall_at_k() — giving
us a fair apples-to-apples comparison on the same users, same test set, same seed.

Cold-start strategy (Rule 28: a model that can't score new items degrades):
  - User has in-vocab history → mean-pool item embeddings → search index
  - User has no in-vocab history → fall back to Phase 0 HeuristicRanker

This two-layer fallback is standard production practice: the ML model handles
warm users (where it has learned signal), heuristics handle cold-start.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
from heuristic_ranker import HeuristicRanker
from signals import seen_positive_item_ids

sys.path.insert(0, str(Path(__file__).parent))
from two_tower import TwoTowerModel
from index import EmbeddingIndex
from dataset import ItemVocab

import polars as pl


class TwoTowerRanker:
    """
    Production-style ranker: ML model for warm users, heuristic fallback for cold-start.

    Exposes the same .recommend() interface as HeuristicRanker so phase0/evaluate.py
    works without modification.
    """

    def __init__(
        self,
        model: TwoTowerModel,
        vocab: ItemVocab,
        index: EmbeddingIndex,
        fallback: HeuristicRanker,
        item_embeddings: torch.Tensor | None = None,
    ):
        self.model    = model
        self.vocab    = vocab
        self.index    = index
        self.fallback = fallback

        self.model.eval()
        # Reuse the embedding matrix already extracted for the index if provided
        # (Rule: don't recompute what you already have). Otherwise extract it once.
        if item_embeddings is not None:
            self._item_emb = item_embeddings.detach().cpu().numpy()
        else:
            with torch.no_grad():
                all_idx        = torch.arange(vocab.size)
                self._item_emb = model.item_emb(all_idx).numpy()  # (n_items, dim)

    @property
    def catalog_size(self) -> int:
        """Number of items in the embedding vocab. Used by evaluate.py for coverage."""
        return self.vocab.size

    def recommend(
        self,
        user_id: str,
        user_events: pl.DataFrame,
        n: int = 20,
    ) -> list[str]:
        """
        Return top-n item IDs for a user.

        user_events: user's historical events BEFORE the evaluation cutoff.
                     Same signature as HeuristicRanker.recommend().
        """
        # Translate user's history to in-vocab item indices
        if len(user_events) > 0:
            history_ids = user_events["item_id"].to_list()
            history_idx = [
                self.vocab.encode(item_id)
                for item_id in history_ids
                if self.vocab.encode(item_id) is not None
            ]
        else:
            history_idx = []

        # Cold-start: no in-vocab history → Phase 0 fallback
        if not history_idx:
            return self.fallback.recommend(
                user_id=user_id,
                user_events=user_events,
                n=n,
            )

        # Warm user: compute user vector from history
        user_vec = self._compute_user_vector(history_idx)

        # Items already positively engaged with (don't re-recommend). SHARED
        # policy with the heuristic so recall_at_k is apples-to-apples -- weak
        # exposures are intentionally NOT excluded (see signals module).
        already_seen = seen_positive_item_ids(user_events)

        # Stage 1: ANN search → top-500 candidates (same as production architecture)
        candidates = self.index.search(
            query_vec=user_vec,
            k=500,
            exclude_item_ids=already_seen,
        )

        # In Phase 1, we skip Stage 2 (the ranker) — candidates ARE the output.
        # Phase 2 adds a logistic regression ranker on top of these candidates.
        return candidates[:n]

    def _compute_user_vector(self, history_idx: list[int]) -> np.ndarray:
        """
        Mean-pool item embeddings for the user's history.
        This is the exact same computation as the model's get_user_vector(),
        just in numpy (no gradient needed at inference).

        Rule 32: same logic runs in training and serving.
        """
        history_emb = self._item_emb[history_idx]      # (seq_len, dim)
        return history_emb.mean(axis=0)                 # (dim,)
