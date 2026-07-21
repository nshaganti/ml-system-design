"""
Phase 13 -- Two-Tower Score as a Ranking Feature ("earn the rerank")
====================================================================
Phase 12's honest result: a two-tower -> LR two-stage pipeline LOST to two-tower
retrieval alone, because the LR's only features were popularity-flavored -- so
reranking undid the retriever's personalization. The fix Phase 12 pointed to:
**give stage 2 the retrieval signal itself.**

This module turns the trained two-tower into a *feature*: the similarity score
`user_vec . item_vec` for each (user, candidate) pair. Add that column to the LR's
inputs and the ranker can PRESERVE good retrieval order instead of fighting it,
while still using popularity features to break ties. Same computation at training
and serving time (Rule 32) -- that's the entire job of this class.

The score is NOT a heavy-tailed count, so it is passed to the LR raw (no log1p) --
which is exactly why Phase 13 also taught `lr_ranker.to_matrix` to take an explicit
`log_columns` list.
"""

from __future__ import annotations

import numpy as np
import polars as pl

TT_SCORE_COL = "tt_score"


class TwoTowerScorer:
    """
    Computes two-tower similarity scores for (user, item) pairs.

    item_embeddings : (n_items, dim) numpy matrix (get_all_item_embeddings()).
    vocab           : ItemVocab (encode item_id -> idx; None if OOV).

    A user's vector is the mean of their in-vocab history item embeddings -- the
    exact same pooling the retriever uses, so training and serving scores match.
    """

    def __init__(self, item_embeddings, vocab):
        self._item_emb = np.asarray(item_embeddings)
        self.vocab = vocab
        self.dim = self._item_emb.shape[1]

    # --------------------------------------------------------- user vectors

    def user_vector_from_idx(self, history_idx: list[int]) -> np.ndarray | None:
        if not history_idx:
            return None
        return self._item_emb[history_idx].mean(axis=0)

    def user_vector_from_events(self, user_events: pl.DataFrame) -> np.ndarray | None:
        if user_events is None or len(user_events) == 0:
            return None
        idx = [
            self.vocab.encode(i)
            for i in user_events["item_id"].to_list()
            if self.vocab.encode(i) is not None
        ]
        return self.user_vector_from_idx(idx)

    def precompute_user_vectors(self, histories: dict[str, list[int]]) -> dict[str, np.ndarray]:
        """Batch user vectors from the Phase 1 histories dict (for training)."""
        return {
            uid: self.user_vector_from_idx(idxs)
            for uid, idxs in histories.items()
            if idxs
        }

    # --------------------------------------------------------- scoring

    def score(self, user_vec: np.ndarray | None, item_id: str) -> float:
        """Similarity for one pair; 0.0 for cold user or OOV item (neutral)."""
        if user_vec is None:
            return 0.0
        idx = self.vocab.encode(item_id)
        if idx is None:
            return 0.0
        return float(np.dot(user_vec, self._item_emb[idx]))

    def score_training_rows(
        self, df: pl.DataFrame, user_vecs: dict[str, np.ndarray]
    ) -> np.ndarray:
        """tt_score for each row of a labelled training frame (user_id, item_id)."""
        users = df["user_id"].to_list()
        items = df["item_id"].to_list()
        return np.array(
            [self.score(user_vecs.get(u), it) for u, it in zip(users, items)],
            dtype=np.float64,
        )

    # --------------------------------------------------------- serving hook

    def augment(self, user_id: str, user_events: pl.DataFrame, features_df: pl.DataFrame) -> pl.DataFrame:
        """
        TwoStageRanker feature_augmenter: add a tt_score column to the candidate
        feature frame at serve time. Signature matches the pipeline's hook.
        """
        user_vec = self.user_vector_from_events(user_events)
        scores = [self.score(user_vec, it) for it in features_df["item_id"].to_list()]
        return features_df.with_columns(pl.Series(TT_SCORE_COL, scores))
