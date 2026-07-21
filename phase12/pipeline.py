"""
Phase 12 -- Two-Stage Retrieval + Ranking (the integration)
============================================================
Every earlier phase built a piece; none of them were wired together. Phase 1 built
two-tower RETRIEVAL, Phase 2 built an LR RANKER on a feature store, Phase 3 built
the SERVICE -- but the service still generates candidates from *popularity*, and the
LR ranker was only ever tested reordering a *popularity* pool. This phase closes the
loop Phase 1 and Phase 2 each promised in their "next steps":

    Stage 1 (retrieval): two-tower ANN turns the catalog into ~N candidates.
    Stage 2 (ranking):   the LR ranker reorders those candidates into the top-k.

`TwoStageRanker` is deliberately a COMPOSITION, not a new model: it takes any
stage-1 candidate generator that exposes `.recommend()` (two-tower OR popularity),
any stage-2 reranker with `.rank()`, and the feature store. Because it exposes the
same `.recommend()` interface as every other ranker, phase0/evaluate.py's
`recall_at_k` grades it apples-to-apples against the single-stage baselines -- which
is the whole point: we can finally MEASURE whether two stages beat one.

Pure composition, no training logic here (Rule 5 / SRP). `run.py` assembles the
concrete parts.
"""

from __future__ import annotations

import polars as pl


class TwoStageRanker:
    """
    Retrieval -> ranking pipeline behind the standard `.recommend()` interface.

    candidate_generator : anything with `.recommend(user_id, user_events, n)` and a
                          `.catalog_size` property (TwoTowerRanker, HeuristicRanker...).
    reranker            : anything with `.rank(features_df, item_col, n)` (LRRanker).
    feature_store       : PointInTimeFeatureStore, queried at serve time for the
                          (user, candidate) features the reranker needs.
    candidate_pool      : how many candidates stage 1 hands to stage 2. Bigger =
                          more chances to recover a relevant item, slower to rank.
    """

    def __init__(
        self,
        candidate_generator,
        reranker,
        feature_store,
        candidate_pool: int = 200,
    ):
        self.candidate_generator = candidate_generator
        self.reranker = reranker
        self.feature_store = feature_store
        self.candidate_pool = candidate_pool

    @property
    def catalog_size(self) -> int:
        """Delegate coverage denominator to stage 1 (it owns the recall universe)."""
        return self.candidate_generator.catalog_size

    def recommend(self, user_id: str, user_events: pl.DataFrame, n: int = 20) -> list[str]:
        # Stage 1: retrieve a candidate pool (already excludes seen items, handles
        # cold-start via its own fallback -- we don't re-implement either here).
        candidates = self.candidate_generator.recommend(
            user_id=user_id, user_events=user_events, n=self.candidate_pool
        )
        if not candidates:
            return []

        # Stage 2: featurize (user x candidate) with online features, then rerank.
        cand_df = pl.DataFrame({"item_id": candidates}).with_columns(
            pl.lit(user_id).alias("user_id")
        )
        features = self.feature_store.get_online_features_batch(cand_df)
        ranked = self.reranker.rank(features, item_col="item_id", n=n)

        # Safety net: if the reranker somehow drops below n (e.g. dedupe), backfill
        # from the stage-1 order so we never return fewer than we could.
        if len(ranked) < n:
            seen = set(ranked)
            ranked += [c for c in candidates if c not in seen][: n - len(ranked)]
        return ranked[:n]
