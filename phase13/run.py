"""
Phase 13 -- Entry Point: does the two-tower SCORE make stage 2 earn its place?
==============================================================================
Phase 12 showed a naive two-stage pipeline regressed vs two-tower alone. The
hypothesis (straight from Phase 12's "next steps"): the LR lost because it never
saw the retrieval signal. Give it the two-tower similarity score as a feature and
the rerank should stop fighting the retriever.

Three rankers, all graded by phase0/evaluate.py's recall_at_k:

    1. Two-tower alone                     (Phase 1 -- the bar to beat)
    2. Two-tower -> LR (popularity feats)  (Phase 12 -- regressed)
    3. Two-tower -> LR + tt_score          (Phase 13 -- the fix under test)

Usage:
    cd phase13 && python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase12"))
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events, load_item_properties
from evaluate import temporal_split, recall_at_k
from heuristic_ranker import HeuristicRanker
from results_io import save_results

from dataset import build_vocab, build_user_histories, BPRDataset
from two_tower import get_all_item_embeddings
from train import train
from index import EmbeddingIndex
from ranker import TwoTowerRanker

from feature_store import PointInTimeFeatureStore, FEATURE_COLUMNS
from lr_ranker import LRRanker
from training import build_labelled_features

from pipeline import TwoStageRanker
from scorer import TwoTowerScorer, TT_SCORE_COL

PHASE_DIR = Path(__file__).parent

K = 20
MAX_USERS = 3000
CANDIDATE_POOL = 200
MAX_POSITIVES = 100_000
SEED = 42


def _summarize(res: dict) -> dict:
    return {"recall": res["recall_at_k"], "ndcg": res["ndcg_at_k"], "coverage": res["catalog_coverage"]}


def main():
    print("\n=== Phase 13: Two-Tower Score as a Ranking Feature ===\n")
    rng = np.random.default_rng(SEED)

    print("Step 1/6: Load data + temporal split...")
    events = load_events()
    item_props = load_item_properties()
    train_events, test_events, cutoff_ms = temporal_split(events, train_fraction=0.8)
    catalog_size = train_events["item_id"].n_unique()

    print("\nStep 2/6: Build two-tower retrieval...")
    vocab = build_vocab(train_events)
    histories = build_user_histories(train_events, vocab)
    dataset = BPRDataset(train_events, vocab, histories)
    model = train(dataset, vocab, run_name="two-tower-score-feature")
    item_emb = get_all_item_embeddings(model)
    index = EmbeddingIndex(item_embeddings=item_emb, idx_to_item_id=vocab.idx_to_item_id)
    fallback = HeuristicRanker()
    fallback.fit(events=train_events, item_properties=item_props, cutoff_timestamp_ms=cutoff_ms)
    two_tower = TwoTowerRanker(
        model=model, vocab=vocab, index=index, fallback=fallback, item_embeddings=item_emb
    )

    print("\nStep 3/6: Fit feature store + build labelled training set...")
    store = PointInTimeFeatureStore().fit(
        train_events, cutoff_timestamp_ms=cutoff_ms, item_properties=item_props
    )
    training_df = build_labelled_features(
        train_events, store, rng, max_positives=MAX_POSITIVES, seed=SEED
    )

    print("\nStep 4/6: Add two-tower score to the training features...")
    scorer = TwoTowerScorer(item_emb, vocab)
    user_vecs = scorer.precompute_user_vectors(histories)
    tt_scores = scorer.score_training_rows(training_df, user_vecs)
    training_df = training_df.with_columns(pl.Series(TT_SCORE_COL, tt_scores))
    print(f"  tt_score: mean={tt_scores.mean():.3f} std={tt_scores.std():.3f} "
          f"(non-zero for {np.mean(tt_scores != 0):.0%} of rows)")

    print("\nStep 5/6: Fit both rankers (popularity-only vs +tt_score)...")
    lr_pop = LRRanker().fit(training_df)   # Phase 12 config (popularity features)
    lr_tt = LRRanker(
        feature_columns=FEATURE_COLUMNS + [TT_SCORE_COL],
        log_columns=FEATURE_COLUMNS,       # log the counts, pass tt_score raw
    ).fit(training_df)

    two_stage_pop = TwoStageRanker(two_tower, lr_pop, store, candidate_pool=CANDIDATE_POOL)
    two_stage_tt = TwoStageRanker(
        two_tower, lr_tt, store, candidate_pool=CANDIDATE_POOL,
        feature_augmenter=scorer.augment,
    )
    rankers = [
        ("two_tower", "Two-tower alone", two_tower),
        ("two_stage_pop", "TT -> LR (pop feats)", two_stage_pop),
        ("two_stage_tt", "TT -> LR + tt_score", two_stage_tt),
    ]

    print("\nStep 6/6: Evaluate all three on the same users/split/metric...")
    results = {}
    for key, _, ranker in rankers:
        results[key] = _summarize(
            recall_at_k(
                ranker=ranker, train_events=train_events, test_events=test_events,
                k=K, max_users=MAX_USERS, catalog_size=catalog_size,
            )
        )

    print("\n" + "=" * 66)
    print("  STAGE-2 EARNS ITS PLACE? -- head to head (same users)")
    print("=" * 66)
    print(f"  {'Ranker':<24}{'Recall@'+str(K):>10}{'NDCG@'+str(K):>10}{'Coverage':>12}")
    print("-" * 66)
    for key, label, _ in rankers:
        r = results[key]
        print(f"  {label:<24}{r['recall']:>10.4f}{r['ndcg']:>10.4f}{r['coverage']:>12.4f}")
    print("=" * 66)

    base = results["two_tower"]["recall"]
    pop = results["two_stage_pop"]["recall"]
    tt = results["two_stage_tt"]["recall"]
    print(f"\n  vs two-tower alone:  pop-feats {(pop/base-1)*100:+.1f}%   "
          f"+tt_score {(tt/base-1)*100:+.1f}%")
    if tt >= base and tt >= pop:
        print("  Fix confirmed: giving stage 2 the retrieval score lets it MATCH/BEAT the")
        print("  retriever -- it now preserves good order and only reorders on real signal.")
    elif tt > pop:
        print("  Partial win: tt_score recovers most of the loss from Phase 12's naive")
        print("  two-stage, but a linear rerank still can't add net lift over strong")
        print("  retrieval on this data. Honest -- earn further complexity only if it pays.")
    else:
        print("  Honest null: even with the retrieval score, the LR rerank doesn't beat")
        print("  two-tower alone here. Strong retrieval is a hard bar for a linear ranker.")

    save_results(PHASE_DIR, results)

    print("\nNext steps:")
    print("  - Add features orthogonal to retrieval (recency, session, price/brand).")
    print("  - Upgrade LR -> gradient-boosted / listwise ranker once features carry signal.")
    print("  - Wire the winning config into the Phase 3 service candidate path.\n")
    return results


if __name__ == "__main__":
    main()
