"""
Phase 12 -- Entry Point: Two-Stage Retrieval + Ranking
=======================================================
Assembles the pieces built across Phases 0-2 into one pipeline and answers the
question the architecture diagrams always assume but rarely test:

    Does a two-stage (two-tower retrieval -> LR rerank) system actually beat the
    single-stage baselines on the SAME users, split, and metric?

Three rankers, all graded by phase0/evaluate.py's recall_at_k (apples-to-apples):

    1. Two-tower alone            (Phase 1: retrieval IS the output)
    2. Popularity -> LR rerank    (Phase 2: LR over a popularity pool)
    3. Two-tower -> LR rerank     (Phase 12: the full two-stage architecture)

Usage:
    cd phase12 && python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase2"))
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events, load_item_properties
from evaluate import temporal_split, recall_at_k
from heuristic_ranker import HeuristicRanker
from results_io import save_results

# Phase 1 (retrieval)
from dataset import build_vocab, build_user_histories, BPRDataset
from two_tower import get_all_item_embeddings
from train import train
from index import EmbeddingIndex
from ranker import TwoTowerRanker

# Phase 2 (ranking)
from feature_store import PointInTimeFeatureStore
from lr_ranker import LRRanker
from training import build_labelled_features

# Phase 12 (integration)
from pipeline import TwoStageRanker

PHASE_DIR = Path(__file__).parent

K = 20
MAX_USERS = 3000        # same cap for all three rankers -> fair within-group
CANDIDATE_POOL = 200    # stage-1 -> stage-2 handoff size
MAX_POSITIVES = 100_000
SEED = 42


def _summarize(res: dict) -> dict:
    return {
        "recall": res["recall_at_k"],
        "ndcg": res["ndcg_at_k"],
        "coverage": res["catalog_coverage"],
    }


def main():
    print("\n=== Phase 12: Two-Stage Retrieval + Ranking ===\n")
    rng = np.random.default_rng(SEED)

    print("Step 1/5: Load data + temporal split...")
    events = load_events()
    item_props = load_item_properties()
    train_events, test_events, cutoff_ms = temporal_split(events, train_fraction=0.8)
    catalog_size = train_events["item_id"].n_unique()

    print("\nStep 2/5: Build two-tower retrieval (Phase 1)...")
    vocab = build_vocab(train_events)
    histories = build_user_histories(train_events, vocab)
    dataset = BPRDataset(train_events, vocab, histories)
    model = train(dataset, vocab, run_name="two-stage-retrieval")
    item_emb = get_all_item_embeddings(model)
    index = EmbeddingIndex(item_embeddings=item_emb, idx_to_item_id=vocab.idx_to_item_id)

    fallback = HeuristicRanker()
    fallback.fit(events=train_events, item_properties=item_props, cutoff_timestamp_ms=cutoff_ms)
    two_tower = TwoTowerRanker(
        model=model, vocab=vocab, index=index, fallback=fallback, item_embeddings=item_emb
    )

    print("\nStep 3/5: Fit feature store + LR ranker (Phase 2)...")
    store = PointInTimeFeatureStore().fit(
        train_events, cutoff_timestamp_ms=cutoff_ms, item_properties=item_props
    )
    training_df = build_labelled_features(
        train_events, store, rng, max_positives=MAX_POSITIVES, seed=SEED
    )
    lr = LRRanker().fit(training_df)

    print("\nStep 4/5: Assemble the three rankers...")
    pop_two_stage = TwoStageRanker(
        candidate_generator=fallback, reranker=lr, feature_store=store,
        candidate_pool=CANDIDATE_POOL,
    )
    tt_two_stage = TwoStageRanker(
        candidate_generator=two_tower, reranker=lr, feature_store=store,
        candidate_pool=CANDIDATE_POOL,
    )
    rankers = [
        ("two_tower", "Two-tower alone (P1)", two_tower),
        ("pop_lr", "Popularity -> LR (P2)", pop_two_stage),
        ("two_stage", "Two-tower -> LR (P12)", tt_two_stage),
    ]

    print("\nStep 5/5: Evaluate all three on the same users/split/metric...")
    results = {}
    for key, _, ranker in rankers:
        res = recall_at_k(
            ranker=ranker, train_events=train_events, test_events=test_events,
            k=K, max_users=MAX_USERS, catalog_size=catalog_size,
        )
        results[key] = _summarize(res)

    print("\n" + "=" * 64)
    print("  TWO-STAGE INTEGRATION -- head to head (same users)")
    print("=" * 64)
    print(f"  {'Ranker':<26}{'Recall@'+str(K):>10}{'NDCG@'+str(K):>10}{'Coverage':>12}")
    print("-" * 64)
    for key, label, _ in rankers:
        r = results[key]
        print(f"  {label:<26}{r['recall']:>10.4f}{r['ndcg']:>10.4f}{r['coverage']:>12.4f}")
    print("=" * 64)

    base = results["two_tower"]["recall"]
    two = results["two_stage"]["recall"]
    delta = (two / base - 1) * 100 if base > 0 else 0.0
    print(f"\n  Two-stage vs two-tower-alone recall: {delta:+.1f}%")
    if delta >= 0:
        print("  The LR rerank added value on top of two-tower retrieval.")
    else:
        print("  Honest result: the LR rerank HURT the two-tower candidates. The stage-2")
        print("  features are popularity-flavored (item_pop/user_pop/user_cat_affinity),")
        print("  so reranking pushes popular items up and undoes the two-tower's")
        print("  personalization + long-tail coverage. A two-stage system is only as")
        print("  good as its stage-2 signal -- architecture alone buys nothing (Rules 4, 14).")

    save_results(PHASE_DIR, results)

    print("\nNext steps:")
    print("  - Give stage 2 features it can actually rank with: two-tower similarity")
    print("    score as a feature, recency, brand/price affinity -> earn the rerank.")
    print("  - Swap LR -> a gradient-boosted or listwise ranker once features carry signal.")
    print("  - Wire this TwoStageRanker into the Phase 3 service in place of the")
    print("    popularity candidate generator.\n")
    return results


if __name__ == "__main__":
    main()
