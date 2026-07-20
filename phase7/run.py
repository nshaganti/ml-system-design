"""
Phase 7 -- Entry Point
=======================
Evaluates the community-standard task: SESSION-BASED next-item prediction,
leave-one-out, Recall@20 / MRR@20 / NDCG@20 -- the protocol public notebooks and
the session-rec literature actually use.

Protocol:
  1. Temporal 80/20 split (same as every other phase).
  2. Fit co-visitation on the training events.
  3. For each TEST session with >=2 items: hide the LAST item (the target),
     feed the earlier items as context, predict top-20, score the hit.
  4. Compare co-visitation vs a popularity baseline under the SAME protocol.

Why this matters: our Phases 0-2 measured the next positive action (engagement)
over the full catalog (~0.07 Recall@20). That is a *different, harder task* than
next-item-in-session, so the numbers here are NOT comparable to those. On
KuaiRand co-visitation beats popularity by ~61% -- a real but modest win, because
the small catalog makes popularity a strong session baseline (on a sparse
e-commerce log the same method often wins by far more).

Usage:
    cd phase7
    python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events
from evaluate import temporal_split
from metrics import recall_at_k, ndcg_at_k, mean

from covisitation import CoVisitationRecommender, sessionize, session_item_lists
from session_eval import build_test_cases, reciprocal_rank

from results_io import save_results

PHASE_DIR = Path(__file__).parent

MAX_TEST_SESSIONS = 30_000
K = 20


def evaluate(recommender, cases, mode: str) -> dict:
    recalls, mrrs, ndcgs = [], [], []
    for context, target in cases:
        if mode == "covis":
            recs = recommender.recommend(context, n=K, exclude=set(context))
        else:  # popularity
            recs = recommender.popular(n=K, exclude=set(context))
        relevant = {target}
        recalls.append(recall_at_k(recs, relevant, K))
        ndcgs.append(ndcg_at_k(recs, relevant, K))
        mrrs.append(reciprocal_rank(recs, target, K))
    return {"recall": mean(recalls), "mrr": mean(mrrs), "ndcg": mean(ndcgs)}


def main():
    print("\n=== Phase 7: Session-Based Co-Visitation (community protocol) ===\n")

    print("Step 1/3: Loading + temporal split...")
    events = load_events()
    train_events, test_events, _ = temporal_split(events, train_fraction=0.8)

    print("\nStep 2/3: Fitting co-visitation on training sessions...")
    covis = CoVisitationRecommender().fit(train_events)

    cases = build_test_cases(test_events)
    if len(cases) > MAX_TEST_SESSIONS:
        cases = cases[:MAX_TEST_SESSIONS]
    print(f"  leave-one-out test cases (sessions with >=2 items): {len(cases):,}")

    print("\nStep 3/3: Evaluating (leave-one-out next-item)...")
    pop = evaluate(covis, cases, mode="popularity")
    cov = evaluate(covis, cases, mode="covis")

    print("\n" + "=" * 60)
    print("  SESSION-BASED NEXT-ITEM PREDICTION  (higher is better)")
    print("=" * 60)
    print(f"  {'Metric':<12} {'Popularity':>12} {'Co-visitation':>14} {'Lift':>8}")
    print(f"  {'-'*48}")
    for key, label in [("recall", "Recall@20"), ("mrr", "MRR@20"), ("ndcg", "NDCG@20")]:
        lift = (cov[key] / pop[key] - 1) * 100 if pop[key] > 0 else 0.0
        print(f"  {label:<12} {pop[key]:>12.4f} {cov[key]:>14.4f} {lift:>+7.0f}%")
    print("=" * 60)
    print("  Context for these numbers:")
    print(f"  - Our full-catalog next-positive-action Recall@20 (Phase 0-2) was ~0.07.")
    print(f"  - This session next-ITEM Recall@20 is ~{cov['recall']:.2f} -- a DIFFERENT,")
    print(f"    easier task, so it is not comparable to the full-catalog numbers.")
    print(f"  - Co-visitation beats popularity by using the session co-occurrence")
    print(f"    signal; the win is modest here because the catalog is small.")

    print("\nNext steps:")
    print("  - Blend co-vis candidates INTO the Phase 3 service candidate union.")
    print("  - Add a sequence model (GRU4Rec/SASRec) and compare on THIS protocol.")
    print("  - See docs/benchmarking-vs-literature.md for the full gap analysis.\n")
    save_results(PHASE_DIR, {"popularity": pop, "covisitation": cov})
    return {"popularity": pop, "covisitation": cov}


if __name__ == "__main__":
    main()
