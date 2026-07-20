"""
Phase 11 -- Entry Point: Position-Bias Debiasing (controlled simulation)
========================================================================
The scenario, end to end:

  1. Ground truth: each item has a true relevance r_i (unknown to us).
  2. Production ranks all items by a PRIOR score (an imperfect guess at r_i), so
     each item is pinned to a slot. Higher slots get far more examination.
  3. We log clicks from the Position-Based Model: click ~ Bernoulli(e_slot * r_i).
  4. We try to recover the item ranking three ways and grade each against truth:
       * naive CTR                -> confounded by slot examination.
       * IPW with TRUE e_p        -> oracle propensities, recovers truth.
       * IPW with ESTIMATED e_p   -> e_p learned from a result-randomization bucket,
                                     recovers truth WITHOUT knowing the curve.

HONESTY NOTE: KuaiRand-Pure logs no on-screen position, so this phase is a
controlled simulation (see position_bias.py). It is the physics-lab demo of a
technique you'd deploy verbatim on position-carrying logs.

Usage:
    cd phase11 && python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent))

from results_io import save_results
import position_bias as pb

PHASE_DIR = Path(__file__).parent

SEED = 42
N_ITEMS = 40
DECAY = 0.6            # examination ~1/(1+pos)^0.6: top slot 1.0 -> bottom ~0.10
PRIOR_NOISE = 0.5      # a genuinely imperfect production ranker (weakly correlated)
BIASED_SESSIONS = 1500 # impressions per item in the production-order log
RANDOM_SESSIONS = 800  # impressions per item in the randomization bucket
TOP_K = 10


def main():
    print("\n=== Phase 11: Position-Bias Debiasing (controlled simulation) ===\n")
    rng = np.random.default_rng(SEED)
    items = np.arange(N_ITEMS)

    print("Step 1/4: Ground-truth relevance + biased production ranking...")
    true_rel = rng.uniform(0.05, 0.6, size=N_ITEMS)
    # Production ranks by an imperfect prior; each item gets pinned to a slot.
    prior = true_rel + rng.normal(0, PRIOR_NOISE, size=N_ITEMS)
    slot_of_item = np.argsort(np.argsort(-prior))     # 0 = top slot
    exam_true = pb.examination_curve(N_ITEMS, DECAY)
    print(f"  {N_ITEMS} items | examination top->bottom: "
          f"{exam_true[0]:.2f} -> {exam_true[-1]:.3f} (decay={DECAY})")
    print(f"  prior vs truth rank corr: {pb.spearman(prior, true_rel):.3f} "
          f"(production's ranking is imperfect on purpose)")

    print("\nStep 2/4: Log clicks from the Position-Based Model...")
    # Biased log: every item always shown at its production slot.
    ids_b = np.repeat(items, BIASED_SESSIONS)
    pos_b = np.repeat(slot_of_item, BIASED_SESSIONS)
    clicks_b = pb.simulate_clicks(true_rel[ids_b], pos_b, exam_true, rng)
    # Randomization bucket: each session shuffles items across slots.
    pos_r = np.concatenate([rng.permutation(N_ITEMS) for _ in range(RANDOM_SESSIONS)])
    ids_r = np.tile(items, RANDOM_SESSIONS)
    clicks_r = pb.simulate_clicks(true_rel[ids_r], pos_r, exam_true, rng)
    print(f"  biased impressions: {len(clicks_b):,} | random impressions: {len(clicks_r):,}")

    print("\nStep 3/4: Estimate relevance three ways...")
    naive = pb.naive_relevance(clicks_b, ids_b, N_ITEMS)
    ipw_oracle = pb.ipw_relevance(clicks_b, pos_b, exam_true, ids_b, N_ITEMS)
    exam_est = pb.estimate_examination_randomized(clicks_r, pos_r, N_ITEMS)
    ipw_est = pb.ipw_relevance(clicks_b, pos_b, exam_est, ids_b, N_ITEMS)

    print("\nStep 4/4: Grade each ranking against the true relevance...")
    rows = [
        ("naive", "Naive CTR", naive),
        ("ipw_true", "IPW (true propensity)", ipw_oracle),
        ("ipw_estimated", "IPW (estimated propensity)", ipw_est),
    ]
    print("\n" + "=" * 64)
    print(f"  {'Estimator':<30}{'Spearman':>12}{'Top-10 recovery':>20}")
    print("-" * 64)
    results = {"exam_curve_spearman": pb.spearman(exam_est, exam_true)}
    for key, label, est in rows:
        sp = pb.spearman(est, true_rel)
        rec = pb.topk_recovery(est, true_rel, TOP_K)
        print(f"  {label:<30}{sp:>12.3f}{rec:>19.0%}")
        results[key] = {"spearman": sp, "topk_recovery": rec}
    print("=" * 64)
    print(f"  Examination-curve recovery (estimated vs true): "
          f"Spearman {results['exam_curve_spearman']:.3f}")
    print()
    print("Reading the numbers:")
    print("  - Naive CTR ranks POSITIONS as much as items -> it mis-ranks relevance.")
    print("  - IPW with the true examination curve divides out the slot effect and")
    print("    recovers the true ordering.")
    print("  - You don't need the true curve: a small result-randomization bucket")
    print("    ESTIMATES it (same idea as Part II's random log), and IPW with the")
    print("    estimated curve recovers the ordering just as well.")

    save_results(PHASE_DIR, results)

    print("\nNext steps:")
    print("  - Fit a full PBM/DLA jointly (relevance + examination) via EM.")
    print("  - On logs that DO carry position, use these exact estimators on real")
    print("    clicks; use IPW-weighted labels to train the Phase 2 ranker.")
    print("  - Combine with Part II: position bias and selection bias are the same")
    print("    disease (confounded logs) with the same cure (known/known-able props).\n")
    return results


if __name__ == "__main__":
    main()
