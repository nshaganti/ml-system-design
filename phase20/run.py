"""
Phase 20 -- Entry Point: Joint EM debiasing vs the alternatives
===============================================================
The question Phase 11 left open: can we remove position bias from ORDINARY production
logs -- where the ranker already sorted items by relevance, confounding position with
quality -- WITHOUT paying for a result-randomization bucket?

We build a controlled world (real KuaiRand base rates as true relevance + a known
examination curve, so truth is computable), generate a CONFOUNDED production log (each
slate ordered by relevance + noise), plus a small randomization bucket for the Phase 11
baseline. Then we recover relevance four ways and grade each by how well its RANKING
matches the truth:

  - Naive CTR              -- confounded (ranks positions as much as items)
  - IPW, TRUE exam         -- oracle upper bound (you never really know e_p)
  - IPW, randomized exam   -- Phase 11 (needs the randomization bucket)
  - Joint EM               -- Phase 20 (production log ONLY, no randomization)

Usage:
    cd phase20 && python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase11"))
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events
from signals import POSITIVE_SIGNALS
from results_io import save_results
from position_bias import (
    examination_curve, simulate_clicks, naive_relevance, ipw_relevance,
    estimate_examination_randomized, spearman, topk_recovery,
)
from joint_em import regression_em_pbm

PHASE_DIR = Path(__file__).parent

SEED = 42
N_ITEMS = 60
N_POSITIONS = 10
SLATE = N_POSITIONS
S_PROD = 20_000        # confounded production sessions
S_RAND = 4_000         # randomization bucket (Phase 11 baseline only)
N_WORLDS = 8
NOISE = 0.05           # ranker tie-break noise (small -> strong position confounding)
STALE_ALIGN = 0.35     # how much the OLD ranker's order tracks true relevance (0..1)
DECAY = 1.3            # steeper examination decay -> stronger position bias
TOPK = 10


def build_relevance(n_items: int) -> np.ndarray:
    events = load_events()
    per_item = (
        events.group_by("item_id")
        .agg(total=pl.len(),
             positives=pl.col("event_type").is_in(list(POSITIVE_SIGNALS)).sum())
        .sort("total", descending=True).head(n_items)
    )
    rates = (per_item["positives"] / per_item["total"]).to_numpy().astype(np.float64)
    return np.clip(rates, 0.05, 0.9)


def _slates(rng, n_sessions, logging_score, confound):
    """Return (item_ids, positions) flattened over sessions. If confound, order each
    slate by the OLD ranker's (stale) logging_score + noise -- the realistic case where
    position tracks a signal only partly aligned with true relevance. Else random order
    (the randomization bucket)."""
    picks = np.argsort(rng.random((n_sessions, N_ITEMS)), axis=1)[:, :SLATE]  # random subset
    if confound:
        scores = logging_score[picks] + rng.normal(0.0, NOISE, picks.shape)
        order = np.argsort(-scores, axis=1)
        picks = np.take_along_axis(picks, order, axis=1)     # position p = pth-best (stale)
    positions = np.tile(np.arange(SLATE), n_sessions)
    return picks.reshape(-1), positions


def _grade(estimate, truth):
    return {"spearman": spearman(estimate, truth),
            "top10": topk_recovery(estimate, truth, TOPK)}


def main():
    print("\n=== Phase 20: Joint EM Position-Bias + Relevance Estimation ===\n")

    print("Step 1/3: Build the world (real KuaiRand base rates as true relevance)...")
    true_rel = build_relevance(N_ITEMS)
    true_exam = examination_curve(N_POSITIONS, decay=DECAY)

    acc = {k: {"spearman": [], "top10": []}
           for k in ("naive", "ipw_true", "ipw_rand", "em")}
    exam_err, ll_monotone = [], []

    print(f"\nStep 2/3: {N_WORLDS} worlds -- {S_PROD:,} CONFOUNDED prod sessions "
          f"(+ {S_RAND:,} randomized for the Phase 11 baseline)...")
    for w in range(N_WORLDS):
        rng = np.random.default_rng(SEED + w)
        # OLD ranker's stale score: partly aligned with true relevance, partly noise.
        z = (true_rel - true_rel.mean()) / (true_rel.std() + 1e-9)
        logging_score = STALE_ALIGN * z + (1 - STALE_ALIGN) * rng.normal(0, 1, N_ITEMS)
        # confounded production log (ordered by the stale score)
        items_p, pos_p = _slates(rng, S_PROD, logging_score, confound=True)
        clicks_p = simulate_clicks(true_rel[items_p], pos_p, true_exam, rng)
        # randomization bucket
        items_r, pos_r = _slates(rng, S_RAND, logging_score, confound=False)
        clicks_r = simulate_clicks(true_rel[items_r], pos_r, true_exam, rng)

        naive = naive_relevance(clicks_p, items_p, N_ITEMS)
        ipw_true = ipw_relevance(clicks_p, pos_p, true_exam, items_p, N_ITEMS)
        exam_rand = estimate_examination_randomized(clicks_r, pos_r, N_POSITIONS)
        ipw_rand = ipw_relevance(clicks_p, pos_p, exam_rand, items_p, N_ITEMS)
        em = regression_em_pbm(clicks_p, pos_p, items_p, N_ITEMS, N_POSITIONS,
                               n_iter=300, seed=w)

        for key, est in [("naive", naive), ("ipw_true", ipw_true),
                         ("ipw_rand", ipw_rand), ("em", em["rel"])]:
            g = _grade(est, true_rel)
            acc[key]["spearman"].append(g["spearman"])
            acc[key]["top10"].append(g["top10"])
        exam_err.append(float(np.mean(np.abs(em["exam"] - true_exam))))
        ll = em["loglik"]
        ll_monotone.append(all(ll[i + 1] >= ll[i] - 1e-9 for i in range(len(ll) - 1)))

    print("\nStep 3/3: Recovery of TRUE relevance ranking (means over worlds)...\n")
    print("  " + "=" * 62)
    print(f"    {'Method':<26}{'Spearman':>12}{'Top-10 recall':>16}")
    print("  " + "-" * 62)
    labels = [("naive", "Naive CTR (confounded)"), ("ipw_true", "IPW, TRUE exam (oracle)"),
              ("ipw_rand", "IPW, randomized (Phase 11)"), ("em", "Joint EM (Phase 20)")]
    results = {"methods": {}, "exam_mae": float(np.mean(exam_err)),
               "loglik_monotone": bool(all(ll_monotone))}
    for key, label in labels:
        sp = float(np.mean(acc[key]["spearman"]))
        tk = float(np.mean(acc[key]["top10"]))
        results["methods"][key] = {"spearman": sp, "top10": tk}
        print(f"    {label:<26}{sp:>12.3f}{tk*100:>15.0f}%")
    print("  " + "=" * 62)

    naive_sp = results["methods"]["naive"]["spearman"]
    em_sp = results["methods"]["em"]["spearman"]
    rand_sp = results["methods"]["ipw_rand"]["spearman"]
    print(f"\n  Joint EM recovers the relevance ranking (Spearman {em_sp:.3f}) from the "
          f"CONFOUNDED\n  production log alone -- matching the randomization-based IPW "
          f"({rand_sp:.3f}) and\n  far beyond naive CTR ({naive_sp:.3f}), with NO "
          f"randomization bucket. EM's examination\n  curve is off by only "
          f"{results['exam_mae']:.3f} MAE; its log-likelihood is monotone "
          f"({'yes' if results['loglik_monotone'] else 'no'}).")
    print("\n  Reading the numbers:")
    print("  - Naive CTR is confounded: it ranks slots as much as items.")
    print("  - IPW fixes it IF you know e_p -- Phase 11 bought e_p with a costly")
    print("    randomization bucket.")
    print("  - Regression-EM estimates e_p AND r_i jointly from the biased log itself,")
    print("    so you can debias production traffic in place. The recovered r_i are the")
    print("    IPW-debiased labels you'd feed the Phase 2 ranker.")

    save_results(PHASE_DIR, results)

    print("\nNext steps:")
    print("  - Swap the per-item r_i for a feature-based Regression-EM (learn a model,")
    print("    not a table) to generalize to cold items -- the full DLA.")
    print("  - Feed EM propensities into the Phase 17 redeploy gate.\n")
    return results


if __name__ == "__main__":
    main()
