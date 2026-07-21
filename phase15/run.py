"""
Phase 15 -- Entry Point: The Contextual Bandit (LinUCB)
========================================================
Extends Phase 14 from "one best item for everyone" to "the best item depends on the
user." We build a contextual world where each arm's reward genuinely varies with a
user context vector, then grade three policies on it:

    1. Context-free Thompson  -- Phase 14's winner, but BLIND to context.
    2. LinUCB (alpha=0)       -- uses context, but pure exploitation (no bonus).
    3. LinUCB (alpha=1)       -- uses context AND explores where its per-arm model
                                 is uncertain.

Two lessons in one table: context helps (2 vs 1), and exploration still helps on top
of context (3 vs 2) -- Phase 14's lesson, now per-user.

HONESTY NOTE: like Phases 8-14's causal/online experiments, this is a *simulation*
(you can't A/B a live policy inside a static log). But each arm's BASE rate (its
context-independent intercept) is a REAL KuaiRand-Pure per-item engagement rate; only
the context-dependent part is synthetic, so the world stays grounded in the data.

Usage:
    cd phase15 && python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events
from signals import POSITIVE_SIGNALS
from results_io import save_results
import linucb as L

PHASE_DIR = Path(__file__).parent

SEED = 42
N_ARMS = 12
N_FEATURES = 4          # user features; context dim = N_FEATURES + 1 (intercept)
DIM = N_FEATURES + 1
N_ROUNDS = 8_000
N_TRIALS = 25
CTX_WEIGHT = 0.25       # spread of each arm's context sensitivity
ALPHA = 1.0
CURVE_POINTS = 60


def build_base_rates(n_arms: int) -> np.ndarray:
    """Real KuaiRand per-item engagement rates for the top-N exposed items."""
    events = load_events()
    per_item = (
        events.group_by("item_id")
        .agg(total=pl.len(),
             positives=pl.col("event_type").is_in(list(POSITIVE_SIGNALS)).sum())
        .sort("total", descending=True)
        .head(n_arms)
    )
    rates = (per_item["positives"] / per_item["total"]).to_numpy().astype(np.float64)
    return np.clip(rates, 0.05, 0.95)


def _downsample(curve: np.ndarray, points: int) -> np.ndarray:
    idx = np.linspace(0, len(curve) - 1, points).astype(int)
    return curve[idx]


def make_policy(kind: str):
    if kind == "thompson":
        return L.ContextFreeThompson(N_ARMS, DIM)
    if kind == "linucb_greedy":
        return L.LinUCB(N_ARMS, DIM, alpha=0.0)
    return L.LinUCB(N_ARMS, DIM, alpha=ALPHA)


def main():
    print("\n=== Phase 15: The Contextual Bandit (LinUCB) ===\n")

    print(f"Step 1/3: Build {N_ARMS} arms (real KuaiRand base rates) + a context world...")
    base_rates = build_base_rates(N_ARMS)
    print(f"  base rates: min={base_rates.min():.3f} max={base_rates.max():.3f} "
          f"mean={base_rates.mean():.3f} | context dim = {DIM} ({N_FEATURES} user feats + bias)")

    policies = [
        ("thompson", "Context-free Thompson"),
        ("linucb_greedy", "LinUCB (alpha=0, greedy)"),
        ("linucb", f"LinUCB (alpha={ALPHA})"),
    ]

    print(f"\nStep 2/3: Run {N_TRIALS} worlds x {N_ROUNDS:,} rounds...")
    agg = {k: {"regret": [], "best": []} for k, _ in policies}
    curves = {k: np.zeros(CURVE_POINTS) for k, _ in policies}

    for trial in range(N_TRIALS):
        rng = np.random.default_rng(SEED + trial)
        # True model: real base rate as intercept, synthetic per-arm context weights.
        true_theta = np.zeros((N_ARMS, DIM))
        true_theta[:, 0] = base_rates
        true_theta[:, 1:] = rng.normal(0.0, CTX_WEIGHT, (N_ARMS, N_FEATURES))
        # Context stream: bias feature = 1, user features ~ uniform[-1, 1].
        contexts = np.ones((N_ROUNDS, DIM))
        contexts[:, 1:] = rng.uniform(-1.0, 1.0, (N_ROUNDS, N_FEATURES))

        for key, _ in policies:
            p_rng = np.random.default_rng(1000 + trial)   # same reward draws per policy
            out = L.simulate_contextual(true_theta, make_policy(key), contexts, p_rng)
            agg[key]["regret"].append(float(out["cum_regret"][-1]))
            agg[key]["best"].append(out["best_arm_rate"])
            curves[key] += _downsample(out["cum_regret"], CURVE_POINTS)

    results = {}
    for key, _ in policies:
        results[key] = {
            "final_regret": float(np.mean(agg[key]["regret"])),
            "regret_std": float(np.std(agg[key]["regret"])),
            "best_arm_pct": float(np.mean(agg[key]["best"])),
        }

    print("\nStep 3/3: Compare (means over worlds)...\n")
    print("=" * 66)
    print(f"  {'Policy':<28}{'Regret':>10}{'+/-sd':>9}{'Best-arm%':>12}")
    print("-" * 66)
    for key, label in policies:
        r = results[key]
        print(f"  {label:<28}{r['final_regret']:>10.0f}{r['regret_std']:>9.0f}"
              f"{r['best_arm_pct']*100:>11.0f}%")
    print("=" * 66)

    tf, lu = results["thompson"], results["linucb"]
    print(f"\n  LinUCB vs context-free Thompson: "
          f"{(1 - lu['final_regret']/tf['final_regret'])*100:.0f}% lower regret, "
          f"picks the per-user best arm {lu['best_arm_pct']*100:.0f}% vs "
          f"{tf['best_arm_pct']*100:.0f}%.")
    print("\n  Reading the numbers:")
    print("  - Context-free Thompson can only learn each arm's AVERAGE rate, so on a")
    print("    world where the best arm depends on the user it is structurally stuck.")
    print("  - LinUCB conditions on context: it learns a per-arm reward model and")
    print("    personalizes -- big regret drop, far more per-user-best picks.")
    print("  - alpha=0 vs alpha=1 shows exploration STILL pays with context: the")
    print("    uncertainty bonus finds good arms in under-seen contexts faster.")

    save_results(PHASE_DIR, {
        **results,
        "_regret_curves": {k: [round(float(v), 2) for v in (curves[k] / N_TRIALS)] for k, _ in policies},
        "_curve_x": [int(x) for x in _downsample(np.arange(N_ROUNDS), CURVE_POINTS)],
    })

    print("\nNext steps:")
    print("  - Close the loop: feed LinUCB's exploration log into Phase 9's OPL and")
    print("    learn an even better policy off-policy, then redeploy.")
    print("  - Add a min-propensity floor / cap alpha so exploration never tanks UX.")
    print("  - Real contexts: use the two-tower user vector (Phase 1) as x.\n")
    return results


if __name__ == "__main__":
    main()
