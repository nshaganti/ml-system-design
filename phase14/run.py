"""
Phase 14 -- Entry Point: The Explore-and-Learn Loop
====================================================
Runs a Bernoulli multi-armed bandit under greedy / epsilon-greedy / Thompson and
shows the trade the whole course has been circling:

  * Greedy exploits and wins early rounds, then LOCKS onto a lucky-but-mediocre arm
    -- high regret, and a log with no support on other arms (Phase 8's bias, live).
  * Epsilon-greedy and Thompson pay a little exploration cost and end with far lower
    regret AND a log that supports off-policy evaluation/learning (Phases 8-9-11).

HONESTY NOTE: like Phases 8-11's causal experiments, the online loop is a
*simulation* -- you cannot A/B a bandit inside a static log. But the arms are not
invented: each arm's true reward rate is a REAL per-item engagement rate measured
from KuaiRand-Pure, so the world the policies act in is grounded in the data.

Usage:
    cd phase14 && python run.py
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
from repro import set_global_seed
import bandit as B

PHASE_DIR = Path(__file__).parent

SEED = 42
N_ARMS = 50            # top-N items by exposure become the arms
N_ROUNDS = 50_000
N_TRIALS = 60          # average over seeds -- one run lets greedy get lucky
EPSILON = 0.1
CURVE_POINTS = 60      # downsampled regret trajectory for the HTML report


def build_arm_rates(n_arms: int) -> np.ndarray:
    """
    Real per-item reward rates from KuaiRand-Pure: for the most-exposed items,
    rate = (# positive-signal events) / (# total events). Clipped away from 0/1 so
    every arm is a genuine Bernoulli. This grounds the simulated world in real data.
    """
    events = load_events()
    per_item = (
        events
        .group_by("item_id")
        .agg(
            total=pl.len(),
            positives=pl.col("event_type").is_in(list(POSITIVE_SIGNALS)).sum(),
        )
        .sort("total", descending=True)
        .head(n_arms)
    )
    rates = (per_item["positives"] / per_item["total"]).to_numpy().astype(np.float64)
    return np.clip(rates, 0.01, 0.99)


def _downsample(curve: np.ndarray, points: int) -> list[float]:
    idx = np.linspace(0, len(curve) - 1, points).astype(int)
    return [round(float(curve[i]), 2) for i in idx]


def main():
    set_global_seed()   # reproducible bandit draws
    print("\n=== Phase 14: The Explore-and-Learn Loop (bandit) ===\n")

    print(f"Step 1/3: Build {N_ARMS} arms from REAL KuaiRand per-item rates...")
    true_rates = build_arm_rates(N_ARMS)
    best = int(np.argmax(true_rates))
    print(f"  reward rates: min={true_rates.min():.3f} max={true_rates.max():.3f} "
          f"mean={true_rates.mean():.3f} | best arm = #{best} @ {true_rates[best]:.3f}")

    print(f"\nStep 2/3: Run {N_TRIALS} trials x {N_ROUNDS:,} rounds per strategy...")
    policies = [
        ("greedy", "Greedy (exploit only)", B.select_greedy, 1),
        ("epsilon_greedy", f"Epsilon-greedy (e={EPSILON})", B.make_epsilon_greedy(EPSILON), 1),
        ("thompson", "Thompson sampling", B.select_thompson, 0),
    ]

    results = {}
    curves = {}
    for key, _, selector, warmup in policies:
        finals, best_pcts, supports, founds = [], [], [], []
        curve_acc = np.zeros(CURVE_POINTS)
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(SEED + trial)   # a different world each trial
            out = B.simulate(true_rates, selector, N_ROUNDS, rng, warmup=warmup)
            regret = B.cumulative_regret(true_rates, out["chosen"])
            finals.append(float(regret[-1]))
            best_pcts.append(B.best_arm_fraction(true_rates, out["chosen"]))
            supports.append(B.arm_support(out["successes"], out["failures"]))
            founds.append(1.0 if B.identified_best(out["successes"], out["failures"], true_rates) else 0.0)
            curve_acc += np.array(_downsample(regret, CURVE_POINTS))
        results[key] = {
            "final_regret": float(np.mean(finals)),
            "regret_std": float(np.std(finals)),
            "best_arm_pct": float(np.mean(best_pcts)),
            "arm_support": float(np.mean(supports)),
            "found_best_pct": float(np.mean(founds)),
        }
        curves[key] = [round(v, 2) for v in (curve_acc / N_TRIALS)]

    print("\nStep 3/3: Compare (means over trials)...\n")
    print("=" * 76)
    print(f"  {'Strategy':<26}{'Regret':>10}{'+/-sd':>8}{'Best-arm%':>11}{'Support':>9}{'FoundBest%':>11}")
    print("-" * 76)
    for key, label, _, _ in policies:
        r = results[key]
        print(f"  {label:<26}{r['final_regret']:>10.0f}{r['regret_std']:>8.0f}"
              f"{r['best_arm_pct']*100:>10.0f}%{r['arm_support']*100:>8.0f}%"
              f"{r['found_best_pct']*100:>10.0f}%")
    print("=" * 76)

    g, th = results["greedy"], results["thompson"]
    print(f"\n  Thompson vs greedy: {(1 - th['final_regret']/g['final_regret'])*100:.0f}% "
          f"lower mean regret, finds the best arm {th['found_best_pct']*100:.0f}% of")
    print(f"  the time (greedy {g['found_best_pct']*100:.0f}%), with {th['arm_support']*100:.0f}% "
          f"arm support vs greedy's {g['arm_support']*100:.0f}%.")
    print("\n  Reading the numbers:")
    print("  - Greedy is a GAMBLE: averaged over worlds its regret is high and high-")
    print("    variance -- one lucky warmup looks great, but it rarely finds the true")
    print("    best arm, and its ongoing log supports almost no arms (Phase 8's bias).")
    print("  - Naive epsilon-greedy explores WASTEFULLY (uniform forever): full support,")
    print("    but it keeps paying to pull known-bad arms.")
    print("  - Thompson explores in proportion to uncertainty: lowest regret, reliably")
    print("    finds the best arm, AND keeps full support -- the unbiased data Phases")
    print("    8-9-11 assumed. Exploration is the online source of causal ground truth.")

    save_results(PHASE_DIR, {**results, "_regret_curves": curves,
                             "_curve_x": _downsample(np.arange(N_ROUNDS), CURVE_POINTS)})

    print("\nNext steps:")
    print("  - Contextual bandit: condition arm choice on user features (LinUCB).")
    print("  - Feed the exploration log into Phase 9's OPL to CLOSE the loop:")
    print("    explore online -> learn off-policy -> deploy -> explore again.")
    print("  - Add a safety floor (min propensity) so exploration never tanks UX.\n")
    return results


if __name__ == "__main__":
    main()
