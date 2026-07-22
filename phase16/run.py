"""
Phase 16 -- Entry Point: Closing the Loop
==========================================
Runs the full explore -> learn-off-policy -> redeploy cycle and shows what each
ingredient buys, by ablation:

    1. No exploration (epsilon=0)     -- learn once on greedy logs; the trap.
    2. Explore, but DON'T correct      -- epsilon-soft logs, naive (no IPS).
    3. Closed loop (explore + IPS)     -- the capstone: the deployed policy's TRUE
                                          value climbs toward the skyline.

Reference lines: the skyline (oracle per-context best) and the uniform floor.

HONESTY NOTE: this is a *simulation* -- you cannot run a live loop inside a static
log. But each arm's base rate is a REAL KuaiRand-Pure per-item engagement rate
(Phase 15's world), so the ceiling the loop climbs toward is grounded in the data.

Usage:
    cd phase16 && python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase15"))
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events
from signals import POSITIVE_SIGNALS
from results_io import save_results
from repro import set_global_seed
import loop as LP

PHASE_DIR = Path(__file__).parent

SEED = 42
N_ARMS = 12
N_FEATURES = 4
DIM = N_FEATURES + 1
N_ITERATIONS = 15
BATCH = 1_000
EVAL_CONTEXTS = 2_000
N_WORLDS = 15
EPSILON = 0.1
CTX_WEIGHT = 0.25


def build_base_rates(n_arms: int) -> np.ndarray:
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


def _make_world(base_rates: np.ndarray, rng: np.random.Generator):
    true_theta = np.zeros((N_ARMS, DIM))
    true_theta[:, 0] = base_rates
    true_theta[:, 1:] = rng.normal(0.0, CTX_WEIGHT, (N_ARMS, N_FEATURES))
    eval_ctx = np.ones((EVAL_CONTEXTS, DIM))
    eval_ctx[:, 1:] = rng.uniform(-1.0, 1.0, (EVAL_CONTEXTS, N_FEATURES))
    return true_theta, eval_ctx


def main():
    set_global_seed()   # reproducible loop
    print("\n=== Phase 16: Closing the Loop (explore -> learn -> redeploy) ===\n")

    print(f"Step 1/3: Build the contextual world ({N_ARMS} arms, real KuaiRand rates)...")
    base_rates = build_base_rates(N_ARMS)

    variants = [
        ("no_explore", "No exploration (trap)", 0.0, True),
        ("explore_no_ips", "Explore, no IPS", EPSILON, False),
        ("closed_loop", "Closed loop (explore+IPS)", EPSILON, True),
    ]

    print(f"\nStep 2/3: Run {N_WORLDS} worlds x {N_ITERATIONS} loop iterations "
          f"({BATCH:,} logged rows each)...")
    traj = {k: np.zeros(N_ITERATIONS) for k, *_ in variants}
    skylines, floors = [], []

    for w in range(N_WORLDS):
        world_rng = np.random.default_rng(SEED + w)
        true_theta, eval_ctx = _make_world(base_rates, world_rng)
        skylines.append(LP.skyline_value(true_theta, eval_ctx))
        floors.append(LP.uniform_value(true_theta, eval_ctx))
        for key, _, eps, ips in variants:
            loop_rng = np.random.default_rng(2000 + w)   # same draws across variants
            vals = LP.simulate_loop(
                true_theta, eval_ctx, loop_rng,
                n_iterations=N_ITERATIONS, batch_size=BATCH, epsilon=eps, use_ips=ips,
            )
            traj[key] += np.array(vals)

    skyline = float(np.mean(skylines))
    floor = float(np.mean(floors))
    results = {}
    for key, _, _, _ in variants:
        curve = (traj[key] / N_WORLDS)
        final = float(curve[-1])
        closed = (final - floor) / (skyline - floor) if skyline > floor else 0.0
        results[key] = {"final_value": final, "gap_closed": closed,
                        "trajectory": [round(float(v), 4) for v in curve]}

    print("\nStep 3/3: Compare TRUE deployed value (means over worlds)...\n")
    print("=" * 68)
    print(f"  {'Variant':<30}{'Final value':>14}{'% skyline gap':>16}")
    print("-" * 68)
    print(f"  {'Uniform floor':<30}{floor:>14.4f}{'0%':>16}")
    for key, label, _, _ in variants:
        r = results[key]
        print(f"  {label:<30}{r['final_value']:>14.4f}{r['gap_closed']*100:>15.0f}%")
    print(f"  {'Skyline (oracle)':<30}{skyline:>14.4f}{'100%':>16}")
    print("=" * 68)

    ne, cl, en = results["no_explore"], results["closed_loop"], results["explore_no_ips"]
    print(f"\n  Exploration is the hero: it lifts the deployed value from "
          f"{ne['gap_closed']*100:.0f}% of the skyline gap (no-explore trap) to "
          f"{cl['gap_closed']*100:.0f}% (closed loop).")
    print("\n  Reading the numbers:")
    print("  - No exploration LEARNS ONCE and stops: it only ever logs the arm it")
    print("    already likes, so every other arm's model stays at the prior and the")
    print("    deployed value plateaus low -- Phase 8/14's feedback trap, capping a")
    print("    live system's ceiling.")
    print("  - Exploring for COVERAGE is what breaks the trap: once every arm is seen")
    print("    across contexts, the off-policy step can learn a good policy for all of")
    print("    them and the value climbs toward the skyline.")
    print(f"  - Honest wrinkle: IPS barely moved the needle here (explore+IPS "
          f"{cl['gap_closed']*100:.0f}% vs explore-no-IPS {en['gap_closed']*100:.0f}%).")
    print("    With a WELL-SPECIFIED linear per-arm model, plain regression is already")
    print("    unbiased on a skewed context distribution, so IPS only adds variance")
    print("    (Phase 11's bias-variance tradeoff). Propensities earn their keep when")
    print("    the model is misspecified or you estimate policy VALUE directly (Phase 8).")

    save_results(PHASE_DIR, {**results, "skyline": skyline, "uniform": floor,
                             "_iterations": list(range(1, N_ITERATIONS + 1))})

    print("\nNext steps:")
    print("  - Use the two-tower user vector (Phase 1) as the real context x.")
    print("  - Add doubly-robust learning (Phase 8's DR) to cut IPS variance.")
    print("  - Guardrails: min-propensity floor + value monitoring (Phase 4) before")
    print("    each redeploy, so a bad iteration can't ship.\n")
    return results


if __name__ == "__main__":
    main()
