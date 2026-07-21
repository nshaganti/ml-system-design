"""
Phase 17 -- Entry Point: Real Context + Safety-Gated Redeploys
==============================================================
The production-real finale. Runs the closed loop on REAL per-user contexts and shows
that gating redeploys through off-policy evaluation (Phase 8) stops bad iterations
from shipping.

    1. Ungated loop   -- refit and redeploy blindly every iteration (Phase 16).
    2. Gated loop     -- OPE-score the candidate vs the incumbent on a fresh log;
                         only redeploy if it wins. Bad candidates are BLOCKED.

With small, noisy batches some learned candidates are worse than what's live. The
ungated loop ships them (value dips); the gate rejects them (higher, steadier value).

HONESTY NOTE: still a *simulation* of the reward (you can't A/B inside a static log),
but the CONTEXTS are real -- standardized per-user activity + signal-mix features
from KuaiRand-Pure -- so the loop acts on a realistic, correlated context distribution.

Usage:
    cd phase17 && python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase15"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase16"))
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events
from signals import STRONG, MEDIUM
from results_io import save_results
import safe_loop as SL

PHASE_DIR = Path(__file__).parent

SEED = 42
N_ARMS = 12
N_ITERATIONS = 15
BATCH = 150            # small + noisy on purpose -> some candidates regress
EVAL_CONTEXTS = 2_000
N_WORLDS = 15
EPSILON = 0.1
CTX_WEIGHT = 0.25
MIN_PROP = 0.02
MIN_USER_EVENTS = 10
POISON_ITER = 8        # a logging bug flips this iteration's rewards


def build_real_contexts() -> np.ndarray:
    """
    Standardized per-user feature vectors from KuaiRand-Pure -> a realistic (non-
    uniform, correlated) context pool. Features: log activity, strong-signal ratio,
    medium-signal ratio, distinct-item ratio. Returns (n_users, 5) with a bias 1.
    """
    events = load_events()
    per_user = (
        events.group_by("user_id")
        .agg(
            total=pl.len(),
            strong=(pl.col("event_type") == STRONG).sum(),
            medium=(pl.col("event_type") == MEDIUM).sum(),
            distinct=pl.col("item_id").n_unique(),
        )
        .filter(pl.col("total") >= MIN_USER_EVENTS)
    )
    feats = np.column_stack([
        np.log(per_user["total"].to_numpy()),
        (per_user["strong"] / per_user["total"]).to_numpy(),
        (per_user["medium"] / per_user["total"]).to_numpy(),
        (per_user["distinct"] / per_user["total"]).to_numpy(),
    ]).astype(np.float64)
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-9)   # standardize
    ctx = np.ones((len(feats), feats.shape[1] + 1))
    ctx[:, 1:] = feats
    return ctx


def build_base_rates(n_arms: int) -> np.ndarray:
    events = load_events()
    from signals import POSITIVE_SIGNALS
    per_item = (
        events.group_by("item_id")
        .agg(total=pl.len(),
             positives=pl.col("event_type").is_in(list(POSITIVE_SIGNALS)).sum())
        .sort("total", descending=True).head(n_arms)
    )
    rates = (per_item["positives"] / per_item["total"]).to_numpy().astype(np.float64)
    return np.clip(rates, 0.05, 0.95)


def main():
    print("\n=== Phase 17: Real Context + Safety-Gated Redeploys ===\n")

    print("Step 1/3: Build REAL per-user contexts + arms from KuaiRand...")
    ctx_pool = build_real_contexts()
    base_rates = build_base_rates(N_ARMS)
    dim = ctx_pool.shape[1]
    print(f"  context pool: {len(ctx_pool):,} real users x {dim} dims "
          f"(bias + activity + signal-mix); {N_ARMS} arms")

    print(f"\nStep 2/3: Run {N_WORLDS} worlds x {N_ITERATIONS} iters "
          f"(batch={BATCH}; a logging bug POISONS iter {POISON_ITER})...")
    scenarios = [
        ("ungated_clean", "Ungated, clean", False, None),
        ("gated_clean", "Gated, clean", True, None),
        ("ungated_poison", "Ungated, POISONED", False, POISON_ITER),
        ("gated_poison", "Gated, POISONED", True, POISON_ITER),
    ]
    traj = {k: np.zeros(N_ITERATIONS) for k, *_ in scenarios}
    mins = {k: [] for k, *_ in scenarios}
    skylines, floors, blocked_total = [], [], 0

    for w in range(N_WORLDS):
        wr = np.random.default_rng(SEED + w)
        true_theta = np.zeros((N_ARMS, dim))
        true_theta[:, 0] = base_rates
        true_theta[:, 1:] = wr.normal(0.0, CTX_WEIGHT, (N_ARMS, dim - 1))
        eval_idx = wr.integers(0, len(ctx_pool), size=EVAL_CONTEXTS)
        eval_ctx = ctx_pool[eval_idx]
        skylines.append(SL.skyline_value(true_theta, eval_ctx))
        floors.append(SL.uniform_value(true_theta, eval_ctx))

        for key, _, gate, poison in scenarios:
            loop_rng = np.random.default_rng(3000 + w)
            out = SL.simulate_safe_loop(
                true_theta, ctx_pool, eval_ctx, loop_rng,
                n_iterations=N_ITERATIONS, batch_size=BATCH, epsilon=EPSILON,
                use_gate=gate, min_prop=MIN_PROP, poison_iter=poison)
            traj[key] += np.array(out["values"])
            mins[key].append(min(out["values"]))
            if key == "gated_poison":
                blocked_total += out["blocked"]

    skyline, floor = float(np.mean(skylines)), float(np.mean(floors))
    results = {}
    for key, _, _, _ in scenarios:
        curve = traj[key] / N_WORLDS
        final = float(curve[-1])
        gap = (final - floor) / (skyline - floor) if skyline > floor else 0.0
        results[key] = {"final_value": final, "gap_closed": gap,
                        "min_value": float(np.mean(mins[key])),
                        "trajectory": [round(float(v), 4) for v in curve]}
    results["gated_poison"]["blocked_per_world"] = blocked_total / N_WORLDS

    print("\nStep 3/3: Compare (means over worlds)...\n")
    print("=" * 72)
    print(f"  {'Scenario':<26}{'Final value':>13}{'% gap':>9}{'Worst deploy':>14}")
    print("-" * 72)
    print(f"  {'Uniform floor':<26}{floor:>13.4f}{'0%':>9}{'':>14}")
    for key, label, _, _ in scenarios:
        r = results[key]
        print(f"  {label:<26}{r['final_value']:>13.4f}"
              f"{r['gap_closed']*100:>8.0f}%{r['min_value']:>14.4f}")
    print(f"  {'Skyline (oracle)':<26}{skyline:>13.4f}{'100%':>9}{'':>14}")
    print("=" * 72)

    up, gp = results["ungated_poison"], results["gated_poison"]
    print(f"\n  The bug hit iter {POISON_ITER}. UNGATED shipped it: worst deployed value "
          f"cratered to {up['min_value']:.4f}. GATED blocked it "
          f"({gp['blocked_per_world']:.1f} rejects/world): worst still {gp['min_value']:.4f}.")
    print("\n  Reading the numbers:")
    print("  - Real contexts: the loop acts on correlated, non-uniform traffic")
    print("    (activity + signal-mix from real users), not tidy uniform noise.")
    print("  - With recency-window retraining, candidates are noisy; the gate helps")
    print("    even in the CLEAN case by validating on a fresh uniform-random bucket")
    print("    (Phase 8's unbiased log) and only shipping candidates that truly win.")
    print("  - When a bug ships a corrupt candidate, the insurance pays off: the OPE")
    print("    gate rejects it and the deployed value never craters. Priced in the")
    print("    good case, cashed in the bad.")
    print("  - The min-propensity floor bounds importance weights so one rare action")
    print("    can't blow up the gate's estimate (Phase 11's variance guard).")

    save_results(PHASE_DIR, {**results, "skyline": skyline, "uniform": floor,
                             "poison_iter": POISON_ITER,
                             "_iterations": list(range(1, N_ITERATIONS + 1))})

    print("\nNext steps:")
    print("  - Use the trained two-tower user embedding (Phase 1) as the context x.")
    print("  - Add a canary: deploy the candidate to a small traffic slice first.")
    print("  - Doubly-robust gate (Phase 8 DR) to cut the OPE estimate's variance.\n")
    return results


if __name__ == "__main__":
    main()
