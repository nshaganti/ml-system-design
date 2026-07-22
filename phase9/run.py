"""
Phase 9 -- Entry Point: Off-Policy LEARNING on KuaiRand
=======================================================
Phase 8 proved that *evaluating* on the biased log overstates a policy's value by
2x. Phase 9 asks the next question: if we instead *learn* our policy from the
unbiased random log, do we get a genuinely BETTER policy? And how much unbiased
data do we actually need?

The experiment:
  1. r_biased(a) = reward rate per item on the BIASED production log.
  2. r_true(a)   = reward rate per item on the UNIFORM-RANDOM log (unconfounded).
  3. Learn three policies and grade each by its TRUE value V(pi) = sum_a pi(a) r_true(a):
       * pi_naive   = softmax(r_biased)  -- what Part I effectively did.
       * pi_learned = softmax(r_true)    -- learned from exploration data.
       * pi_greedy  = argmax(r_true)     -- the skyline (no exploration).
  4. Data-efficiency sweep: learn pi from random logs of size 1k/10k/100k/full and
     watch true value climb past the naive policy -- a little unbiased data beats a
     lot of biased data.

Usage:
    cd phase9 && python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events, load_random_log
from signals import WEAK
from stats import mean_std
from results_io import save_results

import learning

PHASE_DIR = Path(__file__).parent
SEED = 42
TEMPERATURE = 0.05   # low temperature -> policies concentrate on high-value items
SMOOTHING = 20.0     # Bayesian pseudo-counts: shrink thin items toward the mean


def _reward(df: pl.DataFrame) -> pl.DataFrame:
    """Reward = 1 if the user engaged (MEDIUM or STRONG), else 0."""
    return df.with_columns((pl.col("event_type") != WEAK).cast(pl.Float64).alias("reward"))


def main():
    print("\n=== Phase 9: Off-Policy Learning (Part II) ===\n")

    print("Step 1/4: Load logs...")
    std = _reward(load_events())        # biased production-policy log
    rnd = _reward(load_random_log())    # uniform-random log (truth lives here)

    # Item universe = items with random-log support (where truth is defined).
    items = sorted(rnd["item_id"].unique().to_list())
    print(f"  item universe (random-log support): {len(items):,} items")
    print(f"  biased log: {std.height:,} rows | random log: {rnd.height:,} rows")

    print("\nStep 2/4: Estimate per-item value from each log...")
    r_biased = learning.reward_rate_per_item(
        std["item_id"].to_numpy(), std["reward"].to_numpy(), items, smoothing=SMOOTHING
    )
    r_true = learning.reward_rate_per_item(
        rnd["item_id"].to_numpy(), rnd["reward"].to_numpy(), items, smoothing=SMOOTHING
    )
    print(f"  (Bayesian smoothing={SMOOTHING:.0f} pseudo-counts -- stops 1-view items")
    print(f"   from faking a perfect reward rate and hijacking a greedy policy.)")

    print("\nStep 3/4: Learn policies and grade them by TRUE value...")
    pi_naive = learning.softmax_policy(r_biased, items, TEMPERATURE)
    pi_learned = learning.softmax_policy(r_true, items, TEMPERATURE)
    pi_greedy = learning.greedy_policy(r_true, items)
    pi_uniform = np.full(len(items), 1.0 / len(items))

    v_naive = learning.policy_value(pi_naive, r_true, items)
    v_learned = learning.policy_value(pi_learned, r_true, items)
    v_greedy = learning.policy_value(pi_greedy, r_true, items)
    v_uniform = learning.policy_value(pi_uniform, r_true, items)

    print("\n" + "=" * 62)
    print(f"  {'Policy (how it was learned)':<38}{'TRUE value':>12}{'vs naive':>12}")
    print("-" * 62)
    print(f"  {'uniform random (no learning)':<38}{v_uniform:>12.4f}{'--':>12}")
    print(f"  {'pi_naive  = softmax(biased-log rates)':<38}{v_naive:>12.4f}{'--':>12}")

    def rel(v):
        return f"{(v / v_naive - 1) * 100:+.0f}%" if v_naive else "n/a"

    print(f"  {'pi_learned= softmax(random-log rates)':<38}{v_learned:>12.4f}{rel(v_learned):>12}")
    print(f"  {'pi_greedy = argmax(random-log rates)':<38}{v_greedy:>12.4f}{rel(v_greedy):>12}")
    print("=" * 62)
    print("  The naive policy learned high 'value' for items the PRODUCTION policy")
    print("  favored -- confounded. Learning from the random log instead yields a")
    print("  policy with genuinely higher TRUE value. Same math, unbiased input.")

    print("\nStep 4/4: Data efficiency -- how much unbiased data do we need?")
    # Multi-seed: one sample per size hides sampling noise (especially at n=1k),
    # so we repeat each subsample under several seeds and report mean +/- std.
    SWEEP_SEEDS = (42, 43, 44, 45, 46)
    sweep = []
    for size in (1_000, 10_000, 100_000, rnd.height):
        capped = min(size, rnd.height)
        # The full log has no sampling variance -- one 'seed' suffices there.
        seeds = (SEED,) if capped >= rnd.height else SWEEP_SEEDS
        vals = []
        for s in seeds:
            sub = rnd if size >= rnd.height else rnd.sample(size, seed=s)
            r_sub = learning.reward_rate_per_item(
                sub["item_id"].to_numpy(), sub["reward"].to_numpy(), items, smoothing=SMOOTHING
            )
            pi_sub = learning.softmax_policy(r_sub, items, TEMPERATURE)
            vals.append(learning.policy_value(pi_sub, r_true, items))
        mean_v, std_v = mean_std(vals)
        n_beat = sum(1 for v in vals if v > v_naive)
        sweep.append({
            "n": int(capped),
            "true_value": mean_v,
            "true_value_std": std_v,
            "n_seeds": len(vals),
            "seeds_beating_naive": n_beat,
        })
        pm = f"+/-{std_v:.4f} over {len(vals)} seeds" if len(vals) > 1 else "(full log, no sampling)"
        print(f"  random-log rows={capped:>9,}  ->  true value={mean_v:.4f} {pm}"
              f"  ({n_beat}/{len(vals)} beat naive)")
    print("  ^ Bias does NOT average out: the naive policy was learned from 1.44M")
    print("    biased rows, yet a policy learned from a fraction of that many UNBIASED")
    print("    rows overtakes it -- and keeps pulling ahead. Only unbiased data fixes")
    print("    the confounding; more biased data just entrenches it (Rules 23, 36).")

    results = {
        "v_uniform": v_uniform,
        "v_naive_policy": v_naive,
        "v_learned_policy": v_learned,
        "v_greedy_policy": v_greedy,
        "temperature": TEMPERATURE,
        "n_items": len(items),
        "sweep": sweep,
    }
    save_results(PHASE_DIR, results)

    print("\nNext steps:")
    print("  - In production you won't have a random log -- so run a small,")
    print("    principled exploration bucket (epsilon-greedy / Thompson) to collect")
    print("    unbiased data, then learn on it (or IPS-correct the biased log).")
    print("  - Contextual off-policy learning (per-user) is the natural extension.")
    print("  - This closes Part II: evaluate honestly (Phase 8), THEN learn honestly.\n")
    return results


if __name__ == "__main__":
    main()
