"""
Phase 8 -- Entry Point: OPE on KuaiRand's biased vs. random logs
================================================================
The experiment that only KuaiRand makes possible:

  1. Derive a target policy pi from the BIASED production log (as any team would).
  2. Ask "how good is pi?" four ways and compare to ground truth.

Ground truth is computable here ONLY because the RANDOM log exists: under uniform
random exposure, per-item reward rates are unconfounded, so
    V_true(pi) = sum_a pi(a) * reward_rate_random(a)
is the honest value of pi. Everything else is an estimate we grade against it.

  * V_naive / DM : plug in reward rates from the BIASED log -> confounded, wrong.
  * IPS / SNIPS  : reweight the RANDOM log by pi/beta (beta known = 1/N) -> unbiased.
  * DR           : biased reward model + propensity correction -> recovers truth.

Usage:
    cd phase8 && python run.py
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
import ope

SOFTMAX_TEMPERATURE = 1.0


def _reward_col() -> pl.Expr:
    """Reward = 1 if the user engaged (MEDIUM or STRONG), else 0."""
    return (pl.col("event_type") != WEAK).cast(pl.Float64).alias("reward")


def _reward_rate_per_item(events: pl.DataFrame) -> pl.DataFrame:
    return (
        events.with_columns(_reward_col())
        .group_by("item_id")
        .agg(pl.col("reward").mean().alias("reward_rate"),
             pl.len().alias("n"))
    )


def build_target_policy(std_events: pl.DataFrame, items: list[str]) -> dict[str, float]:
    """
    A plausible target policy learned from the biased log: softmax over
    log(1 + positive-interaction count) per item. Context-free (pi(a|x)=pi(a)),
    which keeps the demo tractable while staying a valid stochastic policy.
    """
    pos = (
        std_events.with_columns(_reward_col())
        .group_by("item_id")
        .agg(pl.col("reward").sum().alias("pos"))
    )
    pos_map = dict(zip(pos["item_id"], pos["pos"]))
    scores = np.array([np.log1p(pos_map.get(i, 0.0)) for i in items])
    scores = scores / SOFTMAX_TEMPERATURE
    scores -= scores.max()
    ex = np.exp(scores)
    probs = ex / ex.sum()
    return dict(zip(items, probs))


def main():
    print("\n=== Phase 8: Off-Policy Evaluation (Part II) ===\n")

    print("Step 1/4: Load logs...")
    std = load_events()                 # biased production-policy log
    rnd = load_random_log()             # uniform-random log (propensity known)

    # Item universe = items exposed in the random log (where truth is defined).
    items = sorted(rnd["item_id"].unique().to_list())
    n_items = len(items)
    beta = 1.0 / n_items                # known uniform logging propensity
    print(f"  item universe (random-log support): {n_items:,} items; beta = 1/{n_items} = {beta:.2e}")

    print("\nStep 2/4: Build target policy pi from the BIASED log...")
    pi_map = build_target_policy(std, items)

    print("\nStep 3/4: Reward models + ground truth...")
    rhat_true = {r["item_id"]: r["reward_rate"] for r in _reward_rate_per_item(rnd).iter_rows(named=True)}
    rhat_biased = {r["item_id"]: r["reward_rate"] for r in _reward_rate_per_item(std).iter_rows(named=True)}

    pi_arr = np.array([pi_map[i] for i in items])
    v_true = float(np.sum(pi_arr * np.array([rhat_true.get(i, 0.0) for i in items])))
    v_naive = float(np.sum(pi_arr * np.array([rhat_biased.get(i, 0.0) for i in items])))

    print(f"  V_true  (gold: pi x random-log reward)  = {v_true:.5f}")
    print(f"  V_naive (DM: pi x BIASED-log reward)    = {v_naive:.5f}   <- confounded")

    print("\nStep 4/4: OPE estimators on the RANDOM log...")
    # Per-random-row arrays.
    rnd_r = rnd.with_columns(_reward_col())
    pi_df = pl.DataFrame({"item_id": items, "pi": pi_arr,
                          "rhat_b": [rhat_biased.get(i, 0.0) for i in items]})
    joined = rnd_r.join(pi_df, on="item_id", how="left").drop_nulls("pi")

    pi_probs = joined["pi"].to_numpy()
    rewards = joined["reward"].to_numpy()
    rhat_b_action = joined["rhat_b"].to_numpy()
    beta_probs = np.full_like(pi_probs, beta)

    ips_v = ope.ips(pi_probs, beta_probs, rewards)
    snips_v = ope.snips(pi_probs, beta_probs, rewards)
    dr_v = ope.doubly_robust(pi_probs, beta_probs, rewards, rhat_b_action, dm_value=v_naive)
    ess = ope.effective_sample_size(pi_probs, beta_probs)

    def err(v):
        return abs(v - v_true) / v_true * 100 if v_true else float("nan")

    print("\n" + "=" * 62)
    print(f"  {'Estimator':<26}{'Value':>10}{'Rel.err vs truth':>22}")
    print("-" * 62)
    print(f"  {'V_true (ground truth)':<26}{v_true:>10.5f}{'--':>22}")
    print(f"  {'Naive / Direct Method':<26}{v_naive:>10.5f}{err(v_naive):>20.1f}%   <- biased")
    print(f"  {'IPS':<26}{ips_v:>10.5f}{err(ips_v):>20.1f}%")
    print(f"  {'SNIPS':<26}{snips_v:>10.5f}{err(snips_v):>20.1f}%")
    print(f"  {'Doubly Robust':<26}{dr_v:>10.5f}{err(dr_v):>20.1f}%")
    print("=" * 62)
    print(f"  Effective sample size: {ess:,.0f} of {len(pi_probs):,} random-log rows")
    print()
    print("Reading the numbers:")
    print("  - The naive/DM estimate uses the BIASED log's reward rates -> it is")
    print("    confounded (production only showed items it already liked).")
    print("  - IPS/SNIPS reweight the RANDOM log by pi/beta and recover the truth")
    print("    because beta is KNOWN (uniform). This is the payoff of the random log.")
    print("  - Doubly Robust starts from the biased model and corrects it with the")
    print("    known propensities -- unbiased if EITHER piece is right.")
    print("  - ESS warns when pi strays too far from beta for IPS to be trustworthy.")


if __name__ == "__main__":
    main()
