"""
Phase 19 -- Entry Point: Contextual OPE & OPL
==============================================
Two experiments on a controlled contextual world (real KuaiRand base rates + a known
true reward model, so "truth" is computable):

  A. OPE -- estimate a good CONTEXTUAL target policy's true value from a log written
     by a context-BLIND logging policy. Compare five estimators against the truth:
     context-free IPS (the Phase 8 estimator, blind to x), contextual IPS, SNIPS,
     Direct Method, and Doubly Robust.

  B. OPL -- learn a policy off-policy from that log and grade its TRUE value: a
     contextual learned policy vs a context-free learned policy vs the logging policy.

Headline: ignoring context makes OPE biased for a contextual target, and a contextual
learned policy is the only one that recovers the personalization the logger threw away.

Usage:
    cd phase19 && python run.py
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
from signals import POSITIVE_SIGNALS
from stats import mean_std
from results_io import save_results
import contextual_ope as C

PHASE_DIR = Path(__file__).parent

SEED = 42
N_ARMS = 12
N_FEATURES = 4
DIM = N_FEATURES + 1
N_LOG = 8_000
N_EVAL = 3_000
N_WORLDS = 20
CTX_WEIGHT = 0.25
TEMP_TARGET = 0.3      # the (good) contextual policy we evaluate / aim for
TEMP_LOG = 0.5         # the context-blind logging policy (soft -> keeps support)


def build_base_rates(n_arms: int) -> np.ndarray:
    events = load_events()
    per_item = (
        events.group_by("item_id")
        .agg(total=pl.len(),
             positives=pl.col("event_type").is_in(list(POSITIVE_SIGNALS)).sum())
        .sort("total", descending=True).head(n_arms)
    )
    rates = (per_item["positives"] / per_item["total"]).to_numpy().astype(np.float64)
    return np.clip(rates, 0.05, 0.95)


def _sample_contexts(n, rng):
    X = np.ones((n, DIM))
    X[:, 1:] = rng.uniform(-1.0, 1.0, (n, N_FEATURES))
    return X


def main():
    print("\n=== Phase 19: Contextual Off-Policy Evaluation & Learning ===\n")

    print("Step 1/3: Build the contextual world (real KuaiRand base rates)...")
    base_rates = build_base_rates(N_ARMS)

    # accumulators
    ope_est = {k: [] for k in ("cf_ips", "ips", "snips", "dm", "dr")}
    ope_true = []
    opl = {k: [] for k in ("logging", "context_free", "contextual", "skyline", "target")}

    print(f"\nStep 2/3: {N_WORLDS} worlds -- log {N_LOG:,} rows from a context-BLIND "
          f"logger, evaluate + learn...")
    for w in range(N_WORLDS):
        rng = np.random.default_rng(SEED + w)
        Theta_true = np.zeros((N_ARMS, DIM))
        Theta_true[:, 0] = base_rates
        Theta_true[:, 1:] = rng.normal(0.0, CTX_WEIGHT, (N_ARMS, N_FEATURES))
        # Context-blind logger: correct base rates, ZERO context weights (old system
        # that ignored personalization).
        Theta_log = Theta_true.copy()
        Theta_log[:, 1:] = 0.0

        # --- generate the log ---
        Xlog = _sample_contexts(N_LOG, rng)
        pi_log = C.softmax_policy_matrix(Theta_log, Xlog, TEMP_LOG)      # (N_LOG, n_arms)
        arms = np.array([rng.choice(N_ARMS, p=pi_log[i]) for i in range(N_LOG)])
        probs = np.array([C.reward_prob(Theta_true, Xlog[i]) for i in range(N_LOG)])
        rewards = (rng.random(N_LOG) < probs[np.arange(N_LOG), arms]).astype(float)
        logging_p = pi_log[np.arange(N_LOG), arms]

        # --- target policy + its TRUE value ---
        Xeval = _sample_contexts(N_EVAL, rng)
        pi_target_eval = C.softmax_policy_matrix(Theta_true, Xeval, TEMP_TARGET)
        v_true = C.true_policy_value(Theta_true, pi_target_eval, Xeval)
        ope_true.append(v_true)

        # --- OPE: five estimators on the log ---
        pi_target_log = C.softmax_policy_matrix(Theta_true, Xlog, TEMP_TARGET)
        target_p = pi_target_log[np.arange(N_LOG), arms]
        theta_hat = C.learn_reward_model(Xlog, arms, rewards, logging_p, N_ARMS, DIM, use_ips=True)
        rhat = C.predict_rhat(theta_hat, Xlog)

        # context-free IPS: replace pi(a|x) with marginals pbar(a) (ignore x)
        tbar = pi_target_log.mean(axis=0)          # marginal target action probs
        lbar = pi_log.mean(axis=0)                 # marginal logging action probs
        ope_est["cf_ips"].append(C.contextual_ips(tbar[arms], lbar[arms], rewards))
        ope_est["ips"].append(C.contextual_ips(target_p, logging_p, rewards))
        ope_est["snips"].append(C.contextual_snips(target_p, logging_p, rewards))
        ope_est["dm"].append(C.direct_method(pi_target_log, rhat))
        ope_est["dr"].append(C.doubly_robust(target_p, logging_p, rewards,
                                             pi_target_log, rhat, arms))

        # --- OPL: learn a policy, grade TRUE value ---
        rhat_eval = C.predict_rhat(theta_hat, Xeval)
        ctx_policy = C.softmax_policy_matrix(theta_hat, Xeval, TEMP_TARGET)
        opl["contextual"].append(C.true_policy_value(Theta_true, ctx_policy, Xeval))
        # context-free learned policy: marginal reward per arm -> one distribution
        cf_scores = np.array([rewards[arms == a].mean() if (arms == a).any() else 0.0
                              for a in range(N_ARMS)])
        cf_pi_row = np.exp(cf_scores / TEMP_TARGET); cf_pi_row /= cf_pi_row.sum()
        cf_policy = np.tile(cf_pi_row, (N_EVAL, 1))
        opl["context_free"].append(C.true_policy_value(Theta_true, cf_policy, Xeval))
        opl["logging"].append(C.true_policy_value(
            Theta_true, C.softmax_policy_matrix(Theta_log, Xeval, TEMP_LOG), Xeval))
        onehot = np.zeros((N_EVAL, N_ARMS))
        onehot[np.arange(N_EVAL), np.argmax([C.reward_prob(Theta_true, x) for x in Xeval], axis=1)] = 1.0
        opl["skyline"].append(C.true_policy_value(Theta_true, onehot, Xeval))
        opl["target"].append(v_true)

    v_true_mean = float(np.mean(ope_true))
    n_w = len(ope_true)

    def ci95(values):
        """Mean and 95% CI (mean +/- 1.96*SEM) across the N_WORLDS replicates."""
        m, s = mean_std(values)
        sem = s / np.sqrt(len(values)) if len(values) else 0.0
        return m, m - 1.96 * sem, m + 1.96 * sem

    print("\nStep 3/3: Results (means over worlds, 95% CI over the", n_w, "worlds)...\n")
    print("  A. OPE -- estimating the contextual target's TRUE value "
          f"({v_true_mean:.4f}):")
    print("  " + "=" * 74)
    print(f"    {'Estimator':<26}{'Estimate':>11}{'95% CI':>24}{'|err|':>9}")
    print("  " + "-" * 74)
    labels = [("cf_ips", "Context-free IPS (P8)"), ("ips", "Contextual IPS"),
              ("snips", "Contextual SNIPS"), ("dm", "Direct Method"),
              ("dr", "Doubly Robust")]
    results = {"ope_true": v_true_mean, "n_worlds": n_w, "ope": {}, "opl": {}}
    for key, label in labels:
        est, lo, hi = ci95(ope_est[key])
        err = abs(est - v_true_mean) / v_true_mean * 100
        covers = "" if lo <= v_true_mean <= hi else "  <- CI misses truth"
        results["ope"][key] = {"estimate": est, "abs_error_pct": err, "ci95": [lo, hi]}
        print(f"    {label:<26}{est:>11.4f}   [{lo:.4f}, {hi:.4f}]{err:>8.1f}%{covers}")
    print("  " + "=" * 74)
    print("  " + "=" * 60)

    print("\n  B. OPL -- TRUE value of the learned policy:")
    print("  " + "=" * 66)
    print(f"    {'Policy':<28}{'True value':>13}{'95% CI':>24}")
    print("  " + "-" * 66)
    for key, label in [("logging", "Logging (context-blind)"),
                       ("context_free", "Learned, context-free"),
                       ("contextual", "Learned, CONTEXTUAL"),
                       ("target", "Target (softmax of truth)"),
                       ("skyline", "Skyline (oracle)")]:
        v, lo, hi = ci95(opl[key])
        results["opl"][key] = {"value": v, "ci95": [lo, hi]}
        print(f"    {label:<28}{v:>13.4f}   [{lo:.4f}, {hi:.4f}]")
    print("  " + "=" * 66)

    cf = results["ope"]["cf_ips"]["abs_error_pct"]
    dr = results["ope"]["dr"]["abs_error_pct"]
    lift = (results["opl"]["contextual"]["value"] / results["opl"]["context_free"]["value"] - 1) * 100
    print(f"\n  Context-free IPS is off by {cf:.0f}% (it can't see the target is "
          f"contextual); Doubly Robust {dr:.1f}%.")
    print(f"  The contextual learned policy beats the context-free one by {lift:+.0f}% "
          f"true value -- it recovers the personalization the logger discarded.")
    print("\n  Reading the numbers:")
    print("  - Contextual IPS/SNIPS/DR recover the truth; the context-free estimator")
    print("    is structurally biased for a per-user target (Phase 8 generalized).")
    print("  - Direct Method is low-variance but leans on the reward model; DR adds an")
    print("    IPS residual correction and is the safest -- unbiased if EITHER the")
    print("    model or the propensities are right.")
    print("  - OPL: learning WITH context turns a context-blind log into a")
    print("    personalized policy -- the whole point of going contextual.")

    save_results(PHASE_DIR, results)

    print("\nNext steps:")
    print("  - Plug the two-tower user vector (Phase 1) in as the context x.")
    print("  - Use DR as the redeploy gate in Phase 17 to cut its variance.")
    print("  - 95% CIs (across worlds) now sit beside every estimate above.\n")
    return results


if __name__ == "__main__":
    main()
