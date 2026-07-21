"""
Phase 17 -- Real Context + Safety-Gated Redeploys
==================================================
Phase 16 closed the loop, but with two shortcuts a production system can't take:
  1. the context was synthetic uniform noise, and
  2. every learned candidate was redeployed blindly.

Phase 17 fixes both -- the production-real finale:
  * REAL context: the loop acts on standardized per-user features measured from
    KuaiRand-Pure (activity level + signal-mix), so the context distribution is
    correlated and non-uniform, like real traffic (built in run.py).
  * SAFETY GATE: before a learned candidate replaces the deployed policy, estimate
    its value OFF-POLICY (Phase 8 IPS) on a fresh log and only redeploy if it beats
    the incumbent. A bad iteration is REJECTED, not shipped -- Rule 16 with a
    parachute.
  * MIN-PROPENSITY FLOOR: clip logging propensities before inverting them, bounding
    importance weights so one rare action can't blow up the OPE estimate's variance
    (Phase 11's bias-variance tradeoff, made an explicit safety knob).

Pure NumPy core (Rule 5). Reuses Phase 16's loop primitives -- no re-implementation.
"""

from __future__ import annotations

import numpy as np

from loop import (   # Phase 16
    epsilon_greedy_propensities, fit_reward_models, greedy_value,
    skyline_value, uniform_value,
)
from linucb import reward_prob   # Phase 15


def ips_policy_value(
    theta_candidate: np.ndarray, contexts: np.ndarray, arms: np.ndarray,
    rewards: np.ndarray, propensities: np.ndarray, min_prop: float,
) -> float:
    """
    OPE (Phase 8 IPS): estimate the value of the GREEDY policy induced by
    `theta_candidate` from a log, WITHOUT deploying it.

    The target policy is deterministic greedy, so its probability on a logged row is
    1 if it would have chosen the logged arm, else 0. IPS then reduces to averaging
    reward / logging_propensity over the rows where the target agrees with the log:

        V_hat = (1/n) * sum_i  1[argmax(theta.x_i) == a_i] * r_i / p_i

    `min_prop` clips p_i from below so importance weights stay bounded (variance
    guard). This is the number the safety gate trusts instead of peeking at truth.
    """
    n = len(arms)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        if int(np.argmax(theta_candidate @ contexts[i])) == int(arms[i]):
            total += rewards[i] / max(propensities[i], min_prop)
    return total / n


def _log_batch(deployed, true_theta, ctx_pool, batch_size, epsilon, rng):
    """Deploy `deployed` under an epsilon-soft policy; return (ctx, arm, reward, prop)."""
    idx = rng.integers(0, len(ctx_pool), size=batch_size)
    ctx = ctx_pool[idx]
    arms = np.empty(batch_size, dtype=np.int64)
    rewards = np.empty(batch_size)
    props = np.empty(batch_size)
    for i in range(batch_size):
        p = epsilon_greedy_propensities(deployed @ ctx[i], epsilon)
        a = int(rng.choice(len(p), p=p))
        arms[i] = a
        props[i] = p[a]
        rewards[i] = 1.0 if rng.random() < reward_prob(true_theta, ctx[i])[a] else 0.0
    return ctx, arms, rewards, props


def _uniform_validation(true_theta, ctx_pool, size, rng):
    """
    A small UNIFORM-RANDOM exploration bucket with clean rewards -- the honest data to
    validate a candidate on. Uniform logging gives every arm propensity 1/n, so IPS is
    unbiased for ANY target policy and no incumbent gets a home-field advantage. This
    is Phase 8's random log, used live as a redeploy gate.
    """
    n_arms = true_theta.shape[0]
    idx = rng.integers(0, len(ctx_pool), size=size)
    ctx = ctx_pool[idx]
    arms = rng.integers(0, n_arms, size=size)
    rewards = np.array([1.0 if rng.random() < reward_prob(true_theta, ctx[i])[arms[i]]
                        else 0.0 for i in range(size)])
    props = np.full(size, 1.0 / n_arms)
    return ctx, arms, rewards, props


def simulate_safe_loop(
    true_theta: np.ndarray,
    ctx_pool: np.ndarray,
    eval_contexts: np.ndarray,
    rng: np.random.Generator,
    n_iterations: int,
    batch_size: int,
    epsilon: float,
    use_gate: bool,
    min_prop: float = 0.02,
    window: int = 4,
    val_size: int = 400,
    poison_iter: int | None = None,
) -> dict:
    """
    Explore -> learn (on recent data) -> [gate on a clean uniform bucket] -> redeploy.

    Realistic online retraining: the candidate is fit on the last `window` batches
    (recent data), so a corrupt recent batch genuinely craters it. The gate scores the
    candidate and the incumbent by IPS on a fresh UNIFORM-RANDOM validation bucket
    (clean rewards) and only redeploys the candidate if it wins there.

    poison_iter: that iteration's TRAINING batch has its rewards flipped (1-r), a
    logging/label bug. Returns TRUE deployed value per iteration and blocked count.
    """
    n_arms, dim = true_theta.shape
    ctx_hist, arm_hist, rew_hist, prop_hist = [], [], [], []
    deployed = np.zeros((n_arms, dim))
    values, blocked = [], 0

    for t in range(n_iterations):
        ctx, arms, rewards, props = _log_batch(
            deployed, true_theta, ctx_pool, batch_size, epsilon, rng)
        if t == poison_iter:
            rewards = 1.0 - rewards          # corrupt this batch (bug simulation)
        ctx_hist.append(ctx); arm_hist.append(arms)
        rew_hist.append(rewards); prop_hist.append(props)

        # Fit the candidate on the recent window only (online retraining).
        w = slice(-window, None)
        weights = 1.0 / np.maximum(np.concatenate(prop_hist[w]), min_prop)
        candidate = fit_reward_models(
            np.vstack(ctx_hist[w]), np.concatenate(arm_hist[w]),
            np.concatenate(rew_hist[w]), weights, n_arms, dim)

        if use_gate:
            vctx, varm, vrew, vprop = _uniform_validation(true_theta, ctx_pool, val_size, rng)
            v_cand = ips_policy_value(candidate, vctx, varm, vrew, vprop, min_prop)
            v_dep = ips_policy_value(deployed, vctx, varm, vrew, vprop, min_prop)
            if v_cand >= v_dep:
                deployed = candidate
            else:
                blocked += 1        # rejected: keep the incumbent, ship nothing bad
        else:
            deployed = candidate

        values.append(greedy_value(deployed, eval_contexts, true_theta))

    return {"values": values, "blocked": blocked}
