"""
Phase 9 -- Off-Policy LEARNING (Part II, the sequel to Phase 8)
===============================================================
Phase 8 showed that *evaluating* a policy on biased logs is a factor-of-two lie.
But there is a deeper consequence: the policy in Part I was also *learned* from
those biased logs -- so it isn't just mis-measured, it is genuinely suboptimal.
Learning from confounded data bakes the confounding into the model itself.

This module is the pure, testable numeric core (Rule 5). It answers: given per-item
value estimates, what policy should we deploy, and what is that policy's TRUE value?

Key idea (context-free, one stochastic policy over the catalog):
  * A policy pi(a) is a probability distribution over items.
  * Its TRUE value is V(pi) = sum_a pi(a) * r_true(a), where r_true comes from the
    uniform-random log (unconfounded -- the only place truth is defined).
  * The entropy-regularized optimal policy given value estimates v(a) is
    softmax(v / temperature): mass flows to high-value items, temperature controls
    how greedily.

The lesson lives in *which value estimate you learn from*:
  * r_biased(a) from the production log  -> confounded -> a worse policy.
  * r_random(a) from the exploration log -> unbiased  -> a better policy,
    even when the exploration log is far smaller (see the data-efficiency sweep
    in run.py). A little unbiased data beats a lot of biased data.
"""

from __future__ import annotations

import numpy as np


def reward_rate_per_item(
    item_ids: np.ndarray,
    rewards: np.ndarray,
    items: list[str],
    smoothing: float = 0.0,
    prior: float | None = None,
) -> dict[str, float]:
    """
    Empirical mean reward per item over a log, with optional Bayesian smoothing.

        rate(a) = (sum_a + smoothing * prior) / (count_a + smoothing)

    Why smoothing matters (a real Phase 9 lesson): without it, an item shown ONCE
    that happened to get engagement has an estimated rate of 1.0 -- so a greedy
    policy chases noise. `smoothing` adds `smoothing` pseudo-observations at the
    global `prior` rate, shrinking thinly-sampled items toward the mean until they
    earn a confident estimate. smoothing=0 reproduces the raw empirical mean.

    Items never seen in the log fall back to the prior (or 0.0 if no smoothing).
    """
    item_ids = np.asarray(item_ids)
    rewards = np.asarray(rewards, dtype=np.float64)
    if prior is None:
        prior = float(rewards.mean()) if rewards.size else 0.0
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for a, r in zip(item_ids, rewards):
        sums[a] = sums.get(a, 0.0) + r
        counts[a] = counts.get(a, 0) + 1

    out: dict[str, float] = {}
    for it in items:
        c = counts.get(it, 0)
        s = sums.get(it, 0.0)
        if c == 0 and smoothing == 0.0:
            out[it] = 0.0
        else:
            out[it] = (s + smoothing * prior) / (c + smoothing)
    return out


def softmax_policy(
    value_by_item: dict[str, float], items: list[str], temperature: float = 0.1
) -> np.ndarray:
    """
    Entropy-regularized optimal stochastic policy given value estimates:
    pi(a) = softmax(v(a) / temperature). Lower temperature -> greedier. Returned
    as an array aligned to `items`.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use greedy_policy for argmax).")
    v = np.array([value_by_item.get(it, 0.0) for it in items], dtype=np.float64)
    z = v / temperature
    z -= z.max()  # numerical stability
    ex = np.exp(z)
    return ex / ex.sum()


def greedy_policy(value_by_item: dict[str, float], items: list[str]) -> np.ndarray:
    """All mass on the highest-value item -- the skyline (zero exploration)."""
    v = np.array([value_by_item.get(it, 0.0) for it in items], dtype=np.float64)
    pi = np.zeros_like(v)
    pi[int(np.argmax(v))] = 1.0
    return pi


def policy_value(pi: np.ndarray, true_rate_by_item: dict[str, float], items: list[str]) -> float:
    """
    The HONEST value of a policy: V(pi) = sum_a pi(a) * r_true(a). Only computable
    here because the random log gives us unconfounded r_true. This is the number
    every learned policy is ultimately graded on.
    """
    r = np.array([true_rate_by_item.get(it, 0.0) for it in items], dtype=np.float64)
    return float(np.sum(np.asarray(pi, dtype=np.float64) * r))
