"""
Phase 8 -- Off-Policy Evaluation (Part II: causality-aware)
============================================================
The question every recsys team eventually faces: "we have a new ranking policy --
how good is it, WITHOUT shipping it to users?" Answering that from *logged* data is
off-policy evaluation (OPE), and it is a causal-inference problem, because the logs
were collected by a DIFFERENT policy (the one already in production).

Naive offline metrics computed on biased logs are confounded: the production policy
only ever showed items it already favored, in contexts it favored, so item reward
rates estimated from those logs are wrong for a policy that would show different
things. KuaiRand is special because it also has a UNIFORM-RANDOM log: there, every
item had a known, equal probability of being shown (propensity = 1/N). That known
propensity is what makes the estimators below unbiased.

This module is the pure numeric core (arrays in, numbers out) so it is unit-tested
independently of the data (Rule 5). `run.py` wires it to KuaiRand.

Estimators (target policy pi, logging policy beta, reward r):
  IPS    inverse propensity scoring:   mean( (pi/beta) * r )        -- unbiased, high variance
  SNIPS  self-normalized IPS:          sum(w*r) / sum(w)            -- lower variance, tiny bias
  DM     direct method:               sum_a pi(a) * rhat(a)         -- low variance, model-biased
  DR     doubly robust:               DM + mean( (pi/beta)*(r-rhat))-- unbiased if EITHER is right
"""

from __future__ import annotations

import numpy as np


def importance_weights(pi_probs: np.ndarray, beta_probs: np.ndarray) -> np.ndarray:
    """pi(a|x) / beta(a|x) per logged sample. beta must be > 0 (guaranteed by a
    random logging policy -- this is why unbiased OPE needs full support)."""
    pi_probs = np.asarray(pi_probs, dtype=np.float64)
    beta_probs = np.asarray(beta_probs, dtype=np.float64)
    if np.any(beta_probs <= 0):
        raise ValueError("beta_probs must be strictly positive (need full support).")
    return pi_probs / beta_probs


def ips(pi_probs: np.ndarray, beta_probs: np.ndarray, rewards: np.ndarray) -> float:
    """Inverse propensity scoring: unbiased estimate of the target policy's value."""
    w = importance_weights(pi_probs, beta_probs)
    return float(np.mean(w * np.asarray(rewards, dtype=np.float64)))


def snips(pi_probs: np.ndarray, beta_probs: np.ndarray, rewards: np.ndarray) -> float:
    """Self-normalized IPS: divides by the sum of weights -> far lower variance."""
    w = importance_weights(pi_probs, beta_probs)
    denom = np.sum(w)
    if denom == 0:
        return 0.0
    return float(np.sum(w * np.asarray(rewards, dtype=np.float64)) / denom)


def direct_method(pi_dist: np.ndarray, reward_hat: np.ndarray) -> float:
    """
    Value under a learned reward model rhat(a), averaged over the target policy's
    action distribution: sum_a pi(a) * rhat(a). No propensities needed, but only
    as good as the reward model.
    """
    pi_dist = np.asarray(pi_dist, dtype=np.float64)
    reward_hat = np.asarray(reward_hat, dtype=np.float64)
    return float(np.sum(pi_dist * reward_hat))


def doubly_robust(
    pi_probs: np.ndarray,
    beta_probs: np.ndarray,
    rewards: np.ndarray,
    reward_hat_action: np.ndarray,
    dm_value: float,
) -> float:
    """
    Doubly robust = direct method + IPS-corrected residual. Unbiased if EITHER the
    propensities OR the reward model is correct -- the best of both worlds.
    `reward_hat_action` is the model's predicted reward for each logged action.
    """
    w = importance_weights(pi_probs, beta_probs)
    residual = np.asarray(rewards, dtype=np.float64) - np.asarray(reward_hat_action, dtype=np.float64)
    return float(dm_value + np.mean(w * residual))


def effective_sample_size(pi_probs: np.ndarray, beta_probs: np.ndarray) -> float:
    """
    ESS = (sum w)^2 / sum(w^2). When importance weights are wildly uneven (target
    policy far from logging policy), ESS collapses and IPS becomes noise -- the
    number that tells you whether to trust the estimate.
    """
    w = importance_weights(pi_probs, beta_probs)
    s2 = np.sum(np.square(w))
    if s2 == 0:
        return 0.0
    return float(np.square(np.sum(w)) / s2)
