"""
Phase 19 -- Contextual Off-Policy Evaluation & Learning
========================================================
Phases 8 and 9 evaluated and learned a single CONTEXT-FREE policy: one distribution
over items for everybody. Real policies are contextual -- pi(a | x) depends on the
user. This phase generalizes the Part II machinery to that setting and shows two
things:

  1. OPE: the contextual IPS / SNIPS / Direct-Method / Doubly-Robust estimators
     recover a target policy's TRUE value from a log written by a different policy --
     and a context-FREE estimator is biased for a contextual target (it literally
     can't represent "different users, different items").
  2. OPL: learning a contextual policy off-policy (fit a reward model, deploy its
     softmax) beats both the logging policy and a context-free learned policy.

As in Phases 8/15/16 the world is a controlled simulation with a KNOWN true reward
model (so "truth" is computable), but each arm's base rate is a REAL KuaiRand
per-item rate (grounded in data). Pure NumPy core (Rule 5); reuses Phase 15's reward
model and Phase 16's per-arm ridge fitter (DRY).
"""

from __future__ import annotations

import numpy as np

from linucb import reward_prob            # Phase 15: clip(Theta @ x) -> P(reward)
from loop import fit_reward_models        # Phase 16: per-arm ridge (optional IPS weights)


def softmax_policy_matrix(Theta: np.ndarray, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Contextual stochastic policy pi(a | x) = softmax(Theta_a . x / temp), returned as
    an (n_contexts, n_arms) matrix. Lower temperature -> greedier.
    """
    scores = X @ Theta.T / temperature                 # (n, n_arms)
    scores -= scores.max(axis=1, keepdims=True)        # numerical stability
    ex = np.exp(scores)
    return ex / ex.sum(axis=1, keepdims=True)


def true_policy_value(Theta_true: np.ndarray, policy: np.ndarray, X: np.ndarray) -> float:
    """
    HONEST value of a contextual policy: mean over contexts of sum_a pi(a|x) r_true(x,a).
    Only computable because the simulation gives us the true reward model.
    """
    r = np.array([reward_prob(Theta_true, x) for x in X])   # (n, n_arms)
    return float(np.mean(np.sum(policy * r, axis=1)))


# --------------------------------------------------------------- OPE estimators

def contextual_ips(target_p: np.ndarray, logging_p: np.ndarray, rewards: np.ndarray) -> float:
    """IPS: mean( pi_target(a_i|x_i)/pi_log(a_i|x_i) * r_i ). Unbiased, high variance."""
    return float(np.mean(target_p / logging_p * rewards))


def contextual_snips(target_p: np.ndarray, logging_p: np.ndarray, rewards: np.ndarray) -> float:
    """Self-normalized IPS: sum(w r)/sum(w), w = ratio. Lower variance, tiny bias."""
    w = target_p / logging_p
    return float(np.sum(w * rewards) / np.sum(w))


def direct_method(policy_full: np.ndarray, rhat: np.ndarray) -> float:
    """
    Direct Method: mean_x sum_a pi(a|x) rhat(x,a) using a learned reward model rhat
    (n_contexts, n_arms). Zero variance from the log, but biased if rhat is wrong.
    """
    return float(np.mean(np.sum(policy_full * rhat, axis=1)))


def doubly_robust(
    target_p: np.ndarray, logging_p: np.ndarray, rewards: np.ndarray,
    policy_full: np.ndarray, rhat: np.ndarray, arms: np.ndarray,
) -> float:
    """
    Doubly Robust = Direct Method + IPS correction on the model's residual:
        mean_x [ sum_a pi(a|x) rhat(x,a) + (pi(a_i|x_i)/p_i)(r_i - rhat(x_i, a_i)) ]
    Unbiased if EITHER the model OR the propensities are right; low variance.
    """
    n = len(rewards)
    baseline = np.sum(policy_full * rhat, axis=1)                 # (n,)
    rhat_taken = rhat[np.arange(n), arms]                         # rhat at logged arm
    correction = (target_p / logging_p) * (rewards - rhat_taken)
    return float(np.mean(baseline + correction))


def predict_rhat(theta_hat: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Reward-model predictions for every arm at every context -> (n_contexts, n_arms)."""
    return np.clip(X @ theta_hat.T, 0.0, 1.0)


def learn_reward_model(
    X: np.ndarray, arms: np.ndarray, rewards: np.ndarray, logging_p: np.ndarray,
    n_arms: int, dim: int, use_ips: bool = False,
) -> np.ndarray:
    """
    Fit per-arm ridge reward models from the log (Phase 16, reused). use_ips reweights
    rows by 1/propensity. Returns theta_hat (n_arms, dim); feeds DM/DR and OPL.
    """
    weights = 1.0 / logging_p if use_ips else None
    return fit_reward_models(X, arms, rewards, weights, n_arms, dim)
