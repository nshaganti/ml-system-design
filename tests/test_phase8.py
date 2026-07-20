"""
Phase 8 tests: the off-policy-evaluation estimators.

These lock in the mathematical guarantees that make OPE trustworthy:
  - IPS is unbiased when the logging propensities are known
  - SNIPS matches IPS on a constant reward and is lower-variance
  - the direct method is exactly pi . rhat
  - doubly robust equals the direct method when the reward model is perfect
  - ESS collapses when target and logging policies diverge
"""

from __future__ import annotations

import numpy as np

import ope


def test_ips_unbiased_recovers_known_value():
    # Construct a tiny world: 3 actions, known reward per action, uniform logging.
    rng = np.random.default_rng(0)
    n_actions = 3
    true_reward = np.array([0.1, 0.5, 0.9])
    pi = np.array([0.2, 0.3, 0.5])                     # target policy
    beta = np.array([1 / 3, 1 / 3, 1 / 3])            # uniform logging

    # Simulate a large random log drawn from beta; reward ~ Bernoulli(true_reward[a]).
    n = 200_000
    actions = rng.choice(n_actions, size=n, p=beta)
    rewards = (rng.random(n) < true_reward[actions]).astype(float)

    pi_probs = pi[actions]
    beta_probs = beta[actions]
    est = ope.ips(pi_probs, beta_probs, rewards)
    v_true = float(np.sum(pi * true_reward))
    assert abs(est - v_true) < 0.01   # unbiased -> close for large n


def test_snips_matches_ips_on_constant_reward():
    pi_probs = np.array([0.2, 0.4, 0.6, 0.8])
    beta_probs = np.array([0.25, 0.25, 0.25, 0.25])
    rewards = np.ones(4)  # constant reward -> both estimate ~1
    assert abs(ope.snips(pi_probs, beta_probs, rewards) - 1.0) < 1e-9


def test_direct_method_is_dot_product():
    pi = np.array([0.5, 0.3, 0.2])
    rhat = np.array([0.1, 0.2, 0.7])
    assert abs(ope.direct_method(pi, rhat) - (0.05 + 0.06 + 0.14)) < 1e-12


def test_doubly_robust_equals_dm_when_model_perfect():
    # If rhat == reward for every logged action, the correction term is zero.
    pi_probs = np.array([0.2, 0.3, 0.5, 0.4])
    beta_probs = np.array([0.25, 0.25, 0.25, 0.25])
    rewards = np.array([1.0, 0.0, 1.0, 0.0])
    rhat_action = rewards.copy()
    dm_value = 0.42
    assert abs(ope.doubly_robust(pi_probs, beta_probs, rewards, rhat_action, dm_value) - dm_value) < 1e-12


def test_importance_weights_require_support():
    try:
        ope.importance_weights(np.array([0.5]), np.array([0.0]))
        assert False, "expected ValueError on zero propensity"
    except ValueError:
        pass


def test_ess_full_when_policies_match():
    # pi == beta -> weights all 1 -> ESS == n.
    pi = np.full(1000, 0.001)
    beta = np.full(1000, 0.001)
    assert abs(ope.effective_sample_size(pi, beta) - 1000) < 1e-6


def test_ess_collapses_when_policies_diverge():
    # One sample carries almost all the weight -> ESS near 1.
    pi = np.array([0.999, 0.0005, 0.0005])
    beta = np.array([0.001, 0.4995, 0.4995])
    ess = ope.effective_sample_size(pi, beta)
    assert ess < 1.5
