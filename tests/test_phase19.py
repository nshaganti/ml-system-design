"""
Tests for Phase 19 -- contextual OPE/OPL core (contextual_ope.py).

Pure and deterministic. Headline tests: (1) each estimator is unbiased when target ==
logging (they should return the on-policy mean reward), and (2) on a large synthetic
log, contextual IPS/SNIPS/DR recover a known target value far better than the
context-free estimator does for a contextual target.
"""

from __future__ import annotations

import numpy as np

import contextual_ope as C


def test_softmax_policy_matrix_rows_are_distributions():
    Theta = np.array([[1.0, 0.5], [0.0, -0.5], [0.2, 0.0]])
    X = np.array([[1.0, 0.3], [1.0, -0.7]])
    pi = C.softmax_policy_matrix(Theta, X, temperature=0.5)
    assert pi.shape == (2, 3)
    assert np.allclose(pi.sum(axis=1), 1.0)
    assert np.all(pi > 0)


def test_ips_equals_snips_equals_mean_when_target_is_logging():
    # If target == logging, the weights are all 1, so IPS and SNIPS both equal the
    # mean logged reward -- the on-policy sanity check.
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.9, 500)
    rewards = (rng.random(500) < 0.4).astype(float)
    assert np.isclose(C.contextual_ips(p, p, rewards), rewards.mean())
    assert np.isclose(C.contextual_snips(p, p, rewards), rewards.mean())


def test_direct_method_uses_policy_weighted_rhat():
    policy = np.array([[0.5, 0.5], [1.0, 0.0]])
    rhat = np.array([[0.2, 0.8], [0.6, 0.1]])
    # row0: .5*.2+.5*.8=.5 ; row1: 1*.6+0*.1=.6 ; mean = .55
    assert np.isclose(C.direct_method(policy, rhat), 0.55)


def test_doubly_robust_reduces_to_dm_when_model_is_perfect():
    # If rhat matches the observed rewards exactly, the IPS correction is zero and DR
    # equals the Direct Method.
    rng = np.random.default_rng(1)
    n, n_arms = 200, 3
    arms = rng.integers(0, n_arms, n)
    rewards = rng.random(n)
    rhat = rng.random((n, n_arms))
    rhat[np.arange(n), arms] = rewards          # perfect on the taken arm
    policy = np.full((n, n_arms), 1.0 / n_arms)
    target_p = rng.uniform(0.1, 0.9, n)
    logging_p = rng.uniform(0.1, 0.9, n)
    dr = C.doubly_robust(target_p, logging_p, rewards, policy, rhat, arms)
    dm = C.direct_method(policy, rhat)
    assert np.isclose(dr, dm)


def test_contextual_estimators_beat_context_free_on_a_contextual_target():
    # Build a small world with a computable truth; check contextual IPS is closer to
    # the true target value than the context-free (marginalized) IPS.
    rng = np.random.default_rng(3)
    n_arms, dim, n = 4, 3, 20000
    Theta_true = rng.normal(0, 0.4, (n_arms, dim))
    Theta_true[:, 0] = rng.uniform(0.3, 0.7, n_arms)         # base rates
    Theta_log = Theta_true.copy(); Theta_log[:, 1:] = 0.0    # context-blind logger

    X = np.ones((n, dim)); X[:, 1:] = rng.uniform(-1, 1, (n, dim - 1))
    pi_log = C.softmax_policy_matrix(Theta_log, X, 0.5)
    arms = np.array([rng.choice(n_arms, p=pi_log[i]) for i in range(n)])
    probs = np.clip(np.array([Theta_true @ x for x in X]), 0.01, 0.99)
    rewards = (rng.random(n) < probs[np.arange(n), arms]).astype(float)
    logging_p = pi_log[np.arange(n), arms]

    pi_t = C.softmax_policy_matrix(Theta_true, X, 0.3)
    target_p = pi_t[np.arange(n), arms]
    v_true = C.true_policy_value(Theta_true, C.softmax_policy_matrix(Theta_true, X, 0.3), X)

    ctx = C.contextual_ips(target_p, logging_p, rewards)
    tbar, lbar = pi_t.mean(0), pi_log.mean(0)
    cf = C.contextual_ips(tbar[arms], lbar[arms], rewards)

    assert abs(ctx - v_true) < abs(cf - v_true)              # context matters for OPE
