"""
Tests for Phase 16 -- the closed-loop core (loop.py).

Pure and deterministic. The headline test proves the capstone thesis: a loop that
explores and IPS-corrects its logs reaches a higher TRUE deployed value than a
no-exploration loop that learns once and gets stuck (the feedback trap).
"""

from __future__ import annotations

import numpy as np

import loop as LP


def test_epsilon_greedy_propensities_sum_to_one_and_keep_support():
    scores = np.array([0.1, 0.9, 0.3])
    p = LP.epsilon_greedy_propensities(scores, epsilon=0.2)
    assert np.isclose(p.sum(), 1.0)
    assert np.all(p > 0)                      # every arm loggable
    assert np.argmax(p) == 1                  # greedy arm gets the extra mass
    assert np.isclose(p[1], 0.2 / 3 + 0.8)


def test_epsilon_zero_is_deterministic_greedy():
    p = LP.epsilon_greedy_propensities(np.array([0.2, 0.5, 0.1]), epsilon=0.0)
    assert p[1] == 1.0 and p[0] == 0.0 and p[2] == 0.0


def test_fit_reward_models_recovers_linear_reward():
    # Reward = theta.x with theta_0=[0.5,0.4]; a single arm, noiseless -> ridge
    # should land close to the true weights given enough data.
    rng = np.random.default_rng(0)
    X = np.ones((500, 2))
    X[:, 1] = rng.uniform(-1, 1, 500)
    true = np.array([0.5, 0.4])
    y = X @ true
    theta = LP.fit_reward_models(X, np.zeros(500, dtype=int), y, None, n_arms=1, dim=2, l2=1e-3)
    assert np.allclose(theta[0], true, atol=0.05)


def test_unplayed_arms_keep_prior_zero():
    X = np.ones((10, 2))
    theta = LP.fit_reward_models(X, np.zeros(10, dtype=int), np.ones(10), None, n_arms=3, dim=2)
    assert np.allclose(theta[1], 0.0) and np.allclose(theta[2], 0.0)   # never played


def test_skyline_is_at_least_uniform():
    true_theta = np.array([[0.6, 0.3], [0.4, -0.3], [0.5, 0.0]])
    ctx = np.ones((200, 2))
    ctx[:, 1] = np.linspace(-1, 1, 200)
    assert LP.skyline_value(true_theta, ctx) >= LP.uniform_value(true_theta, ctx)


def test_closed_loop_beats_the_no_exploration_trap():
    # Best arm flips with the sign of feature x1 -> a greedy-only loop that starts
    # committed cannot discover the other arm; exploration + IPS can.
    true_theta = np.array([[0.5, 0.45], [0.5, -0.45], [0.4, 0.0]])
    rng_eval = np.random.default_rng(3)
    eval_ctx = np.ones((1500, 2))
    eval_ctx[:, 1] = rng_eval.uniform(-1, 1, 1500)

    def run(epsilon, use_ips):
        r = np.random.default_rng(11)
        return LP.simulate_loop(true_theta, eval_ctx, r, n_iterations=8,
                                batch_size=600, epsilon=epsilon, use_ips=use_ips)

    trap = run(0.0, True)          # no exploration
    closed = run(0.2, True)        # explore + IPS

    assert closed[-1] > trap[-1]                          # loop escapes the trap
    assert closed[-1] >= closed[0] - 1e-9                 # and improves (or holds)


def test_simulate_loop_returns_value_per_iteration():
    true_theta = np.array([[0.6, 0.1], [0.4, -0.1]])
    eval_ctx = np.ones((100, 2))
    eval_ctx[:, 1] = np.linspace(-1, 1, 100)
    vals = LP.simulate_loop(true_theta, eval_ctx, np.random.default_rng(0),
                            n_iterations=5, batch_size=200, epsilon=0.1)
    assert len(vals) == 5
    assert all(0.0 <= v <= 1.0 for v in vals)
