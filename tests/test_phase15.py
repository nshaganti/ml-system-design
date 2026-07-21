"""
Tests for Phase 15 -- contextual bandit core (linucb.py).

Pure and deterministic. The headline test proves the point of the phase: on a world
where the best arm DEPENDS on context, LinUCB beats a context-free Thompson sampler
that can only learn each arm's average rate.
"""

from __future__ import annotations

import numpy as np

import linucb as L


def test_reward_prob_clips_into_bernoulli_range():
    theta = np.array([[2.0, 0.0], [-2.0, 0.0]])   # would give 2.0 and -2.0
    x = np.array([1.0, 0.0])
    p = L.reward_prob(theta, x)
    assert p[0] == 0.99 and p[1] == 0.01          # clipped to valid probabilities


def test_ridge_theta_solves_the_system():
    A = np.array([[2.0, 0.0], [0.0, 4.0]])
    b = np.array([2.0, 8.0])
    theta = L.ridge_theta(A, b)
    assert np.allclose(theta, [1.0, 2.0])


def test_linucb_learns_which_arm_a_context_prefers():
    # Arm 0 is great when feature x1>0, arm 1 when x1<0. LinUCB should learn this.
    true_theta = np.array([[0.5, 0.4], [0.5, -0.4]])
    pol = L.LinUCB(n_arms=2, dim=2, alpha=0.5, l2=1.0)
    rng = np.random.default_rng(0)
    # Train on many contexts of both signs.
    for _ in range(2000):
        x = np.array([1.0, rng.uniform(-1, 1)])
        arm = pol.select(x, rng)
        probs = L.reward_prob(true_theta, x)
        reward = 1.0 if rng.random() < probs[arm] else 0.0
        pol.update(arm, x, reward)
    # After learning, a positive-feature context should prefer arm 0, negative arm 1.
    assert pol.select(np.array([1.0, 0.9]), rng) == 0
    assert pol.select(np.array([1.0, -0.9]), rng) == 1


def test_update_accumulates_A_and_b():
    pol = L.LinUCB(n_arms=1, dim=2, alpha=1.0, l2=1.0)
    x = np.array([1.0, 2.0])
    pol.update(0, x, reward=1.0)
    assert np.allclose(pol.A[0], np.eye(2) + np.outer(x, x))
    assert np.allclose(pol.b[0], x)               # reward=1 => b += x


def test_linucb_beats_context_free_on_a_contextual_world():
    # Best arm flips with the sign of feature x1 -> context-free is structurally stuck.
    true_theta = np.array([[0.5, 0.45], [0.5, -0.45], [0.4, 0.0]])
    n_rounds = 4000
    rng = np.random.default_rng(7)
    contexts = np.ones((n_rounds, 2))
    contexts[:, 1] = rng.uniform(-1, 1, n_rounds)

    def run(policy):
        r = np.random.default_rng(99)             # same reward draws for both
        return L.simulate_contextual(true_theta, policy, contexts, r)

    linucb = run(L.LinUCB(3, 2, alpha=1.0))
    tf = run(L.ContextFreeThompson(3, 2))

    assert linucb["cum_regret"][-1] < tf["cum_regret"][-1]     # less regret
    assert linucb["best_arm_rate"] > tf["best_arm_rate"]       # more per-user-best picks


def test_simulate_shapes_and_reward_conservation():
    true_theta = np.array([[0.6, 0.1], [0.4, -0.1]])
    contexts = np.ones((50, 2))
    out = L.simulate_contextual(true_theta, L.LinUCB(2, 2), contexts, np.random.default_rng(0))
    assert len(out["chosen"]) == 50
    assert out["cum_regret"].shape == (50,)
    assert np.all(np.diff(out["cum_regret"]) >= 0)            # regret is cumulative
