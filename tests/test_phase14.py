"""
Tests for Phase 14 -- bandit core (bandit.py).

Pure and deterministic (fixed seeds). The headline test proves the phase's whole
point: on a deceptive world, exploring policies end with LESS regret than greedy AND
a log that supports more arms -- i.e. exploration buys both performance and unbiased
data.
"""

from __future__ import annotations

import numpy as np

import bandit as B


def test_empirical_means_handles_unplayed_arms():
    s = np.array([3.0, 0.0, 0.0])
    f = np.array([1.0, 0.0, 2.0])
    m = B.empirical_means(s, f)
    assert m[0] == 0.75      # 3/4
    assert m[1] == 0.0       # unplayed -> 0, no divide-by-zero
    assert m[2] == 0.0       # 0/2


def test_greedy_picks_the_empirical_best():
    s = np.array([1.0, 5.0, 2.0])
    f = np.array([1.0, 1.0, 1.0])
    assert B.select_greedy(s, f, np.random.default_rng(0)) == 1


def test_epsilon_zero_is_greedy_and_epsilon_one_is_random():
    s = np.array([1.0, 9.0, 0.0])
    f = np.array([1.0, 1.0, 1.0])
    always_exploit = B.make_epsilon_greedy(0.0)
    assert always_exploit(s, f, np.random.default_rng(0)) == 1     # never explores
    always_explore = B.make_epsilon_greedy(1.0)
    picks = {always_explore(s, f, np.random.default_rng(i)) for i in range(50)}
    assert len(picks) > 1                                          # explores widely


def test_thompson_returns_a_valid_arm():
    s = np.array([2.0, 0.0, 5.0])
    f = np.array([1.0, 3.0, 1.0])
    arm = B.select_thompson(s, f, np.random.default_rng(0))
    assert 0 <= arm < 3


def test_cumulative_regret_is_monotone_and_correct():
    rates = np.array([0.2, 0.9])            # best = 0.9
    chosen = np.array([0, 1, 0])            # regrets: 0.7, 0.0, 0.7
    reg = B.cumulative_regret(rates, chosen)
    assert np.all(np.diff(reg) >= 0)
    assert np.isclose(reg[-1], 0.7 + 0.0 + 0.7)


def test_exploration_beats_greedy_on_a_deceptive_world():
    # 20 arms: one clearly best (0.65), the rest a hair below (0.55) so a lucky
    # warmup can trap greedy on a runner-up.
    rates = np.full(20, 0.55)
    rates[7] = 0.65
    n_rounds = 8000

    def run(selector, warmup):
        rng = np.random.default_rng(123)   # same draws for every policy
        out = B.simulate(rates, selector, n_rounds, rng, warmup=warmup)
        return out, B.cumulative_regret(rates, out["chosen"])[-1]

    g_out, g_regret = run(B.select_greedy, 1)
    t_out, t_regret = run(B.select_thompson, 0)
    e_out, e_regret = run(B.make_epsilon_greedy(0.1), 1)

    # Thompson ends with clearly less regret than greedy...
    assert t_regret < g_regret
    # ...and identifies the true best arm, which greedy need not.
    assert B.identified_best(t_out["successes"], t_out["failures"], rates)
    # Exploring policies support far more arms than the locked-in greedy log.
    assert B.arm_support(e_out["successes"], e_out["failures"]) > \
           B.arm_support(g_out["successes"], g_out["failures"])


def test_simulate_shapes_and_conservation():
    rates = np.array([0.3, 0.7, 0.5])
    rng = np.random.default_rng(0)
    out = B.simulate(rates, B.select_greedy, 100, rng, warmup=1)
    assert len(out["chosen"]) == 100
    assert out["successes"].sum() + out["failures"].sum() == 100    # every round counted
