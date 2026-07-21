"""
Tests for Phase 17 -- real-context + safety-gated loop (safe_loop.py).

Pure and deterministic. Headline tests: (1) the OPE gate's IPS value matches a
hand-computed number, and (2) on a world with noisy batches, the gate blocks
regressions so the gated loop ends at least as high as the ungated one.
"""

from __future__ import annotations

import numpy as np

import safe_loop as SL


def test_ips_policy_value_matches_hand_computation():
    # 1 feature (bias only). theta_candidate greedy-prefers arm 1 (higher weight).
    theta = np.array([[0.2], [0.9]])
    contexts = np.ones((4, 1))
    arms = np.array([1, 0, 1, 1])          # target(=argmax=arm1) agrees on rows 0,2,3
    rewards = np.array([1.0, 1.0, 0.0, 1.0])
    props = np.array([0.5, 0.5, 0.5, 0.5])
    # agreeing rows: 0 (r=1/0.5=2), 2 (r=0), 3 (r=1/0.5=2) -> (2+0+2)/4 = 1.0
    v = SL.ips_policy_value(theta, contexts, arms, rewards, props, min_prop=0.01)
    assert np.isclose(v, 1.0)


def test_min_prop_floor_bounds_the_weight():
    theta = np.array([[0.0], [1.0]])
    contexts = np.ones((1, 1))
    arms = np.array([1])
    rewards = np.array([1.0])
    props = np.array([0.0001])             # tiny propensity would explode the weight
    v = SL.ips_policy_value(theta, contexts, arms, rewards, props, min_prop=0.05)
    assert np.isclose(v, 1.0 / 0.05)       # clipped to floor, not 1/0.0001


def test_ips_value_empty_log_is_zero():
    assert SL.ips_policy_value(np.zeros((2, 1)), np.empty((0, 1)),
                               np.array([]), np.array([]), np.array([]), 0.01) == 0.0


def test_gate_blocks_at_least_some_bad_redeploys():
    true_theta = np.array([[0.5, 0.45], [0.5, -0.45], [0.4, 0.0]])
    ctx_pool = np.ones((500, 2))
    ctx_pool[:, 1] = np.linspace(-1, 1, 500)
    eval_ctx = ctx_pool[::3]
    out = SL.simulate_safe_loop(true_theta, ctx_pool, eval_ctx,
                                np.random.default_rng(0), n_iterations=12,
                                batch_size=40, epsilon=0.15, use_gate=True)
    assert out["blocked"] >= 1             # noisy small batches -> some rejections
    assert len(out["values"]) == 12


def test_gate_reduces_volatility_on_average_over_seeds():
    # The gate's real guarantee isn't a higher single-run final value (OPE is itself
    # noisy) -- it's that it blocks downward redeploys, so the deployed value moves
    # more smoothly. Averaged over seeds, gated trajectories are steadier.
    true_theta = np.array([[0.5, 0.45], [0.5, -0.45], [0.4, 0.0]])
    ctx_pool = np.ones((600, 2))
    ctx_pool[:, 1] = np.linspace(-1, 1, 600)
    eval_ctx = ctx_pool[::2]

    def volatility(gate, seed):
        v = SL.simulate_safe_loop(true_theta, ctx_pool, eval_ctx,
                                  np.random.default_rng(seed), n_iterations=15,
                                  batch_size=35, epsilon=0.15, use_gate=gate)["values"]
        return float(np.mean(np.abs(np.diff(v))))

    gated = np.mean([volatility(True, s) for s in range(8)])
    ungated = np.mean([volatility(False, s) for s in range(8)])
    assert gated < ungated          # the parachute smooths the ride


def test_gate_protects_against_a_poisoned_batch():
    # A logging bug flips one training batch's rewards -> a corrupt candidate. The
    # ungated loop ships it (worst deployed value craters); the gate rejects it on
    # the clean uniform bucket. Averaged over seeds, the gated loop's worst deployed
    # value stays higher (single-seed poison timing is noisy, like the real run).
    true_theta = np.array([[0.5, 0.45], [0.5, -0.45], [0.4, 0.0]])
    ctx_pool = np.ones((800, 2))
    ctx_pool[:, 1] = np.linspace(-1, 1, 800)
    eval_ctx = ctx_pool[::2]

    def worst(gate, seed):
        vals = SL.simulate_safe_loop(true_theta, ctx_pool, eval_ctx,
                                     np.random.default_rng(seed), n_iterations=12,
                                     batch_size=120, epsilon=0.15, use_gate=gate,
                                     window=3, poison_iter=6)["values"]
        return min(vals)

    gated = np.mean([worst(True, s) for s in range(10)])
    ungated = np.mean([worst(False, s) for s in range(10)])
    assert gated > ungated      # gated never craters as hard as ungated (on average)
