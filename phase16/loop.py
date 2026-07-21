"""
Phase 16 -- Closing the Loop (explore -> learn off-policy -> redeploy -> repeat)
================================================================================
The capstone. Every previous phase was one arc of the cycle a real recommender runs
forever; this phase wires them into the loop itself:

    deploy a policy  ->  it EXPLORES and logs (context, action, propensity, reward)
        ^                                                     |
        |                                                     v
    redeploy the  <-  LEARN a better policy OFF-POLICY from that log (Phase 9),
    greedy policy      correcting the logging bias with propensities (Part II)

It unifies:
  * Phase 15 -- the contextual world + an epsilon-soft LinUCB-style logging policy.
  * Part II  -- known propensities; IPS-correct the per-arm reward models so the
                context distribution each arm was *shown in* doesn't bias its model.
  * Phase 9  -- learn a policy off-policy and grade it by its TRUE value.

The headline comparison (in run.py): a closed loop that keeps exploring climbs its
TRUE deployed value toward the skyline, while a no-exploration variant learns once,
stops gathering evidence on the arms it dismissed, and STALLS -- the Phase 8/14
feedback trap, now shown to cap a live system's ceiling.

Pure NumPy core (Rule 5). Reuses Phase 15's reward model of the world.
"""

from __future__ import annotations

import numpy as np

from linucb import reward_prob   # Phase 15: clip(true_theta @ x) -> per-arm P(click)


def epsilon_greedy_propensities(scores: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Propensity of each arm under an epsilon-soft greedy policy on `scores`:
    epsilon/n mass spread over all arms + (1-epsilon) on the argmax. Every arm keeps
    propensity > 0 (for epsilon > 0), which is the support condition Part II needs and
    the reason we can IPS-correct the log later.
    """
    n = len(scores)
    p = np.full(n, epsilon / n, dtype=np.float64)
    p[int(np.argmax(scores))] += 1.0 - epsilon
    return p


def fit_reward_models(
    contexts: np.ndarray, arms: np.ndarray, rewards: np.ndarray,
    weights: np.ndarray | None, n_arms: int, dim: int, l2: float = 1.0,
) -> np.ndarray:
    """
    One ridge reward model per arm: theta_a = A_a^{-1} b_a over the rows where that
    arm was played. `weights` (e.g. IPS 1/propensity) reweights each row so an arm's
    model reflects the WHOLE context distribution, not just the contexts it tended to
    be greedily chosen in. Unplayed arms keep theta = 0 (the prior).
    """
    contexts = np.asarray(contexts, dtype=np.float64)
    arms = np.asarray(arms)
    rewards = np.asarray(rewards, dtype=np.float64)
    w = np.ones(len(arms)) if weights is None else np.asarray(weights, dtype=np.float64)

    theta = np.zeros((n_arms, dim))
    for a in range(n_arms):
        mask = arms == a
        if not mask.any():
            continue
        X, y, wa = contexts[mask], rewards[mask], w[mask]
        A = l2 * np.eye(dim) + (X * wa[:, None]).T @ X
        b = (X * (wa * y)[:, None]).sum(axis=0)
        theta[a] = np.linalg.solve(A, b)
    return theta


def greedy_value(theta: np.ndarray, eval_contexts: np.ndarray, true_theta: np.ndarray) -> float:
    """
    TRUE value of the greedy policy induced by a learned `theta`: for each eval
    context pick argmax(theta . x), then score it under the TRUE world model. This is
    the honest number the deployed policy is graded on (Phase 9's policy_value, made
    contextual).
    """
    total = 0.0
    for x in eval_contexts:
        arm = int(np.argmax(theta @ x))
        total += float(reward_prob(true_theta, x)[arm])
    return total / len(eval_contexts)


def skyline_value(true_theta: np.ndarray, eval_contexts: np.ndarray) -> float:
    """Value of the oracle policy that always plays the true-best arm per context."""
    return float(np.mean([reward_prob(true_theta, x).max() for x in eval_contexts]))


def uniform_value(true_theta: np.ndarray, eval_contexts: np.ndarray) -> float:
    """Value of picking arms uniformly at random -- the no-learning floor."""
    return float(np.mean([reward_prob(true_theta, x).mean() for x in eval_contexts]))


def simulate_loop(
    true_theta: np.ndarray,
    eval_contexts: np.ndarray,
    rng: np.random.Generator,
    n_iterations: int,
    batch_size: int,
    epsilon: float,
    use_ips: bool = True,
    l2: float = 1.0,
) -> list[float]:
    """
    Run the full explore -> learn -> redeploy cycle for `n_iterations` and return the
    TRUE deployed value after each iteration.

    epsilon = 0 disables exploration (the trap baseline): the logging policy is
    deterministic greedy, so only the currently-best arm is ever logged and the other
    arms' models never improve. use_ips toggles the propensity correction.
    """
    n_arms, dim = true_theta.shape
    ctx_log, arm_log, rew_log, w_log = [], [], [], []
    theta = np.zeros((n_arms, dim))   # initial model: all scores 0
    values: list[float] = []

    for _ in range(n_iterations):
        # DEPLOY + EXPLORE: act with the current model under an epsilon-soft policy.
        for _ in range(batch_size):
            x = np.ones(dim)
            x[1:] = rng.uniform(-1.0, 1.0, dim - 1)
            props = epsilon_greedy_propensities(theta @ x, epsilon)
            arm = int(rng.choice(n_arms, p=props))
            reward = 1.0 if rng.random() < reward_prob(true_theta, x)[arm] else 0.0
            ctx_log.append(x)
            arm_log.append(arm)
            rew_log.append(reward)
            w_log.append(1.0 / props[arm] if use_ips else 1.0)

        # LEARN off-policy from the ENTIRE accumulated log, then measure true value.
        theta = fit_reward_models(
            np.array(ctx_log), np.array(arm_log), np.array(rew_log),
            np.array(w_log), n_arms, dim, l2=l2,
        )
        values.append(greedy_value(theta, eval_contexts, true_theta))

    return values
