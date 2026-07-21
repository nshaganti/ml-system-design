r"""
Phase 15 -- The Contextual Bandit (LinUCB)
===========================================
Phase 14's bandit learned ONE reward rate per arm -- it treated every user the same.
Real recommenders don't: the best item depends on WHO is asking. A contextual bandit
conditions its choice (and its exploration) on a context vector x -- the user's
features -- so it can personalize while still exploring enough to keep an unbiased
log (the Phase 14 lesson, now per-user).

LinUCB (Li et al., 2010) is the canonical algorithm. Per arm it fits a ridge-
regression reward model on the contexts it has seen, and picks the arm maximizing

    UCB_a(x) = theta_a . x   +   alpha * sqrt( x^T A_a^{-1} x )
               \___________/     \_________________________/
               predicted reward   uncertainty bonus (explore where unsure)

where A_a = l2*I + sum(x x^T) and theta_a = A_a^{-1} b_a, b_a = sum(reward * x).
The bonus shrinks as an arm is seen in similar contexts, so LinUCB explores exactly
where its per-arm model is still uncertain -- the contextual analogue of Thompson's
uncertainty-proportional exploration.

A constant 1.0 as the first context feature acts as the per-arm intercept (the base
rate), so `theta_a[0]` plays the role of Phase 14's context-free rate.

Pure NumPy (Rule 5): the learners are small stateful objects with select()/update();
`run.py` builds the world and drives the loop. No I/O here.
"""

from __future__ import annotations

import numpy as np


def ridge_theta(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A theta = b for the ridge reward weights (never invert explicitly)."""
    return np.linalg.solve(A, b)


def reward_prob(true_theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    True click probability per arm for context x: clip(theta_a . x) into [0.01, 0.99]
    so every arm is a genuine Bernoulli. `true_theta` is (n_arms, dim); x is (dim,).
    """
    return np.clip(true_theta @ x, 0.01, 0.99)


class LinUCB:
    """
    Disjoint LinUCB: an independent ridge model + uncertainty bonus per arm.

    alpha : exploration strength (bigger = more optimism about unseen contexts).
    l2    : ridge prior; also keeps A invertible before any data arrives.
    """

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.0, l2: float = 1.0):
        self.n_arms = n_arms
        self.dim = dim
        self.alpha = alpha
        self.A = np.array([np.eye(dim) * l2 for _ in range(n_arms)])
        self.b = np.zeros((n_arms, dim))

    def select(self, x: np.ndarray, rng: np.random.Generator | None = None) -> int:
        scores = np.empty(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            scores[a] = theta @ x + self.alpha * np.sqrt(max(x @ A_inv @ x, 0.0))
        return int(np.argmax(scores))

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x


class ContextFreeThompson:
    """
    Phase 14's Thompson sampler, wrapped in the contextual policy interface so it can
    be graded on the SAME contextual world -- the baseline that IGNORES context. It
    can only ever learn each arm's average rate across all users, so it cannot
    personalize (which is exactly the gap LinUCB is meant to close).
    """

    def __init__(self, n_arms: int, dim: int | None = None):
        self.n_arms = n_arms
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)

    def select(self, x: np.ndarray, rng: np.random.Generator) -> int:
        return int(np.argmax(rng.beta(1.0 + self.successes, 1.0 + self.failures)))

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        if reward > 0:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1


def simulate_contextual(
    true_theta: np.ndarray,
    policy,
    contexts: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """
    Drive `policy` over a stream of `contexts` (n_rounds, dim). Each round: policy
    picks an arm for the context, we draw a Bernoulli reward from the TRUE model,
    the policy learns, and we log per-round regret vs the context's best arm.
    """
    n = len(contexts)
    chosen = np.empty(n, dtype=np.int64)
    rewards = np.empty(n, dtype=np.float64)
    inst_regret = np.empty(n, dtype=np.float64)
    best_hits = 0

    for t in range(n):
        x = contexts[t]
        arm = policy.select(x, rng)
        probs = reward_prob(true_theta, x)
        reward = 1.0 if rng.random() < probs[arm] else 0.0
        policy.update(arm, x, reward)

        chosen[t] = arm
        rewards[t] = reward
        inst_regret[t] = float(probs.max() - probs[arm])
        if arm == int(np.argmax(probs)):
            best_hits += 1

    return {
        "chosen": chosen,
        "rewards": rewards,
        "cum_regret": np.cumsum(inst_regret),
        "best_arm_rate": best_hits / n if n else 0.0,
    }
