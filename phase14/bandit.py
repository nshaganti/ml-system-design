"""
Phase 14 -- The Explore-and-Learn Loop (contextual-free bandit simulation)
==========================================================================
Part II kept assuming a source of *unbiased* data: Phase 8/9 needed a uniform-random
log, Phase 11 needed a result-randomization bucket. Where does that data come from in
a live system? **Exploration.** A recommender that only ever shows what it currently
believes is best (pure exploitation) manufactures its own selection bias -- it never
gathers evidence about the arms it dismissed early, so it can lock onto a lucky-but-
mediocre item forever. That is the Phase 8 feedback loop, created live.

This module is the smallest honest demonstration: a multi-armed bandit (each "arm" =
an item with a hidden true reward rate) under three strategies:

  * greedy       -- warm up once, then always exploit the empirical best (the trap).
  * epsilon-greedy -- exploit, but keep a slice of random exploration forever.
  * Thompson     -- Bayesian: sample each arm's reward from its Beta posterior and
                    play the sampled-best. Explores exactly as much as its
                    uncertainty warrants.

The punchline connects the whole course: **exploration is the price of unbiased
data, and it is not optional.** Greedy's log has near-zero propensity on most arms
(so it's useless for OPE/OPL, Phase 8-9); the exploring policies keep every arm's
propensity > 0, which is precisely the support condition Part II required.

Pure numeric core (Rule 5): NumPy arrays in, arms/statistics out. `run.py` builds
the world and drives the loop.
"""

from __future__ import annotations

import numpy as np


def empirical_means(successes: np.ndarray, failures: np.ndarray) -> np.ndarray:
    """Reward-rate estimate per arm; an unplayed arm reads 0.0 (pessimistic)."""
    trials = successes + failures
    return np.divide(successes, trials, out=np.zeros_like(successes, dtype=np.float64),
                     where=trials > 0)


def select_greedy(successes: np.ndarray, failures: np.ndarray, rng: np.random.Generator) -> int:
    """Pure exploitation: play the arm with the highest empirical mean."""
    return int(np.argmax(empirical_means(successes, failures)))


def make_epsilon_greedy(epsilon: float):
    """Return an epsilon-greedy selector: explore uniformly w.p. epsilon, else exploit."""
    def select(successes, failures, rng):
        if rng.random() < epsilon:
            return int(rng.integers(len(successes)))
        return select_greedy(successes, failures, rng)
    return select


def select_thompson(successes: np.ndarray, failures: np.ndarray, rng: np.random.Generator) -> int:
    """Sample each arm's rate from Beta(1+succ, 1+fail); play the sampled-best."""
    samples = rng.beta(1.0 + successes, 1.0 + failures)
    return int(np.argmax(samples))


def simulate(
    true_rates: np.ndarray,
    selector,
    n_rounds: int,
    rng: np.random.Generator,
    warmup: int = 1,
) -> dict:
    """
    Run a Bernoulli bandit for `n_rounds`.

    warmup: pull each arm this many times up front (round-robin) before the policy
    takes over -- a launch-time exploration slice. Set 0 for policies (Thompson)
    whose prior already handles cold arms.

    Returns chosen arms, per-arm successes/failures, and the reward trajectory.
    """
    n = len(true_rates)
    successes = np.zeros(n, dtype=np.float64)
    failures = np.zeros(n, dtype=np.float64)
    chosen = np.empty(n_rounds, dtype=np.int64)
    rewards = np.empty(n_rounds, dtype=np.float64)

    for t in range(n_rounds):
        arm = (t % n) if t < warmup * n else selector(successes, failures, rng)
        reward = 1.0 if rng.random() < true_rates[arm] else 0.0
        if reward > 0:
            successes[arm] += 1
        else:
            failures[arm] += 1
        chosen[t] = arm
        rewards[t] = reward

    return {"chosen": chosen, "rewards": rewards, "successes": successes, "failures": failures}


def cumulative_regret(true_rates: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """Running sum of (best possible rate - chosen arm's true rate) per round."""
    best = float(true_rates.max())
    return np.cumsum(best - true_rates[chosen])


def best_arm_fraction(true_rates: np.ndarray, chosen: np.ndarray) -> float:
    """Fraction of rounds the truly-best arm was played."""
    return float(np.mean(chosen == int(np.argmax(true_rates))))


def arm_support(successes: np.ndarray, failures: np.ndarray, min_pulls: int = 5) -> float:
    """
    Fraction of arms played at least `min_pulls` times -- a proxy for whether the
    policy's log has usable propensity (support) across the catalog. Greedy locks
    onto one arm, so its ongoing log supports almost nothing (Phase 8's problem).
    """
    trials = successes + failures
    return float(np.mean(trials >= min_pulls))


def identified_best(successes: np.ndarray, failures: np.ndarray, true_rates: np.ndarray) -> bool:
    """Did the policy's final empirical estimate point at the truly-best arm?"""
    return int(np.argmax(empirical_means(successes, failures))) == int(np.argmax(true_rates))
