"""
Bootstrap confidence intervals for off-policy estimates
=======================================================
The review of this repo made a fair and pointed criticism: Part II preaches
trustworthy evaluation, yet every OPE/OPL/bandit headline was a single point
estimate with no uncertainty. An IPS value of 0.57 means very different things at
n=200 with heavy tails than at n=50,000 -- and shipping a point estimate with no
interval is exactly the production footgun these phases warn about.

This module provides a percentile bootstrap that resamples the LOGGED ROWS (with
replacement) and recomputes any per-row estimator, yielding a 95% CI. It is pure
NumPy and unit-tested, and it is reused by the Part II phases.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def bootstrap_ci(
    estimate_fn: Callable[..., float],
    arrays: Sequence[np.ndarray],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a per-row estimator.

    Args:
        estimate_fn: callable taking the resampled arrays (same order as
            `arrays`) and returning a scalar (e.g. `ips`, `snips`).
        arrays: per-row arrays of equal length, resampled TOGETHER so each
            bootstrap replicate is a coherent set of logged rows.
        n_boot: number of bootstrap replicates.
        alpha: 1 - confidence (0.05 -> 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        (point_estimate, ci_low, ci_high).
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("all arrays must share the same length (per-row).")
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample.")

    rng = np.random.default_rng(seed)
    point = float(estimate_fn(*arrays))

    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = estimate_fn(*(a[idx] for a in arrays))

    lo = float(np.percentile(boots, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    return point, lo, hi


def fmt_ci(point: float, lo: float, hi: float, places: int = 4) -> str:
    """Human-readable 'value [lo, hi]' for logs and docs."""
    return f"{point:.{places}f} [{lo:.{places}f}, {hi:.{places}f}]"


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and (population) std of a set of per-seed estimates -- the multi-seed
    complement to the bootstrap, used where a whole simulation is re-run under
    different seeds rather than resampled."""
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())
