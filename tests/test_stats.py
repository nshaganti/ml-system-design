"""Tests for the shared bootstrap CI utility (phase0/stats.py)."""

from __future__ import annotations

import numpy as np

from stats import bootstrap_ci, fmt_ci, mean_std


def _mean(x):
    return float(np.mean(x))


def test_ci_brackets_the_point_estimate():
    x = np.random.default_rng(0).normal(5.0, 1.0, size=500)
    point, lo, hi = bootstrap_ci(_mean, [x], n_boot=500, seed=1)
    assert lo < point < hi
    assert abs(point - 5.0) < 0.2          # near the true mean
    assert lo < 5.0 < hi                    # CI covers the truth


def test_ci_narrows_with_more_data():
    rng = np.random.default_rng(2)
    small = rng.normal(0, 1, size=50)
    large = rng.normal(0, 1, size=5000)
    _, lo_s, hi_s = bootstrap_ci(_mean, [small], n_boot=400, seed=3)
    _, lo_l, hi_l = bootstrap_ci(_mean, [large], n_boot=400, seed=3)
    assert (hi_l - lo_l) < (hi_s - lo_s)   # more data -> tighter interval


def test_arrays_are_resampled_together():
    # A perfect per-row product estimator: if rows were resampled independently
    # the correlation structure would break. Here y = 2*x exactly, so mean(y/x)=2
    # for every bootstrap replicate -> a degenerate (zero-width) CI at 2.0.
    x = np.arange(1, 101, dtype=float)
    y = 2.0 * x

    def ratio(a, b):
        return float(np.mean(b / a))

    point, lo, hi = bootstrap_ci(ratio, [x, y], n_boot=200, seed=4)
    assert abs(point - 2.0) < 1e-9
    assert abs(hi - lo) < 1e-9


def test_reproducible_with_seed():
    x = np.random.default_rng(5).normal(size=100)
    a = bootstrap_ci(_mean, [x], n_boot=200, seed=7)
    b = bootstrap_ci(_mean, [x], n_boot=200, seed=7)
    assert a == b


def test_fmt_and_mean_std():
    assert fmt_ci(0.5, 0.4, 0.6, places=2) == "0.50 [0.40, 0.60]"
    m, s = mean_std([1.0, 1.0, 1.0])
    assert m == 1.0 and s == 0.0
