"""
Phase 4 tests: the monitoring numeric core (PSI, KL, ECE) and the check framework.

The numeric functions are where correctness matters most -- a wrong drift metric
means either false alarms or silent misses.
"""

from __future__ import annotations

import numpy as np

from monitors import (
    population_stability_index,
    kl_divergence,
    expected_calibration_error,
    check_row_count,
    check_null_rate,
    check_drift,
    check_fallback_rate,
    check_diversity,
    check_calibration,
    Monitor,
)


# ------------------------------------------------------------------- PSI


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    psi = population_stability_index(x, x.copy())
    assert psi < 1e-6


def test_psi_grows_with_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, size=5000)
    small = rng.normal(0.2, 1, size=5000)
    big = rng.normal(2.0, 1, size=5000)
    psi_small = population_stability_index(base, small)
    psi_big = population_stability_index(base, big)
    assert psi_big > psi_small > 0


def test_psi_handles_empty():
    assert population_stability_index(np.array([]), np.array([1.0])) == 0.0


# -------------------------------------------------------------------- KL


def test_kl_zero_for_identical():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert kl_divergence(p, p) < 1e-9


def test_kl_positive_for_different():
    p = np.array([0.9, 0.1])
    q = np.array([0.1, 0.9])
    assert kl_divergence(p, q) > 0


# ------------------------------------------------------------------- ECE


def test_ece_zero_for_perfect_calibration():
    # predicted prob == observed frequency in every bin.
    predicted = np.array([0.0] * 50 + [1.0] * 50)
    actual = np.array([0.0] * 50 + [1.0] * 50)
    assert expected_calibration_error(predicted, actual) < 1e-9


def test_ece_high_for_overconfident():
    # Model says 0.9 everywhere but only half actually convert.
    predicted = np.full(100, 0.9)
    actual = np.array([1.0, 0.0] * 50)
    assert expected_calibration_error(predicted, actual) > 0.3


# ------------------------------------------------------------ check framework


def test_row_count_check_pass_and_fail():
    assert check_row_count(90, 100, min_ratio=0.8).passed
    assert not check_row_count(50, 100, min_ratio=0.8).passed


def test_null_rate_check():
    assert check_null_rate(0, 1000).passed
    assert not check_null_rate(100, 1000, max_rate=0.01).passed


def test_drift_check_flags_major_shift():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, size=5000)
    shifted = rng.normal(3, 1, size=5000)
    assert not check_drift(base, shifted, max_psi=0.2).passed
    assert check_drift(base, base.copy(), max_psi=0.2).passed


def test_fallback_rate_check():
    assert check_fallback_rate(1, 100, max_rate=0.05).passed
    assert not check_fallback_rate(20, 100, max_rate=0.05).passed


def test_diversity_check():
    assert check_diversity(5.0, min_categories=3.0).passed
    assert not check_diversity(1.0, min_categories=3.0).passed


def test_calibration_check():
    assert check_calibration(0.05, max_ece=0.1).passed
    assert not check_calibration(0.25, max_ece=0.1).passed


# ---------------------------------------------------------------- Monitor


def test_monitor_gate_passes_when_all_pass():
    m = Monitor([check_row_count(95, 100), check_null_rate(0, 100)])
    assert m.passed
    assert m.gate_status == "PASS"


def test_monitor_gate_fails_if_any_fails():
    m = Monitor([check_row_count(95, 100), check_null_rate(50, 100, max_rate=0.01)])
    assert not m.passed
    assert m.gate_status == "FAIL"
    assert "FAIL" in m.report()
