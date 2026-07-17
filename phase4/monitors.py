"""
Phase 4 -- Monitoring & Drift Detection (Rule 10)
===================================================
> "Watch for silent failures."

Production ML systems rarely crash. They DEGRADE: a pipeline stalls and serves
week-old features; an item goes out of stock and keeps getting recommended; a
category goes viral and the model's prior is wrong. None of these throw an
exception. This module gives you the assertions that turn silent degradation into
a loud, countable signal.

Three layers (design doc, Phase 4):
  1. Data health   -- row counts, null rates, distribution drift
  2. Model health  -- prediction stability, calibration, fallback rate, diversity
  3. Business      -- CTR / add-to-cart / revenue (consumed from the event stream)

The numeric core (PSI, KL divergence) lives here as pure functions so it's
trivially testable. A `Monitor` runs a suite of checks and returns a single
pipeline-gate status: PASS or FAIL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-9   # avoid log(0) / divide-by-zero in drift math


# ----------------------------------------------------------- numeric core


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Population Stability Index -- the industry-standard drift metric.

    Bins `expected` by its quantiles, then measures how much `actual`'s mass has
    shifted between those bins:

        PSI = sum( (a% - e%) * ln(a% / e%) )

    Rule-of-thumb thresholds:
        PSI < 0.1        no significant shift
        0.1 <= PSI < 0.2 moderate shift -- investigate
        PSI >= 0.2       major shift -- likely a problem
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        return 0.0

    # Quantile edges from the reference distribution; unique() guards against
    # heavy point masses producing duplicate edges.
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(expected, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)

    e_pct = e_counts / max(e_counts.sum(), 1)
    a_pct = a_counts / max(a_counts.sum(), 1)

    e_pct = np.clip(e_pct, EPS, None)
    a_pct = np.clip(a_pct, EPS, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) over two discrete distributions (auto-normalized)."""
    p = np.clip(np.asarray(p, dtype=float), EPS, None)
    q = np.clip(np.asarray(q, dtype=float), EPS, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def expected_calibration_error(
    predicted: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Mean absolute gap between predicted probability and observed frequency,
    averaged over prediction-score bins (weighted by bin size). 0 = perfectly
    calibrated. A ranker can rank well yet be badly calibrated -- worth watching.
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if predicted.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(predicted, edges[1:-1]), 0, bins - 1)
    total = predicted.size
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        ece += (n / total) * abs(predicted[mask].mean() - actual[mask].mean())
    return float(ece)


# --------------------------------------------------------------- framework


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str

    def line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"  [{status}] {self.name:<26} value={self.value:.4f} (thr {self.threshold:.4f}) -- {self.detail}"


# --- Layer 1: data health ---------------------------------------------------


def check_row_count(current: int, baseline: int, min_ratio: float = 0.8) -> CheckResult:
    """Volume check: alert if today's row count drops below min_ratio of baseline."""
    ratio = current / baseline if baseline > 0 else 0.0
    return CheckResult(
        "row_count_ratio", ratio >= min_ratio, ratio, min_ratio,
        f"{current:,} vs baseline {baseline:,}",
    )


def check_null_rate(null_count: int, total: int, max_rate: float = 0.01) -> CheckResult:
    rate = null_count / total if total > 0 else 0.0
    return CheckResult(
        "null_rate", rate <= max_rate, rate, max_rate,
        f"{null_count:,}/{total:,} null",
    )


def check_drift(expected: np.ndarray, actual: np.ndarray, max_psi: float = 0.2) -> CheckResult:
    psi = population_stability_index(expected, actual)
    return CheckResult(
        "feature_drift_psi", psi <= max_psi, psi, max_psi,
        "stable" if psi < 0.1 else ("moderate shift" if psi < 0.2 else "major shift"),
    )


# --- Layer 2: model health --------------------------------------------------


def check_fallback_rate(fallbacks: int, total: int, max_rate: float = 0.05) -> CheckResult:
    rate = fallbacks / total if total > 0 else 0.0
    return CheckResult(
        "fallback_rate", rate <= max_rate, rate, max_rate,
        f"{fallbacks:,}/{total:,} requests degraded",
    )


def check_diversity(avg_categories: float, min_categories: float = 3.0) -> CheckResult:
    return CheckResult(
        "recommendation_diversity", avg_categories >= min_categories,
        avg_categories, min_categories, "avg unique categories in top-K",
    )


def check_calibration(ece: float, max_ece: float = 0.1) -> CheckResult:
    return CheckResult("calibration_ece", ece <= max_ece, ece, max_ece, "expected calibration error")


class Monitor:
    """Runs a list of CheckResults and produces a single gate status."""

    def __init__(self, checks: list[CheckResult]):
        self.checks = checks

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def gate_status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def report(self) -> str:
        lines = [c.line() for c in self.checks]
        lines.append(f"\n  GATE: {self.gate_status}"
                     + ("" if self.passed else "  (one or more checks failed -- alert!)"))
        return "\n".join(lines)
