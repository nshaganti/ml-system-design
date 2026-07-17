"""
Phase 5 -- A/B Testing / Online Experimentation (Rule 16)
===========================================================
> "Measure the delta between models, not the absolute performance."

Offline metrics (Phases 0-4) tell you a model *might* be better. Only a live A/B
test tells you it *is* -- because offline eval can't capture how users actually
react. This module implements the two pieces you cannot get wrong:

  1. Deterministic, sticky assignment. A user must ALWAYS see the same variant
     for the life of the experiment -- otherwise you contaminate the comparison.
     We hash (experiment_name + unit_id) so assignment is stable, salted per
     experiment, and needs no database.

  2. Statistical significance. A raw "+2% conversion" is meaningless without a
     p-value and a confidence interval -- it could be noise. We implement a
     two-proportion z-test and a required-sample-size calculation in pure Python
     (no scipy dependency), using math.erf for the normal CDF.

The cardinal sin this prevents: shipping a model because its number went up,
when the "improvement" was inside the noise band.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

_UINT32_MAX = 0xFFFFFFFF


# ----------------------------------------------------------- assignment


def assign_variant(
    unit_id: str,
    experiment_name: str,
    variants: list[str],
    weights: list[float] | None = None,
) -> str:
    """
    Deterministically map a unit (user) to a variant.

    Sticky: same (experiment, unit) -> same variant, forever, with no state.
    Salted: the experiment_name is mixed into the hash, so two experiments
    assign the *same* user independently (no cross-experiment correlation).
    """
    if weights is None:
        weights = [1.0] * len(variants)
    if len(weights) != len(variants):
        raise ValueError("variants and weights must have equal length")
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")

    digest = hashlib.sha256(f"{experiment_name}:{unit_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / _UINT32_MAX   # uniform in [0, 1]

    cumulative = 0.0
    for variant, w in zip(variants, weights):
        cumulative += w / total
        if bucket < cumulative:
            return variant
    return variants[-1]   # float-rounding guard


# ---------------------------------------------------------- significance


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via the error function (no scipy)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class ABResult:
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    z_score: float
    p_value: float
    ci95: tuple[float, float]
    significant: bool
    n_control: int
    n_treatment: int

    def line(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"  control={self.control_rate:.4f} (n={self.n_control:,})  "
            f"treatment={self.treatment_rate:.4f} (n={self.n_treatment:,})\n"
            f"  absolute lift={self.absolute_lift:+.4f}  "
            f"relative={self.relative_lift:+.1%}\n"
            f"  z={self.z_score:.3f}  p={self.p_value:.4f}  "
            f"95% CI=[{self.ci95[0]:+.4f}, {self.ci95[1]:+.4f}]  -> {sig}"
        )


def two_proportion_ztest(
    successes_control: int, n_control: int,
    successes_treatment: int, n_treatment: int,
    alpha: float = 0.05,
) -> ABResult:
    """
    Two-sided two-proportion z-test for a binary metric (e.g. conversion).

    Null hypothesis: treatment and control convert at the same rate. We compute
    the pooled-variance z statistic, its two-sided p-value, and a 95% CI on the
    difference (unpooled variance, the standard reporting convention).
    """
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("both arms need at least one observation")

    p_c = successes_control / n_control
    p_t = successes_treatment / n_treatment
    p_pool = (successes_control + successes_treatment) / (n_control + n_treatment)

    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    z = (p_t - p_c) / se_pool if se_pool > 0 else 0.0
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))

    se_diff = math.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment)
    diff = p_t - p_c
    z_crit = 1.959963985  # 97.5th percentile of N(0,1)
    ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    return ABResult(
        control_rate=p_c, treatment_rate=p_t,
        absolute_lift=diff,
        relative_lift=(diff / p_c) if p_c > 0 else 0.0,
        z_score=z, p_value=p_value, ci95=ci,
        significant=p_value < alpha,
        n_control=n_control, n_treatment=n_treatment,
    )


def required_sample_size(
    baseline_rate: float,
    min_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Approximate per-arm sample size to detect a relative lift of
    `min_detectable_effect` on a baseline conversion rate, at the given alpha
    and power. Answers "how long must this experiment run?" BEFORE you start --
    the discipline that stops teams peeking and calling noise a win.
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + min_detectable_effect)
    z_alpha = 1.959963985            # two-sided alpha=0.05
    z_beta = 0.8416212336            # power=0.80
    pooled = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * pooled * (1 - pooled))
                 + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2
    return int(math.ceil(numerator / denominator)) if denominator > 0 else 0
