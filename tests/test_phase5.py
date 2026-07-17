"""
Phase 5 tests: deterministic assignment + A/B statistics.

Assignment correctness (sticky, balanced, salted) and the z-test math are the
load-bearing pieces -- a bug here means shipping models on noise.
"""

from __future__ import annotations

import pytest

from experiment import (
    assign_variant,
    two_proportion_ztest,
    required_sample_size,
)


# ------------------------------------------------------------- assignment


def test_assignment_is_sticky():
    for uid in ["u1", "u2", "user_9999", "abc"]:
        a = assign_variant(uid, "exp", ["control", "treatment"])
        b = assign_variant(uid, "exp", ["control", "treatment"])
        assert a == b


def test_assignment_is_salted_per_experiment():
    # The same user can land in different arms across DIFFERENT experiments;
    # across many users the two experiments must not be identical.
    users = [f"user_{i}" for i in range(2000)]
    a = [assign_variant(u, "exp_A", ["c", "t"]) for u in users]
    b = [assign_variant(u, "exp_B", ["c", "t"]) for u in users]
    assert a != b


def test_assignment_respects_weights():
    users = [f"user_{i}" for i in range(20000)]
    assigns = [assign_variant(u, "exp", ["control", "treatment"], [0.9, 0.1]) for u in users]
    frac_treatment = assigns.count("treatment") / len(assigns)
    assert 0.08 < frac_treatment < 0.12   # ~10%, within sampling noise


def test_assignment_roughly_even_for_equal_weights():
    users = [f"user_{i}" for i in range(20000)]
    assigns = [assign_variant(u, "exp", ["control", "treatment"]) for u in users]
    frac = assigns.count("control") / len(assigns)
    assert 0.47 < frac < 0.53


def test_assignment_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        assign_variant("u", "exp", ["a", "b"], [1.0])


def test_assignment_handles_three_way_split():
    users = [f"user_{i}" for i in range(30000)]
    assigns = [assign_variant(u, "exp", ["a", "b", "c"]) for u in users]
    for v in ["a", "b", "c"]:
        assert 0.30 < assigns.count(v) / len(assigns) < 0.36


# ---------------------------------------------------------- significance


def test_ztest_no_difference_is_not_significant():
    r = two_proportion_ztest(100, 1000, 100, 1000)
    assert r.absolute_lift == 0.0
    assert r.p_value > 0.9
    assert not r.significant


def test_ztest_large_clear_difference_is_significant():
    r = two_proportion_ztest(100, 1000, 200, 1000)
    assert r.absolute_lift > 0
    assert r.p_value < 0.001
    assert r.significant
    assert r.relative_lift == pytest.approx(1.0, abs=0.01)  # 0.20 vs 0.10


def test_ztest_tiny_difference_in_small_sample_not_significant():
    # 51 vs 50 out of 100 -- obviously noise.
    r = two_proportion_ztest(50, 100, 51, 100)
    assert not r.significant


def test_ztest_ci_brackets_the_difference():
    r = two_proportion_ztest(100, 1000, 150, 1000)
    lo, hi = r.ci95
    assert lo < r.absolute_lift < hi


def test_ztest_requires_nonempty_arms():
    with pytest.raises(ValueError):
        two_proportion_ztest(0, 0, 1, 10)


# ------------------------------------------------------ sample size


def test_sample_size_grows_as_effect_shrinks():
    small_effect = required_sample_size(0.1, 0.05)
    big_effect = required_sample_size(0.1, 0.20)
    # Detecting a smaller lift requires MORE samples.
    assert small_effect > big_effect > 0


def test_sample_size_is_positive_int():
    n = required_sample_size(0.1, 0.1)
    assert isinstance(n, int) and n > 0
