"""
Tests for Phase 9 -- off-policy LEARNING core (learning.py).

All pure-numeric, no dataset. The through-line: learning from unbiased value
estimates yields a policy with higher TRUE value than learning from confounded
ones -- which is the whole point of Phase 9.
"""

from __future__ import annotations

import numpy as np
import pytest

import learning


ITEMS = ["a", "b", "c"]


def test_reward_rate_per_item_averages_and_zero_fills():
    item_ids = np.array(["a", "a", "b"])
    rewards = np.array([1.0, 0.0, 1.0])
    rates = learning.reward_rate_per_item(item_ids, rewards, ITEMS)
    assert rates["a"] == pytest.approx(0.5)   # (1+0)/2
    assert rates["b"] == pytest.approx(1.0)
    assert rates["c"] == 0.0                   # unseen -> zero prior


def test_softmax_policy_is_a_distribution_and_favors_high_value():
    v = {"a": 1.0, "b": 0.0, "c": 0.0}
    pi = learning.softmax_policy(v, ITEMS, temperature=0.1)
    assert pi.shape == (3,)
    assert pi.sum() == pytest.approx(1.0)
    assert (pi >= 0).all()
    assert pi[0] > pi[1] and pi[0] > pi[2]     # mass flows to the best item


def test_lower_temperature_is_greedier():
    v = {"a": 1.0, "b": 0.5, "c": 0.0}
    hot = learning.softmax_policy(v, ITEMS, temperature=1.0)
    cold = learning.softmax_policy(v, ITEMS, temperature=0.05)
    assert cold[0] > hot[0]                     # colder -> more mass on the best


def test_softmax_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        learning.softmax_policy({"a": 1.0}, ITEMS, temperature=0.0)


def test_greedy_policy_is_a_one_hot_on_the_best_item():
    v = {"a": 0.2, "b": 0.9, "c": 0.1}
    pi = learning.greedy_policy(v, ITEMS)
    assert pi.tolist() == [0.0, 1.0, 0.0]


def test_policy_value_is_expected_true_reward():
    r_true = {"a": 0.1, "b": 0.8, "c": 0.3}
    pi = np.array([0.0, 1.0, 0.0])             # always pick b
    assert learning.policy_value(pi, r_true, ITEMS) == pytest.approx(0.8)


def test_learning_from_unbiased_beats_learning_from_biased():
    """
    The core Phase 9 claim, in miniature. Item 'b' is genuinely best (true rate
    0.9) but the biased log makes 'a' look best (production over-showed 'a' to
    users who liked it). A policy learned from biased rates picks 'a'; one learned
    from true rates picks 'b' -> higher true value.
    """
    r_biased = {"a": 0.9, "b": 0.4, "c": 0.1}   # confounded: 'a' looks great
    r_true = {"a": 0.2, "b": 0.9, "c": 0.1}     # reality: 'b' is best

    pi_naive = learning.softmax_policy(r_biased, ITEMS, temperature=0.05)
    pi_learned = learning.softmax_policy(r_true, ITEMS, temperature=0.05)

    v_naive = learning.policy_value(pi_naive, r_true, ITEMS)
    v_learned = learning.policy_value(pi_learned, r_true, ITEMS)

    assert v_learned > v_naive
