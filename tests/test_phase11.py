"""
Tests for Phase 11 -- position-bias debiasing core (position_bias.py).

Pure, dataset-free. The headline test is that IPW recovers the true relevance
RANKING from position-confounded clicks while naive CTR does not -- the whole point
of the phase, proven on a controlled draw with a fixed seed.
"""

from __future__ import annotations

import numpy as np

import position_bias as pb


def test_examination_curve_is_decreasing_and_normalized():
    e = pb.examination_curve(10, decay=1.0)
    assert e[0] == 1.0
    assert np.all(np.diff(e) < 0)          # strictly decreasing down the page


def test_spearman_perfect_and_inverse():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert pb.spearman(a, a) == 1.0
    assert pb.spearman(a, a[::-1]) == -1.0


def test_topk_recovery_counts_overlap():
    truth = np.array([0.1, 0.9, 0.8, 0.2, 0.7])   # top-2 = items 1, 2
    est = np.array([0.0, 1.0, 0.5, 0.0, 0.9])     # top-2 = items 1, 4
    assert pb.topk_recovery(est, truth, k=2) == 0.5


def test_naive_and_ipw_grouped_means():
    # Two items, item 0 at position 0 (exam 1.0), item 1 at position 1 (exam 0.5).
    clicks = np.array([1.0, 1.0, 0.0, 1.0])
    item_ids = np.array([0, 0, 1, 1])
    positions = np.array([0, 0, 1, 1])
    exam = np.array([1.0, 0.5])
    naive = pb.naive_relevance(clicks, item_ids, 2)
    assert naive[0] == 1.0 and naive[1] == 0.5     # raw CTR
    ipw = pb.ipw_relevance(clicks, positions, exam, item_ids, 2)
    assert ipw[0] == 1.0 and ipw[1] == 1.0         # item 1's clicks upweighted by 1/0.5


def test_ipw_recovers_ranking_that_naive_ctr_destroys():
    """
    Controlled draw: relevance is independent of slot, so naive CTR is dominated
    by the examination curve and mis-ranks. IPW with the true curve recovers it.
    """
    rng = np.random.default_rng(0)
    n_items, sessions = 40, 2000
    true_rel = rng.uniform(0.1, 0.7, size=n_items)
    slot = rng.permutation(n_items)                # relevance-independent slots
    exam = pb.examination_curve(n_items, decay=0.6)   # bounded props -> stable IPW

    ids = np.repeat(np.arange(n_items), sessions)
    pos = np.repeat(slot, sessions)
    clicks = pb.simulate_clicks(true_rel[ids], pos, exam, rng)

    naive = pb.naive_relevance(clicks, ids, n_items)
    ipw = pb.ipw_relevance(clicks, pos, exam, ids, n_items)

    sp_naive = pb.spearman(naive, true_rel)
    sp_ipw = pb.spearman(ipw, true_rel)
    assert sp_ipw > 0.9                            # IPW nails the ranking
    assert sp_ipw > sp_naive + 0.2                 # and clearly beats naive CTR


def test_randomization_estimates_the_examination_curve():
    """A randomization bucket recovers the examination curve up to scale."""
    rng = np.random.default_rng(1)
    n_pos, sessions = 40, 4000
    exam = pb.examination_curve(n_pos, decay=1.2)
    tr = rng.uniform(0.2, 0.5, size=n_pos)   # n_items == n_pos here

    pos = np.concatenate([rng.permutation(n_pos) for _ in range(sessions)])
    ids = np.tile(np.arange(n_pos), sessions)
    clicks = pb.simulate_clicks(tr[ids], pos, exam, rng)

    est = pb.estimate_examination_randomized(clicks, pos, n_pos)
    assert pb.spearman(est, exam) > 0.9
