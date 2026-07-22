"""
Tests for Phase 20 -- joint EM position-bias/relevance estimation (joint_em.py).

Deterministic checks on a controlled PBM world: EM's log-likelihood is monotone,
its recovered examination curve and relevance RANKING match the truth, and it beats
naive CTR on a confounded log -- all with no randomization bucket.
"""

from __future__ import annotations

import numpy as np

from position_bias import (
    examination_curve, simulate_clicks, naive_relevance, spearman, topk_recovery,
)
from joint_em import regression_em_pbm


def _confounded_log(rng, n_items=40, n_pos=8, sessions=8000, noise=0.05, align=0.35):
    true_rel = rng.uniform(0.1, 0.9, n_items)
    true_exam = examination_curve(n_pos, decay=1.3)
    # OLD ranker's stale score: partly aligned with relevance, partly noise -> position
    # confounding that actually distorts naive CTR's RANKING.
    z = (true_rel - true_rel.mean()) / (true_rel.std() + 1e-9)
    logging_score = align * z + (1 - align) * rng.normal(0, 1, n_items)
    picks = np.argsort(rng.random((sessions, n_items)), axis=1)[:, :n_pos]
    scores = logging_score[picks] + rng.normal(0, noise, picks.shape)
    picks = np.take_along_axis(picks, np.argsort(-scores, axis=1), axis=1)
    items = picks.reshape(-1)
    positions = np.tile(np.arange(n_pos), sessions)
    clicks = simulate_clicks(true_rel[items], positions, true_exam, rng)
    return clicks, positions, items, true_rel, true_exam, n_items, n_pos


def test_loglik_is_monotone_nondecreasing():
    rng = np.random.default_rng(0)
    clicks, pos, items, _, _, n_items, n_pos = _confounded_log(rng)
    out = regression_em_pbm(clicks, pos, items, n_items, n_pos, n_iter=40, seed=0)
    ll = out["loglik"]
    assert all(ll[i + 1] >= ll[i] - 1e-9 for i in range(len(ll) - 1))


def test_examination_is_anchored_and_decreasing():
    rng = np.random.default_rng(1)
    clicks, pos, items, _, true_exam, n_items, n_pos = _confounded_log(rng)
    out = regression_em_pbm(clicks, pos, items, n_items, n_pos, n_iter=60, seed=1)
    assert np.isclose(out["exam"][0], 1.0)               # anchored
    assert out["exam"][0] > out["exam"][-1]              # top slot examined more
    assert np.mean(np.abs(out["exam"] - true_exam)) < 0.1


def test_em_recovers_relevance_ranking_and_beats_naive():
    rng = np.random.default_rng(2)
    clicks, pos, items, true_rel, _, n_items, n_pos = _confounded_log(rng)
    out = regression_em_pbm(clicks, pos, items, n_items, n_pos, n_iter=60, seed=2)
    naive = naive_relevance(clicks, items, n_items)
    em_sp = spearman(out["rel"], true_rel)
    naive_sp = spearman(naive, true_rel)
    assert em_sp > 0.9                                    # strong recovery
    assert em_sp > naive_sp                               # beats the confounded baseline
    assert topk_recovery(out["rel"], true_rel, 10) >= topk_recovery(naive, true_rel, 10)


def test_converges_before_max_iter_on_easy_data():
    rng = np.random.default_rng(3)
    clicks, pos, items, _, _, n_items, n_pos = _confounded_log(rng, sessions=12000)
    out = regression_em_pbm(clicks, pos, items, n_items, n_pos, n_iter=400, tol=1e-3, seed=3)
    assert out["n_iter_run"] < 400                        # tol reached, not just capped
