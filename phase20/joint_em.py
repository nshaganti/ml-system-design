"""
Phase 20 -- Joint Position-Bias + Relevance Estimation via EM (Regression-EM / DLA)
===================================================================================
Phase 11 removed position bias with inverse-propensity weighting -- but it needed the
examination curve `e_p` handed to it, learned from a RESULT-RANDOMIZATION bucket
(showing items in random slots). Randomization costs real revenue, so you can't always
run it. This phase closes that gap: estimate the examination curve AND per-item
relevance *jointly, from ordinary (biased) production click logs*, with no
randomization at all.

The Position-Based Model again:
    P(click | item i, position p) = e_p * r_i          (click needs BOTH examine AND relevant)

Given only clicks, `e_p` and `r_i` are confounded -- but they are jointly identifiable
(up to scale) as long as items appear across a RANGE of positions. Regression-EM
(Wang et al., 2018; the estimation heart of the Dual Learning Algorithm, Ai et al.,
2018) recovers them by alternating:

  E-step -- for each impression, infer the posterior of the latent examine/relevant
            bits given the click and the current e_p, r_i estimates.
  M-step -- re-estimate e_p as the mean posterior examination at each position, and
            r_i as the mean posterior relevance for each item.

Anchor e_0 = 1 each round (fixes the unidentifiable global scale). The recovered r_i
are exactly the debiased labels you'd feed a feature-based ranker (Phase 2).

Pure NumPy core (Rule 5). Reuses Phase 11's PBM primitives for DRY.
"""

from __future__ import annotations

import numpy as np

from position_bias import _grouped_mean     # Phase 11 helper (DRY)


def regression_em_pbm(
    clicks: np.ndarray,
    positions: np.ndarray,
    item_ids: np.ndarray,
    n_items: int,
    n_positions: int,
    n_iter: int = 50,
    tol: float = 1e-6,
    seed: int = 0,
) -> dict:
    """
    Jointly estimate the examination curve e_p and per-item relevance r_i from a
    click log via EM on the Position-Based Model. Returns a dict with:
        exam  -- (n_positions,) estimated examination, anchored so exam[0] == 1
        rel   -- (n_items,)     estimated relevance
        loglik -- list of per-iteration average log-likelihoods (monotone up)
        n_iter_run -- iterations actually run before convergence

    A click (c=1) forces examine=1 AND relevant=1. For a non-click (c=0) the posterior
    that the slot WAS examined / the item WAS relevant is:
        P(E=1 | c=0) = e_p (1 - r_i) / (1 - e_p r_i)
        P(R=1 | c=0) = (1 - e_p) r_i / (1 - e_p r_i)
    """
    clicks = np.asarray(clicks, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    item_ids = np.asarray(item_ids, dtype=np.int64)

    rng = np.random.default_rng(seed)
    exam = np.linspace(1.0, 0.3, n_positions)                 # rough monotone init
    rel = rng.uniform(0.2, 0.6, n_items)                       # neutral init

    loglik_hist: list[float] = []
    run = 0
    for it in range(n_iter):
        run = it + 1
        e_p = exam[positions]                                  # per-impression e
        r_i = rel[item_ids]                                    # per-impression r
        pc = np.clip(e_p * r_i, 1e-9, 1 - 1e-9)               # P(click)

        # E-step: posterior latent bits.
        denom = np.clip(1.0 - pc, 1e-9, None)
        p_exam = np.where(clicks > 0, 1.0, e_p * (1.0 - r_i) / denom)
        p_rel = np.where(clicks > 0, 1.0, (1.0 - e_p) * r_i / denom)

        # M-step: means of the posterior bits, grouped by position / item.
        exam_new = _grouped_mean(p_exam, positions, n_positions)
        rel_new = _grouped_mean(p_rel, item_ids, n_items)
        exam_new = np.clip(exam_new, 1e-6, 1.0)
        rel_new = np.clip(rel_new, 1e-6, 1.0)
        if exam_new[0] > 0:                                    # anchor the scale
            exam_new = exam_new / exam_new[0]

        # Observed-data log-likelihood (monotone non-decreasing under EM).
        ll = float(np.mean(clicks * np.log(pc) + (1 - clicks) * np.log(1 - pc)))
        loglik_hist.append(ll)

        shift = max(np.max(np.abs(exam_new - exam)), np.max(np.abs(rel_new - rel)))
        exam, rel = exam_new, rel_new
        if shift < tol:
            break

    return {"exam": exam, "rel": rel, "loglik": loglik_hist, "n_iter_run": run}
