"""
Phase 11 -- Position-Bias Debiasing (controlled simulation)
============================================================
Clicks are not relevance. An item clicked at rank 1 got a huge examination boost an
item at rank 40 never had -- so naive click-through rates rank *positions* as much
as items. This is the single most important bias in learning-to-rank from click
logs, and correcting it is the Position-Based Model (PBM) + inverse-propensity
weighting (IPW).

HONESTY NOTE: KuaiRand-Pure does not log the on-screen POSITION of each impression
(its `tab` field is a feed id, not a rank). So -- unlike every other phase, which
runs on the real logs -- this phase is a *controlled simulation*. We generate
clicks from a known examination curve and known relevance so we can prove the
estimators recover the truth. Treat it as the physics-lab demo of the technique;
the math is exactly what you'd deploy on logs that DO carry position.

The Position-Based Model:
    P(click | item, position) = P(examine | position) * P(relevant | item)
                              =        e_p            *        r_i

Given clicks, naive CTR estimates `e_p * r_i` (confounded). If we KNOW `e_p`
(the propensity), IPW divides it out and recovers `r_i`. We also show how to
ESTIMATE `e_p` from a small result-randomization bucket -- the same "randomize to
learn the logging policy" idea that powers Part II's random log.

Pure numeric core (Rule 5): arrays in, numbers out. `run.py` wires the scenario.
"""

from __future__ import annotations

import numpy as np


def examination_curve(n_positions: int, decay: float = 1.0) -> np.ndarray:
    """
    P(examine | position) for positions 0..n_positions-1, normalized so the top
    slot is 1.0. Uses the standard inverse decay e_p = 1 / (1 + p)^decay -- higher
    `decay` = attention falls off faster down the page.
    """
    p = np.arange(n_positions, dtype=np.float64)
    e = 1.0 / np.power(1.0 + p, decay)
    return e / e[0]


def simulate_clicks(
    item_relevance: np.ndarray, positions: np.ndarray, exam: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw a click per impression from the PBM: click ~ Bernoulli(e_p * r_i).
    `item_relevance` and `positions` are per-impression arrays; `exam` is indexed
    by position. Returns a 0/1 click array.
    """
    item_relevance = np.asarray(item_relevance, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    p_click = exam[positions] * item_relevance
    return (rng.random(p_click.shape) < p_click).astype(np.float64)


def _grouped_mean(values: np.ndarray, item_ids: np.ndarray, n_items: int) -> np.ndarray:
    """Mean of `values` per item id (0..n_items-1); 0.0 for items with no rows."""
    sums = np.zeros(n_items, dtype=np.float64)
    counts = np.zeros(n_items, dtype=np.float64)
    np.add.at(sums, item_ids, values)
    np.add.at(counts, item_ids, 1.0)
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def naive_relevance(clicks: np.ndarray, item_ids: np.ndarray, n_items: int) -> np.ndarray:
    """Plain CTR per item -- the confounded estimate everyone reaches for first."""
    return _grouped_mean(np.asarray(clicks, dtype=np.float64), np.asarray(item_ids), n_items)


def ipw_relevance(
    clicks: np.ndarray, positions: np.ndarray, exam: np.ndarray,
    item_ids: np.ndarray, n_items: int,
) -> np.ndarray:
    """
    IPW estimate: per item, mean of click / e_p. Dividing each click by the
    examination propensity of the slot it was shown in removes position bias, so
    this recovers true relevance `r_i` in expectation.
    """
    positions = np.asarray(positions, dtype=np.int64)
    weighted = np.asarray(clicks, dtype=np.float64) / exam[positions]
    return _grouped_mean(weighted, np.asarray(item_ids), n_items)


def estimate_examination_randomized(
    clicks: np.ndarray, positions: np.ndarray, n_positions: int,
) -> np.ndarray:
    """
    Estimate the examination curve from a RESULT-RANDOMIZATION bucket, where items
    are placed in random positions. Because relevance averages out across randomly
    assigned items, mean CTR at position p is proportional to e_p. Normalize to the
    top slot to get a propensity curve (scale is irrelevant for ranking).
    """
    ctr_by_pos = _grouped_mean(
        np.asarray(clicks, dtype=np.float64), np.asarray(positions, dtype=np.int64), n_positions
    )
    if ctr_by_pos[0] <= 0:
        return ctr_by_pos
    return ctr_by_pos / ctr_by_pos[0]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman rank correlation (Pearson on ranks), no scipy. 1.0 = identical
    ordering. We grade recovered relevance by how well its RANKING matches truth --
    ranking is what a recommender ultimately acts on.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2))
    return float(np.sum(ra * rb) / denom) if denom > 0 else 0.0


def topk_recovery(estimate: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Fraction of the true top-k items that the estimate also ranks in its top-k."""
    est_top = set(np.argsort(estimate)[::-1][:k].tolist())
    true_top = set(np.argsort(truth)[::-1][:k].tolist())
    return len(est_top & true_top) / k if k > 0 else 0.0
