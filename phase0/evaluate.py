"""
Phase 0 — Evaluation
======================
Evaluates the heuristic ranker using a temporal split (Rule 33).

WHY temporal split matters:
  A random 80/20 split lets your model "see the future" -- test events that
  happened before some training events. In production the model never sees
  future data, so random splits produce optimistic metrics that don't reflect
  real performance. Always split by time.

Metrics:
  - Recall@K: of all items a user engaged with in the test period,
               what fraction appear in the top-K recommendations?
  - Coverage:  what % of catalog items ever appear in any recommendation?
               Low coverage = the model is a popularity trap.
"""

from __future__ import annotations

import polars as pl
from heuristic_ranker import HeuristicRanker
from metrics import ndcg_at_k, average_precision_at_k, precision_at_k, mean
from signals import POSITIVE_SIGNALS


def temporal_split(
    events: pl.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """
    Split events into train/test by time (not randomly).

    Returns:
        train_events, test_events, cutoff_timestamp_ms
    """
    timestamps = events["timestamp_ms"].sort()
    n = len(timestamps)
    cutoff_idx = int(n * train_fraction)
    cutoff_ms = int(timestamps[cutoff_idx])

    train = events.filter(pl.col("timestamp_ms") < cutoff_ms)
    test  = events.filter(pl.col("timestamp_ms") >= cutoff_ms)

    print(f"[evaluate] Temporal split at {train_fraction:.0%}")
    print(f"  Train: {len(train):,} events  ({train['user_id'].n_unique():,} users)")
    print(f"  Test:  {len(test):,} events  ({test['user_id'].n_unique():,} users)")
    print(f"  Cutoff timestamp: {cutoff_ms:,} ms")
    return train, test, cutoff_ms


def recall_at_k(
    ranker: HeuristicRanker,
    train_events: pl.DataFrame,
    test_events: pl.DataFrame,
    k: int = 20,
    max_users: int = 5000,
    catalog_size: int | None = None,
) -> dict:
    """
    Recall@K: for each user with a positive signal (target action) in the test
    period, did our top-K recommendations include that item?

    max_users: cap evaluation to this many users (speed).
               Results are statistically stable at 5K users.

    catalog_size: denominator for the coverage metric — the size of the
                  recommendable universe. Pass the SAME value across models
                  (e.g. train_events['item_id'].n_unique()) so coverage is a
                  fair apples-to-apples comparison. Each model's own
                  ranker.catalog_size uses a different denominator (popularity
                  catalog vs. embedding vocab), which makes cross-model coverage
                  meaningless. If None, falls back to ranker.catalog_size.

    Returns a dict with recall@k, coverage, and cold/warm breakdown.
    """
    # Only evaluate on users with a positive signal in the test period
    test_positives = (
        test_events
        .filter(pl.col("event_type").is_in(list(POSITIVE_SIGNALS)))
        .group_by("user_id")
        .agg(pl.col("item_id").alias("positive_items"))
        # group_by order is NOT stable; without this sort the seeded .sample()
        # below draws a DIFFERENT 5,000-user subset each run -> recall drifts
        # even when the model is identical. Pin the order first.
        .sort("user_id")
    )

    if len(test_positives) == 0:
        raise ValueError("No positive-signal events in test set. Check your data and split.")

    # Cap for speed
    if len(test_positives) > max_users:
        test_positives = test_positives.sample(max_users, seed=42)

    print(f"\n[evaluate] Scoring {len(test_positives):,} users with test positives...")

    hits        = 0
    total       = 0
    cold_hits   = 0
    cold_total  = 0
    warm_hits   = 0
    warm_total  = 0
    all_recommended = set()

    # Rank-aware metrics accumulated per user, then averaged (macro average).
    ndcg_scores: list[float] = []
    ap_scores:   list[float] = []
    prec_scores: list[float] = []

    for row in test_positives.iter_rows(named=True):
        user_id        = row["user_id"]
        positive_items = set(row["positive_items"])

        # User history = their events in TRAINING data only (no future leakage)
        user_train_events = train_events.filter(pl.col("user_id") == user_id)
        is_cold_start = len(user_train_events) == 0

        recommendations = ranker.recommend(
            user_id=user_id,
            user_events=user_train_events,
            n=k,
        )
        all_recommended.update(recommendations)

        # Count how many positive items appear in top-K recommendations
        n_hits = len(positive_items & set(recommendations))

        hits       += n_hits
        total      += len(positive_items)

        # Rank-aware metrics (order matters, unlike raw recall)
        ndcg_scores.append(ndcg_at_k(recommendations, positive_items, k))
        ap_scores.append(average_precision_at_k(recommendations, positive_items, k))
        prec_scores.append(precision_at_k(recommendations, positive_items, k))

        if is_cold_start:
            cold_hits  += n_hits
            cold_total += len(positive_items)
        else:
            warm_hits  += n_hits
            warm_total += len(positive_items)

    recall         = hits / total if total > 0 else 0.0
    cold_recall    = cold_hits / cold_total if cold_total > 0 else 0.0
    warm_recall    = warm_hits / warm_total if warm_total > 0 else 0.0
    # Use the shared catalog_size denominator for a fair cross-model comparison.
    # Fall back to the ranker's own count only when no explicit size is given.
    denom          = catalog_size if catalog_size is not None else ranker.catalog_size
    coverage       = len(all_recommended) / denom if denom > 0 else 0.0

    results = {
        "recall_at_k":       recall,
        "k":                 k,
        "total_positives":   total,
        "total_hits":        hits,
        "cold_start_recall": cold_recall,
        "warm_user_recall":  warm_recall,
        "catalog_coverage":  coverage,
        "ndcg_at_k":         mean(ndcg_scores),
        "map_at_k":          mean(ap_scores),
        "precision_at_k":    mean(prec_scores),
        "users_evaluated":   len(test_positives),
    }

    # Label the printout with the ranker under test so Phase 1's output isn't
    # mislabelled "HEURISTIC BASELINE".
    _print_results(results, label=type(ranker).__name__)
    return results


def _print_results(r: dict, label: str = "Ranker") -> None:
    print()
    print("=" * 50)
    print(f"  {label} — Recall@{r['k']}")
    print("=" * 50)
    print(f"  Overall Recall@{r['k']}  : {r['recall_at_k']:.4f}  ({r['recall_at_k']*100:.2f}%)")
    print(f"  NDCG@{r['k']}           : {r['ndcg_at_k']:.4f}")
    print(f"  MAP@{r['k']}            : {r['map_at_k']:.4f}")
    print(f"  Precision@{r['k']}      : {r['precision_at_k']:.4f}")
    print(f"  Warm users           : {r['warm_user_recall']:.4f}")
    print(f"  Cold-start users     : {r['cold_start_recall']:.4f}")
    print(f"  Catalog coverage     : {r['catalog_coverage']:.4f}  ({r['catalog_coverage']*100:.1f}%)")
    print(f"  Users evaluated      : {r['users_evaluated']:,}")
    print(f"  Positives evaluated  : {r['total_positives']:,}")
    print()
    print("  What this means:")
    print(f"    For every 100 items users engaged with, our ranker")
    print(f"    put {r['recall_at_k']*100:.1f} of them in the top-{r['k']} recommendations.")
    print()
    print("  What to expect in Phase 1 (first ML model):")
    print("    A simple two-tower model typically 2-3x this recall.")
    print("    If Phase 1 doesn't beat this, the model isn't learning — debug features first.")
    print("=" * 50)
