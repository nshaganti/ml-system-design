"""
Phase 0 — Evaluation
======================
Evaluates the heuristic ranker using a temporal split (Rule 33).

WHY temporal split matters:
  A random 80/20 split lets your model "see the future" — test events that
  happened before some training events. In production the model never sees
  future data, so random splits produce optimistic metrics that don't reflect
  real performance. Always split by time.

Metrics:
  - Recall@K: of all items a user purchased in the test period,
               what fraction appear in the top-K recommendations?
  - Coverage:  what % of catalog items ever appear in any recommendation?
               Low coverage = the model is a popularity trap.
"""

from __future__ import annotations

import polars as pl
from heuristic_ranker import HeuristicRanker


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
    Recall@K: for each user who made a purchase in the test period,
    did our top-K recommendations include that item?

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
    # Only evaluate on users with purchases in test period
    test_purchasers = (
        test_events
        .filter(pl.col("event_type") == "purchase")
        .group_by("user_id")
        .agg(pl.col("item_id").alias("purchased_items"))
    )

    if len(test_purchasers) == 0:
        raise ValueError("No purchase events in test set. Check your data and split.")

    # Cap for speed
    if len(test_purchasers) > max_users:
        test_purchasers = test_purchasers.sample(max_users, seed=42)

    print(f"\n[evaluate] Scoring {len(test_purchasers):,} users with test purchases...")

    hits        = 0
    total       = 0
    cold_hits   = 0
    cold_total  = 0
    warm_hits   = 0
    warm_total  = 0
    all_recommended = set()

    for row in test_purchasers.iter_rows(named=True):
        user_id         = row["user_id"]
        purchased_items = set(row["purchased_items"])

        # User history = their events in TRAINING data only (no future leakage)
        user_train_events = train_events.filter(pl.col("user_id") == user_id)
        is_cold_start = len(user_train_events) == 0

        recommendations = ranker.recommend(
            user_id=user_id,
            user_events=user_train_events,
            n=k,
        )
        all_recommended.update(recommendations)

        # Count how many purchased items appear in top-K recommendations
        n_hits = len(purchased_items & set(recommendations))

        hits       += n_hits
        total      += len(purchased_items)

        if is_cold_start:
            cold_hits  += n_hits
            cold_total += len(purchased_items)
        else:
            warm_hits  += n_hits
            warm_total += len(purchased_items)

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
        "total_purchases":   total,
        "total_hits":        hits,
        "cold_start_recall": cold_recall,
        "warm_user_recall":  warm_recall,
        "catalog_coverage":  coverage,
        "users_evaluated":   len(test_purchasers),
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
    print(f"  Warm users           : {r['warm_user_recall']:.4f}")
    print(f"  Cold-start users     : {r['cold_start_recall']:.4f}")
    print(f"  Catalog coverage     : {r['catalog_coverage']:.4f}  ({r['catalog_coverage']*100:.1f}%)")
    print(f"  Users evaluated      : {r['users_evaluated']:,}")
    print(f"  Purchases evaluated  : {r['total_purchases']:,}")
    print()
    print("  What this means:")
    print(f"    For every 100 items users bought, our heuristic")
    print(f"    put {r['recall_at_k']*100:.1f} of them in the top-{r['k']} recommendations.")
    print()
    print("  What to expect in Phase 1 (first ML model):")
    print("    A simple two-tower model typically 2-3x this recall.")
    print("    If Phase 1 doesn't beat this, the model isn't learning — debug features first.")
    print("=" * 50)
