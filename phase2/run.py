"""
Phase 2 -- Entry Point
=======================
Logistic-regression ranker on point-in-time-correct features, plus a concrete
demonstration of training-serving skew (the design doc's headline concept).

Pipeline:
  1. Load data + temporal split (reuse phase0)
  2. Fit the point-in-time feature store on training events
  3. Build a labelled training set (strong events = positive, sampled = negative)
  4. Train the LR ranker on POINT-IN-TIME-correct features
  5. Quantify the skew: how different are point-in-time vs "join today's totals"?
  6. Evaluate the ranker on test engagers vs a popularity baseline
     (candidate pool = top popular items; the ranker reorders them)

Usage:
    cd phase2
    python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events, load_item_properties
from evaluate import temporal_split
from signals import TARGET_SIGNAL
from metrics import ndcg_at_k, average_precision_at_k, recall_at_k, mean

from feature_store import PointInTimeFeatureStore, FEATURE_COLUMNS
from lr_ranker import LRRanker
from training import build_labelled_features
from results_io import save_results

PHASE_DIR = Path(__file__).parent

MAX_POSITIVES   = 100_000   # cap training rows for speed
CANDIDATE_POOL  = 500       # popularity candidate pool the ranker reorders
K               = 20
MAX_EVAL_USERS  = 5000
SEED            = 42


def build_training_set(
    train_events: pl.DataFrame,
    store: PointInTimeFeatureStore,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Positives = strong events; negatives = a random item at the same time."""
    return build_labelled_features(train_events, store, rng, max_positives=MAX_POSITIVES, seed=SEED)


def measure_skew(entity_df: pl.DataFrame, store: PointInTimeFeatureStore) -> None:
    """Compare point-in-time features to the naive 'join today's totals' bug."""
    base = entity_df.select(["user_id", "item_id", "timestamp_ms", "label"])
    correct = store.get_historical_features(base)
    skewed = store.get_skewed_features(base)

    print("\n" + "=" * 55)
    print("  TRAINING-SERVING SKEW (point-in-time vs leaked)")
    print("=" * 55)
    for col in FEATURE_COLUMNS:
        c = correct[col].to_numpy().astype(float)
        s = skewed[col].to_numpy().astype(float)
        mad = float(np.mean(np.abs(c - s)))
        infl = (s.mean() / c.mean()) if c.mean() > 0 else float("nan")
        print(f"  {col:<10} point-in-time mean={c.mean():8.2f} | "
              f"leaked mean={s.mean():8.2f} | inflation x{infl:4.1f} | MAD={mad:6.2f}")
    print("  ^ The leaked ('join today') features are systematically inflated:")
    print("    old events get credited with popularity they only earned later.")
    print("    Train on that and offline eval lies; production degrades (Rules 29-37).")
    print("=" * 55)


def evaluate_ranker(
    ranker: LRRanker,
    train_events: pl.DataFrame,
    test_events: pl.DataFrame,
    store: PointInTimeFeatureStore,
) -> dict:
    """Reorder a popularity candidate pool with the LR ranker; score vs target actions."""
    # Candidate pool = most popular items as of the cutoff (Phase 1 would supply
    # these via two-tower ANN; popularity is a fine stand-in for the ranker demo).
    pool = (
        store._item_totals.sort("item_pop", descending=True)
        .head(CANDIDATE_POOL)["item_id"].to_list()
    )
    pool_df = pl.DataFrame({"item_id": pool})

    test_positives = (
        test_events.filter(pl.col("event_type") == TARGET_SIGNAL)
        .group_by("user_id")
        .agg(pl.col("item_id").alias("positive_items"))
    )
    if len(test_positives) > MAX_EVAL_USERS:
        test_positives = test_positives.sample(MAX_EVAL_USERS, seed=SEED)

    lr_ndcg, lr_recall, pop_ndcg, pop_recall = [], [], [], []

    for row in test_positives.iter_rows(named=True):
        user_id = row["user_id"]
        positive = set(row["positive_items"])

        # Online features for this user across the whole pool. The cross feature
        # (user_cat_affinity) varies per candidate, so the LR can now PERSONALIZE.
        cand = pool_df.with_columns(pl.lit(user_id).alias("user_id"))
        cand = store.get_online_features_batch(cand)

        lr_top = ranker.rank(cand, item_col="item_id", n=K)
        pop_top = pool[:K]  # popularity baseline = pool order

        lr_ndcg.append(ndcg_at_k(lr_top, positive, K))
        lr_recall.append(recall_at_k(lr_top, positive, K))
        pop_ndcg.append(ndcg_at_k(pop_top, positive, K))
        pop_recall.append(recall_at_k(pop_top, positive, K))

    return {
        "lr_ndcg": mean(lr_ndcg), "lr_recall": mean(lr_recall),
        "pop_ndcg": mean(pop_ndcg), "pop_recall": mean(pop_recall),
        "users": len(test_positives),
    }


def main():
    print("\n=== Phase 2: LR Ranker + Point-in-Time Feature Store ===\n")
    rng = np.random.default_rng(SEED)

    print("Step 1/5: Loading data...")
    events = load_events()
    item_props = load_item_properties()

    print("\nStep 2/5: Temporal split...")
    train_events, test_events, cutoff_ms = temporal_split(events, train_fraction=0.8)

    print("\nStep 3/5: Fitting point-in-time feature store...")
    store = PointInTimeFeatureStore().fit(
        train_events, cutoff_timestamp_ms=cutoff_ms, item_properties=item_props
    )

    print("\nStep 4/5: Building training set + training LR ranker...")
    training_df = build_training_set(train_events, store, rng)
    ranker = LRRanker().fit(training_df)
    measure_skew(training_df, store)

    print("\nStep 5/5: Evaluating ranker vs popularity baseline...")
    res = evaluate_ranker(ranker, train_events, test_events, store)

    print("\n" + "=" * 55)
    print("  PHASE 2 RANKER vs POPULARITY (same candidate pool)")
    print("=" * 55)
    print(f"  {'Metric':<16} {'Popularity':>12} {'LR ranker':>12}")
    print(f"  {'-'*42}")
    print(f"  {'Recall@'+str(K):<16} {res['pop_recall']:>12.4f} {res['lr_recall']:>12.4f}")
    print(f"  {'NDCG@'+str(K):<16} {res['pop_ndcg']:>12.4f} {res['lr_ndcg']:>12.4f}")
    print(f"  users evaluated : {res['users']:,}")
    print("=" * 55)
    lift = (res['lr_ndcg'] / res['pop_ndcg'] - 1) * 100 if res['pop_ndcg'] > 0 else 0.0
    if lift >= 0:
        print(f"  The user_cat_affinity CROSS feature lets the ranker PERSONALIZE:")
        print(f"  it reorders the same popular pool per user by category affinity,")
        print(f"  giving {lift:+.0f}% NDCG vs raw popularity order (Rule 20).")
    else:
        print(f"  Honest result: the user_cat_affinity cross feature gives {lift:+.0f}% NDCG")
        print(f"  vs raw popularity -- i.e. it HURTS here. On this dataset category is a")
        print(f"  weak personalization signal (coarse tags, popularity-driven engagement),")
        print(f"  so a one-feature LR can't beat the popularity prior. That's the lesson,")
        print(f"  not a bug: a feature only helps if it carries signal (Rules 17 & 20).")
        print(f"  The point-in-time store + skew audit still matter regardless of lift.")
    print("\nNext steps:")
    print("  - Feed the LR ranker the two-tower's 500 candidates (Phase 1) instead")
    print("    of the popularity pool for the full two-stage architecture.")
    print("  - Add more cross features (price vs user avg, brand affinity, recency).")
    print("  - Upgrade LR -> XGBoost only if it plateaus; move features to Redis.\n")
    save_results(PHASE_DIR, res)
    return res


if __name__ == "__main__":
    main()
