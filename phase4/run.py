"""
Phase 4 -- Entry Point
=======================
Runs the three monitoring layers against real data and produces a single
pipeline-gate status (PASS/FAIL). This is the code that would run on a schedule
(Airflow) and page someone when it goes red.

  Layer 1 (data health) : row-count volume, null rate, feature drift (PSI)
  Layer 2 (model health): fallback rate, recommendation diversity, calibration
  Layer 3 (business)    : engagement / target-action rate from the event stream

We deliberately look for DRIFT between the training window and the serving
(test) window -- because popularity genuinely shifts over time, this is where the
skew we measured in Phase 2 shows up as a monitorable signal.

Usage:
    cd phase4
    python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase3"))
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events, load_item_properties
from evaluate import temporal_split
from signals import POSITIVE_SIGNALS, MEDIUM, STRONG, TARGET_SIGNAL

from feature_store import PointInTimeFeatureStore
from lr_ranker import LRRanker
from training import build_labelled_features

from candidate_generator import PopularityCandidateGenerator
from service import RecommendationService, RecommendationRequest

from monitors import (
    Monitor,
    check_row_count, check_null_rate, check_drift,
    check_fallback_rate, check_diversity, check_calibration,
    expected_calibration_error,
)
from results_io import save_results

PHASE_DIR = Path(__file__).parent

SEED = 42
DAY_MS = 86_400 * 1000
POSITIVE = list(POSITIVE_SIGNALS)


def build_test_labelled(test_events, store, rng, cap=50_000):
    """Serving-time (online) features + labels for the TEST window -> calibration."""
    strong = test_events.filter(pl.col("event_type").is_in(POSITIVE))
    if len(strong) > cap:
        strong = strong.sample(cap, seed=SEED)
    all_items = test_events["item_id"].unique().to_list()
    pos = strong.select(["user_id", "item_id"]).with_columns(pl.lit(1).alias("label"))
    neg_items = rng.choice(np.array(all_items), size=len(pos))
    neg = pos.select(["user_id"]).with_columns(
        pl.Series("item_id", neg_items).cast(pl.Utf8), pl.lit(0).alias("label"),
    ).select(["user_id", "item_id", "label"])
    entity = pl.concat([pos, neg])
    return store.get_online_features_batch(entity)


def main():
    print("\n=== Phase 4: Monitoring & Drift Detection ===\n")
    rng = np.random.default_rng(SEED)

    print("Step 1/4: Loading data + fitting store/ranker...")
    events = load_events()
    item_props = load_item_properties()
    train_events, test_events, cutoff_ms = temporal_split(events, train_fraction=0.8)
    store = PointInTimeFeatureStore().fit(train_events, cutoff_timestamp_ms=cutoff_ms, item_properties=item_props)
    ranker = LRRanker().fit(build_labelled_features(train_events, store, rng))
    cg = PopularityCandidateGenerator(store._item_totals, pool_size=500)
    service = RecommendationService(cg, store, ranker=ranker, model_version="lr_ranker_v1")

    # ---- Layer 1: data health --------------------------------------------
    print("\nStep 2/4: Layer 1 -- data health...")
    # Equal-length 7-day windows on either side of the cutoff (apples-to-apples).
    ref_win = train_events.filter(pl.col("timestamp_ms") >= cutoff_ms - 7 * DAY_MS)
    cur_win = test_events.filter(pl.col("timestamp_ms") < cutoff_ms + 7 * DAY_MS)

    null_count = sum(cur_win[c].null_count() for c in ["user_id", "item_id", "timestamp_ms"])

    # Feature drift: item_pop (as-of-cutoff) distribution, ref vs current window.
    ref_feat = store.get_online_features_batch(
        ref_win.filter(pl.col("event_type").is_in(POSITIVE)).select(["user_id", "item_id"])
    )["item_pop"].to_numpy()
    cur_feat = store.get_online_features_batch(
        cur_win.filter(pl.col("event_type").is_in(POSITIVE)).select(["user_id", "item_id"])
    )["item_pop"].to_numpy()

    layer1 = [
        check_row_count(cur_win.height, ref_win.height, min_ratio=0.8),
        check_null_rate(null_count, cur_win.height, max_rate=0.01),
        check_drift(ref_feat, cur_feat, max_psi=0.2),
    ]
    print(Monitor(layer1).report())

    # ---- Layer 2: model health -------------------------------------------
    print("\nStep 3/4: Layer 2 -- model health...")
    item_cat = store._item_category  # item_id -> category_id

    sample_users = test_events.filter(pl.col("event_type") == TARGET_SIGNAL)["user_id"].unique().to_list()
    rng.shuffle(sample_users)
    sample_users = sample_users[:2000]

    fallbacks, diversities = 0, []
    for uid in sample_users:
        resp = service.recommend(RecommendationRequest(user_id=uid, n=20))
        fallbacks += int(resp.fallback_used)
        cats = (
            pl.DataFrame({"item_id": resp.items})
            .join(item_cat, on="item_id", how="left")["category_id"]
            .drop_nulls().n_unique()
        )
        diversities.append(cats)

    # Calibration on serving-time features/labels from the test window.
    test_lab = build_test_labelled(test_events, store, rng)
    pred = ranker.predict_proba(test_lab)
    actual = test_lab["label"].to_numpy().astype(float)
    ece = expected_calibration_error(pred, actual)

    layer2 = [
        check_fallback_rate(fallbacks, len(sample_users), max_rate=0.05),
        check_diversity(float(np.mean(diversities)), min_categories=3.0),
        check_calibration(ece, max_ece=0.1),
    ]
    print(Monitor(layer2).report())

    # ---- Layer 3: business metrics ---------------------------------------
    print("\nStep 4/4: Layer 3 -- business metrics (from the event stream)...")
    n_test = test_events.height
    eng = test_events.filter(pl.col("event_type") == MEDIUM).height
    conv = test_events.filter(pl.col("event_type") == STRONG).height
    print(f"  engagement rate (MEDIUM) : {eng / n_test:.4f}  ({eng:,} / {n_test:,} events)")
    print(f"  target-action rate (STRONG): {conv / n_test:.4f}  ({conv:,} / {n_test:,} events)")
    print("  (In production these stream from Kafka into ClickHouse/Grafana in real time.)")

    # ---- Overall gate -----------------------------------------------------
    overall = Monitor(layer1 + layer2)
    print("\n" + "=" * 55)
    print(f"  OVERALL PIPELINE GATE: {overall.gate_status}")
    print("=" * 55)
    if not overall.passed:
        print("  A red gate here would block the deploy and page on-call.")
        print("  Note: feature drift between train and serve windows is EXPECTED")
        print("  on this dataset -- popularity shifts over time. That is exactly")
        print("  the signal monitoring exists to surface (Rule 10).")
    print("\nNext steps:")
    print("  - Wire these checks into an Airflow DAG; alert via PagerDuty on FAIL.")
    print("  - Use Evidently AI for richer drift; Great Expectations for data asserts.")
    print("  - Phase 5: A/B test lr_ranker_v1 vs v2 using the model_version tag.\n")
    save_results(PHASE_DIR, {
        "gate_status": overall.gate_status,
        "gate_passed": bool(overall.passed),
        "calibration_ece": float(ece),
        "mean_diversity": float(np.mean(diversities)),
        "fallback_rate": float(fallbacks / len(sample_users)) if sample_users else 0.0,
        "row_count_ratio": float(cur_win.height / ref_win.height) if ref_win.height else 0.0,
        "engagement_rate": float(eng / n_test) if n_test else 0.0,
        "target_action_rate": float(conv / n_test) if n_test else 0.0,
    })
    return overall


if __name__ == "__main__":
    main()
