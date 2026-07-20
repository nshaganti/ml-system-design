"""
Phase 5 -- Entry Point
=======================
Runs an offline REPLAY A/B test between two ranker variants and analyzes it with
the same statistics you'd use on live traffic.

  control   = popularity ordering        (service with ranker=None -> candidate order)
  treatment = lr_ranker_v1               (service with the Phase 2 LR ranker)
  metric    = hit@20 (did any of the user's actual test-window target actions land
              in the top-20?) -- a per-user binary "conversion" proxy

Important honesty: this is a REPLAY on logged target actions, not a live test -- we
can't observe how users would react to recommendations they never saw. But the
machinery on display -- deterministic sticky assignment, a two-proportion z-test,
confidence intervals, and up-front sample-size planning -- is exactly what you run
online. The point is the method, not the specific delta.

Usage:
    cd phase5
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
from signals import TARGET_SIGNAL

from feature_store import PointInTimeFeatureStore
from lr_ranker import LRRanker
from training import build_labelled_features

from candidate_generator import PopularityCandidateGenerator
from service import RecommendationService, RecommendationRequest

from experiment import assign_variant, two_proportion_ztest, required_sample_size

SEED = 42
EXPERIMENT = "ranker_popularity_vs_lr_v1"
K = 20
MAX_USERS = 15_000


def main():
    print("\n=== Phase 5: A/B Testing (Replay) ===\n")
    rng = np.random.default_rng(SEED)

    print("Step 1/5: Loading data + fitting store/ranker...")
    events = load_events()
    item_props = load_item_properties()
    train_events, test_events, cutoff_ms = temporal_split(events, train_fraction=0.8)
    store = PointInTimeFeatureStore().fit(train_events, cutoff_timestamp_ms=cutoff_ms, item_properties=item_props)
    ranker = LRRanker().fit(build_labelled_features(train_events, store, rng))
    cg = PopularityCandidateGenerator(store._item_totals, pool_size=500)

    control_svc = RecommendationService(cg, store, ranker=None, model_version="control_popularity")
    treatment_svc = RecommendationService(cg, store, ranker=ranker, model_version="lr_ranker_v1")

    # ---- Sample-size planning (do this BEFORE you look at results) --------
    print("\nStep 2/5: Experiment design...")
    baseline = 0.10   # rough expected hit@20 for planning
    for mde in (0.05, 0.10, 0.20):
        n = required_sample_size(baseline, mde)
        print(f"  to detect a {mde:.0%} relative lift (baseline {baseline:.0%}): "
              f"need ~{n:,} users PER ARM")

    # ---- Assignment sanity: sticky + balanced ----------------------------
    print("\nStep 3/5: Assignment checks...")
    demo_user = "user_12345"
    a1 = assign_variant(demo_user, EXPERIMENT, ["control", "treatment"])
    a2 = assign_variant(demo_user, EXPERIMENT, ["control", "treatment"])
    print(f"  sticky: {demo_user} -> {a1} then {a2} "
          f"({'STABLE' if a1 == a2 else 'BROKEN!'})")

    # ---- Run the replay --------------------------------------------------
    print("\nStep 4/5: Running the replay...")
    purchases = (
        test_events.filter(pl.col("event_type") == TARGET_SIGNAL)
        .group_by("user_id").agg(pl.col("item_id").alias("bought"))
    )
    if purchases.height > MAX_USERS:
        purchases = purchases.sample(MAX_USERS, seed=SEED)

    c_hits = c_n = t_hits = t_n = 0
    for row in purchases.iter_rows(named=True):
        uid, bought = row["user_id"], set(row["bought"])
        variant = assign_variant(uid, EXPERIMENT, ["control", "treatment"])
        svc = control_svc if variant == "control" else treatment_svc
        recs = svc.recommend(RecommendationRequest(user_id=uid, n=K)).items
        hit = 1 if bought & set(recs) else 0
        if variant == "control":
            c_hits += hit; c_n += 1
        else:
            t_hits += hit; t_n += 1

    print(f"  assigned {c_n:,} -> control, {t_n:,} -> treatment "
          f"(split {c_n / (c_n + t_n):.1%} / {t_n / (c_n + t_n):.1%})")

    # ---- Analyze ---------------------------------------------------------
    print("\nStep 5/5: Significance analysis (hit@20 as conversion)...")
    result = two_proportion_ztest(c_hits, c_n, t_hits, t_n)
    print("=" * 60)
    print(result.line())
    print("=" * 60)

    if result.significant:
        verdict = ("SHIP IT" if result.absolute_lift > 0
                   else "DO NOT SHIP -- treatment is significantly WORSE")
    else:
        verdict = ("INCONCLUSIVE -- the delta is inside the noise band. Do NOT "
                   "ship on this; either run longer or accept no difference.")
    print(f"  Decision: {verdict}")

    print("\nNext steps:")
    print("  - In production: assign at request time, log variant + outcome to")
    print("    Kafka, and analyze with the SAME z-test on live conversions.")
    print("  - Add guardrail metrics (latency, diversity) that can veto a ship.")
    print("  - Phase 6: near-real-time freshness so v2 trains on fresh data.\n")
    return result


if __name__ == "__main__":
    main()
