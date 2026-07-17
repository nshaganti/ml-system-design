"""
Phase 3 -- Entry Point
=======================
Assembles Phases 1-2 into a live recommendation SERVICE and exercises it like a
production system would.

What it demonstrates:
  1. A single request through the full 100ms path, with per-stage latency
  2. p50/p99 latency over many requests (you only trust latency you measure)
  3. Graceful degradation: a broken ranker still returns results (Rule 10)
  4. Business rules: out-of-stock items are filtered (derived from the
     'available' item property)
  5. Rule 29 inference feature logging: the exact features served, keyed by
     request_id -- the seed of skew-free next-gen training data

Usage:
    cd phase3
    python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase2"))
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events, load_item_properties
from evaluate import temporal_split

from feature_store import PointInTimeFeatureStore
from lr_ranker import LRRanker

from candidate_generator import PopularityCandidateGenerator
from service import RecommendationService, RecommendationRequest

SEED = 42
MAX_POSITIVES = 100_000


def build_training_set(train_events, store, rng):
    """Positives = strong events; negatives = a random item. (Phase 2 pattern.)"""
    strong = train_events.filter(pl.col("event_type").is_in(["add_to_cart", "purchase"]))
    if len(strong) > MAX_POSITIVES:
        strong = strong.sample(MAX_POSITIVES, seed=SEED)
    all_items = train_events["item_id"].unique().to_list()

    pos = strong.select(["user_id", "item_id", "timestamp_ms"]).with_columns(pl.lit(1).alias("label"))
    neg_items = rng.choice(np.array(all_items), size=len(pos))
    neg = pos.select(["user_id", "timestamp_ms"]).with_columns(
        pl.Series("item_id", neg_items).cast(pl.Utf8), pl.lit(0).alias("label"),
    ).select(["user_id", "item_id", "timestamp_ms", "label"])

    return store.get_historical_features(pl.concat([pos, neg]))


def derive_out_of_stock(item_props: pl.DataFrame, cutoff_ms: int) -> set[str]:
    """Most-recent 'available' value before cutoff == '0' -> out of stock."""
    avail = item_props.filter(
        (pl.col("property") == "available") & (pl.col("timestamp_ms") < cutoff_ms)
    )
    latest = (
        avail.sort("timestamp_ms", descending=True)
        .unique(subset=["item_id"], keep="first")
    )
    return set(latest.filter(pl.col("value") == "0")["item_id"].to_list())


def percentile(values, p):
    return round(float(np.percentile(values, p)), 3) if values else 0.0


def main():
    print("\n=== Phase 3: The Recommendation Service ===\n")
    rng = np.random.default_rng(SEED)

    print("Step 1/5: Loading data...")
    events = load_events()
    item_props = load_item_properties()

    print("\nStep 2/5: Temporal split + feature store + ranker...")
    train_events, test_events, cutoff_ms = temporal_split(events, train_fraction=0.8)
    store = PointInTimeFeatureStore().fit(train_events, cutoff_timestamp_ms=cutoff_ms, item_properties=item_props)
    ranker = LRRanker().fit(build_training_set(train_events, store, rng))

    print("\nStep 3/5: Assembling the service...")
    oos = derive_out_of_stock(item_props, cutoff_ms)
    cg = PopularityCandidateGenerator(store._item_totals, pool_size=500)
    service = RecommendationService(
        candidate_generator=cg, feature_store=store, ranker=ranker,
        out_of_stock=oos, model_version="lr_ranker_v1",
    )
    print(f"  candidate pool: {cg.pool_size} | out-of-stock items filtered: {len(oos):,}")

    # A representative sample of real users to fire requests for.
    sample_users = (
        test_events.filter(pl.col("event_type") == "purchase")["user_id"]
        .unique().to_list()
    )
    rng.shuffle(sample_users)
    sample_users = sample_users[:2000]

    print("\nStep 4/5: Single request (per-stage latency breakdown)...")
    resp = service.recommend(RecommendationRequest(user_id=sample_users[0], n=20))
    print(f"  request_id : {resp.request_id}")
    print(f"  model      : {resp.model_version} | fallback: {resp.fallback_used}")
    print(f"  returned   : {len(resp.items)} items")
    print(f"  latency (ms):")
    for stage, ms in resp.latency_ms.items():
        print(f"    {stage:<22} {ms:>7.3f}")
    print(f"    {'TOTAL':<22} {resp.total_latency_ms:>7.3f}  "
          f"(budget {service.latency_budget_ms:.0f}ms -> "
          f"{'OK' if service.within_budget(resp) else 'OVER'})")

    print("\nStep 5/5: Load test + fault injection...")
    totals = []
    for uid in sample_users:
        r = service.recommend(RecommendationRequest(user_id=uid, n=20))
        totals.append(r.total_latency_ms)
    print(f"  {len(totals):,} requests served")
    print(f"  p50 latency : {percentile(totals, 50):.3f} ms")
    print(f"  p99 latency : {percentile(totals, 99):.3f} ms")
    print(f"  within {service.latency_budget_ms:.0f}ms budget: "
          f"{100*np.mean([t <= service.latency_budget_ms for t in totals]):.1f}% of requests")

    # Graceful degradation: a ranker that always throws.
    class BrokenRanker:
        def rank(self, *a, **k):
            raise RuntimeError("simulated model timeout")

    broken = RecommendationService(cg, store, ranker=BrokenRanker(), out_of_stock=oos)
    br = broken.recommend(RecommendationRequest(user_id=sample_users[0], n=20))
    print(f"\n  Fault injection: ranker raised -> fallback_used={br.fallback_used}, "
          f"still returned {len(br.items)} items (no 500). Rule 10 in action.")

    # Rule 29: inspect the inference feature log.
    print(f"\n  Inference feature log: {len(service.feature_log):,} rows "
          f"(exact features served, keyed by request_id).")
    if service.feature_log:
        ex = service.feature_log[0]
        print(f"    sample: item={ex['item_id']} features={ex['features']}")
    print("    ^ Join tomorrow's clicks to THIS to train v2 with zero skew (Rule 29).")

    print("\nNext steps:")
    print("  - Swap PopularityCandidateGenerator for the Phase 1 two-tower ANN.")
    print("  - Serve behind Ray Serve / BentoML; move the store online to Redis.")
    print("  - Phase 4: monitor these logs for drift; Phase 5: A/B test v1 vs v2.\n")
    return service


if __name__ == "__main__":
    main()
