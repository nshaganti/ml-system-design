"""
Phase 6 -- Entry Point
=======================
Demonstrates near-real-time freshness: streaming a user's in-session behavior
into the online feature store changes their next recommendations WITHOUT
retraining the model.

Two parts:
  1. Qualitative -- pick one user, show their features + recs before and after we
     stream one strong event. You can watch the category affinity jump and the
     recs re-rank.
  2. Quantitative -- an in-session experiment. For users with >=2 strong events,
     use the EARLIEST as a "seed" we stream in, then measure hit@20 on their
     LATER items. Compare FROZEN batch features vs FRESH streamed features. Both
     arms exclude the seed item, so the only difference is feature freshness.

Usage:
    cd phase6
    python run.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase3"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase5"))
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events, load_item_properties
from evaluate import temporal_split

from feature_store import PointInTimeFeatureStore
from lr_ranker import LRRanker
from training import build_labelled_features

from candidate_generator import PopularityCandidateGenerator
from service import RecommendationService, RecommendationRequest

from experiment import two_proportion_ztest

from streaming_store import StreamingFeatureStore

SEED = 42
K = 20
STRONG = ["add_to_cart", "purchase"]


def user_strong_sequences(test_events: pl.DataFrame) -> dict[str, list[tuple]]:
    """user_id -> time-ordered list of (item_id, event_type) strong events."""
    strong = (
        test_events.filter(pl.col("event_type").is_in(STRONG))
        .select(["user_id", "item_id", "event_type", "timestamp_ms"])
        .sort("timestamp_ms")
    )
    seqs: dict[str, list[tuple]] = defaultdict(list)
    for r in strong.iter_rows(named=True):
        seqs[r["user_id"]].append((r["item_id"], r["event_type"]))
    return seqs


def main():
    print("\n=== Phase 6: Near-Real-Time Freshness ===\n")
    rng = np.random.default_rng(SEED)

    print("Step 1/4: Loading data + fitting store/ranker...")
    events = load_events()
    item_props = load_item_properties()
    train_events, test_events, cutoff_ms = temporal_split(events, train_fraction=0.8)
    batch = PointInTimeFeatureStore().fit(train_events, cutoff_timestamp_ms=cutoff_ms, item_properties=item_props)
    ranker = LRRanker().fit(build_labelled_features(train_events, batch, rng))
    cg = PopularityCandidateGenerator(batch._item_totals, pool_size=500)

    frozen_svc = RecommendationService(cg, batch, ranker=ranker, model_version="frozen_batch")
    fresh_svc = RecommendationService(cg, batch, ranker=ranker, model_version="fresh_streaming")

    seqs = user_strong_sequences(test_events)
    multi = {u: s for u, s in seqs.items() if len({i for i, _ in s}) >= 2}
    print(f"  users with >=2 distinct strong-event items: {len(multi):,}")

    # ---- Part 1: qualitative before/after --------------------------------
    print("\nStep 2/4: Qualitative -- one user, before vs after a streamed event...")
    demo_uid, demo_seq = next(iter(multi.items()))
    seed_item = demo_seq[0][0]
    seed_cat = batch._lookup_str(batch._item_category, seed_item)

    sfs = StreamingFeatureStore(batch)
    before = batch.get_online_features(demo_uid, seed_item)
    sfs.ingest(demo_uid, seed_item, demo_seq[0][1])
    after = sfs.get_online_features(demo_uid, seed_item)
    print(f"  user {demo_uid} streams a '{demo_seq[0][1]}' on item {seed_item} (category {seed_cat})")
    print(f"    user_pop          : {before['user_pop']} -> {after['user_pop']}")
    print(f"    user_cat_affinity : {before['user_cat_affinity']} -> {after['user_cat_affinity']}  "
          f"(the model now knows the in-session intent)")
    print(f"    session dedup     : item {seed_item} now suppressed -> {seed_item in sfs.session_seen(demo_uid)}")

    # ---- Part 2: quantitative in-session experiment ----------------------
    print("\nStep 3/4: Quantitative -- frozen vs fresh hit@20 on later items...")
    frozen_hits = fresh_hits = n = 0
    for uid, seq in multi.items():
        seed_item, seed_type = seq[0]
        later_items = {i for i, _ in seq[1:] if i != seed_item}
        if not later_items:
            continue
        n += 1

        # Frozen arm: batch features, exclude only the seed item (fair baseline).
        frozen_recs = frozen_svc.recommend(
            RecommendationRequest(user_id=uid, n=K), already_seen={seed_item}
        ).items
        frozen_hits += int(bool(later_items & set(frozen_recs)))

        # Fresh arm: stream the seed event, then recommend.
        sfs = StreamingFeatureStore(batch)
        sfs.ingest(uid, seed_item, seed_type)
        fresh_svc.store = sfs
        fresh_recs = fresh_svc.recommend(
            RecommendationRequest(user_id=uid, n=K), already_seen=sfs.session_seen(uid)
        ).items
        fresh_hits += int(bool(later_items & set(fresh_recs)))

    print(f"  evaluated {n:,} users with a seed + >=1 later item")

    # ---- Part 3: significance --------------------------------------------
    print("\nStep 4/4: Did freshness help? (hit@20 on later items)")
    result = two_proportion_ztest(frozen_hits, n, fresh_hits, n)
    print("=" * 60)
    print(f"  frozen  batch features : hit@20 = {frozen_hits / n:.4f} ({frozen_hits:,}/{n:,})")
    print(f"  fresh streaming features: hit@20 = {fresh_hits / n:.4f} ({fresh_hits:,}/{n:,})")
    print(f"  absolute lift={result.absolute_lift:+.4f}  relative={result.relative_lift:+.1%}")
    print(f"  z={result.z_score:.3f}  p={result.p_value:.4f}  -> "
          f"{'SIGNIFICANT' if result.significant else 'not significant'}")
    print("=" * 60)
    print("  Key point: NO RETRAIN happened. We changed recommendations purely by")
    print("  streaming fresh features into the online store (Rule 8). The model is")
    print("  byte-for-byte identical between the two arms.")

    print("\nNext steps:")
    print("  - In production: Kafka -> Flink -> Redis keeps this delta layer fresh")
    print("    at ~30s lag; the batch store is rebuilt daily underneath it.")
    print("  - Reserve online LEARNING (retrain on the stream) for <1min SLAs only;")
    print("    it adds catastrophic-forgetting and time-skew failure modes.")
    print("  - That completes Phases 0-6: the full offline-to-online lifecycle.\n")
    return result


if __name__ == "__main__":
    main()
