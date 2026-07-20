"""
Phase 0 — Entry Point
======================
Run this to execute the full Phase 0 pipeline:
  1. Load & validate the KuaiRand-Pure dataset
  2. Build heuristic ranker on training data
  3. Evaluate with temporal split

Usage:
    cd phase0
    python run.py

Prerequisites:
    pip install -i https://pypi.ci.artifacts.walmart.com/artifactory/api/pypi/external-pypi/simple \
        polars duckdb scikit-learn

Dataset:
    Download KuaiRand-Pure from https://kuairand.com and place it at
    ../data/KuaiRand-Pure/ (see data/README.md for the exact layout).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events, load_item_properties, summarize
from heuristic_ranker import HeuristicRanker
from evaluate import temporal_split, recall_at_k
from results_io import save_results

PHASE_DIR = Path(__file__).parent


def main():
    print("\n=== Phase 0: Heuristic Baseline ===\n")

    # Step 1: Load data
    print("Step 1/3: Loading data...")
    events     = load_events()
    item_props = load_item_properties()
    summarize(events, item_props)

    # Step 2: Temporal split — Rule 33 (test on data AFTER training cutoff)
    print("\nStep 2/3: Temporal split...")
    train_events, test_events, cutoff_ms = temporal_split(events, train_fraction=0.8)

    # Step 3: Fit heuristic ranker on training data
    print("\nStep 3/3: Fitting ranker and evaluating...")
    ranker = HeuristicRanker()
    ranker.fit(
        events=train_events,
        item_properties=item_props,
        cutoff_timestamp_ms=cutoff_ms,
    )

    # Step 4: Evaluate
    # Shared coverage denominator = the full recommendable universe.
    # Phase 1 must use this exact same value for an apples-to-apples comparison.
    catalog_size = train_events["item_id"].n_unique()
    results = recall_at_k(
        ranker=ranker,
        train_events=train_events,
        test_events=test_events,
        k=20,
        max_users=5000,
        catalog_size=catalog_size,
    )

    # Persist so Phase 1 can load the baseline, and so the scoreboard generator
    # (scripts/build_results.py) can read real numbers instead of hand-copied ones.
    save_results(PHASE_DIR, results)
    print(f"\n[run] Saved Phase 0 baseline metrics to {PHASE_DIR / 'results.json'}")

    print("\nNext steps:")
    print("  - This Recall@20 is your Phase 1 target to beat.")
    print("  - Note the cold-start recall gap: that's what user history adds.")
    print("  - Note the catalog coverage: a popularity ranker covers very few items.")
    print("    (A Phase 1 model should improve coverage, not just recall.)")
    print("\nSee ml-system-design-recommendation-engine.md -> Phase 1 to continue.\n")

    return results


if __name__ == "__main__":
    main()
