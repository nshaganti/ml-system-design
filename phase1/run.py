"""
Phase 1 — Entry Point
======================
Full Phase 1 pipeline:
  1. Load data (reuse phase0/load_data.py)
  2. Temporal split with the SAME cutoff as Phase 0 (Rule 33)
  3. Build item vocab + BPR training dataset
  4. Train two-tower model (MLflow tracking)
  5. Build embedding index
  6. Evaluate with phase0/evaluate.py recall_at_k — SAME function, same seed

After this runs, open the MLflow UI to inspect the training run:
    mlflow ui --port 5000   (from the project root)
    then visit http://localhost:5000

Usage:
    cd phase1
    python run.py
"""

import sys
from pathlib import Path

import json

# Make phase0 importable
sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events, load_item_properties
from evaluate  import temporal_split, recall_at_k
from heuristic_ranker import HeuristicRanker

from dataset import build_vocab, build_user_histories, BPRDataset
from two_tower import get_all_item_embeddings
from train import train
from index import EmbeddingIndex
from ranker import TwoTowerRanker
from results_io import save_results

PHASE_DIR = Path(__file__).parent

# Phase 0 writes its metrics here; we load them for a fair comparison instead
# of hardcoding numbers that silently rot when Phase 0 is rerun.
PHASE0_RESULTS_PATH = Path(__file__).parent.parent / "phase0" / "results.json"

# Fallback baseline (from a prior Phase 0 run) if results.json is missing.
_PHASE0_FALLBACK = {
    "recall_at_k":       0.0308,
    "catalog_coverage":  0.0626,
    "warm_user_recall":  0.0466,
    "cold_start_recall": 0.0244,
}


def _load_phase0_baseline() -> dict:
    """Load Phase 0 metrics from disk; warn and use fallback if not found."""
    if PHASE0_RESULTS_PATH.exists():
        return json.loads(PHASE0_RESULTS_PATH.read_text())
    print(
        f"[run] WARNING: {PHASE0_RESULTS_PATH} not found — run phase0/run.py first "
        "for an up-to-date baseline. Using stale fallback numbers."
    )
    return _PHASE0_FALLBACK


def main():
    print("\n=== Phase 1: Two-Tower Candidate Generation ===\n")

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print("Step 1/6: Loading data...")
    events     = load_events()
    item_props = load_item_properties()

    # ── Step 2: Temporal split (same 80/20 split as Phase 0) ─────────────────
    print("\nStep 2/6: Temporal split...")
    train_events, test_events, cutoff_ms = temporal_split(events, train_fraction=0.8)

    # ── Step 3: Build item vocab + training dataset ───────────────────────────
    print("\nStep 3/6: Building vocab and training dataset...")
    vocab           = build_vocab(train_events)
    user_histories  = build_user_histories(train_events, vocab)
    dataset         = BPRDataset(train_events, vocab, user_histories)

    # ── Step 4: Train two-tower model ─────────────────────────────────────────
    print("\nStep 4/6: Training two-tower model...")
    model = train(dataset, vocab, run_name="two-tower-v1")

    # ── Step 5: Build ANN index from trained item embeddings ──────────────────
    print("\nStep 5/6: Building embedding index...")
    item_embeddings = get_all_item_embeddings(model)
    index = EmbeddingIndex(
        item_embeddings=item_embeddings,
        idx_to_item_id=vocab.idx_to_item_id,
    )

    # ── Step 6: Evaluate — reuse phase0/evaluate.py (apples-to-apples) ────────
    print("\nStep 6/6: Evaluating...")

    # Phase 0 fallback for cold-start users
    fallback = HeuristicRanker()
    fallback.fit(
        events=train_events,
        item_properties=item_props,
        cutoff_timestamp_ms=cutoff_ms,
    )

    phase1_ranker = TwoTowerRanker(
        model=model,
        vocab=vocab,
        index=index,
        fallback=fallback,
        item_embeddings=item_embeddings,   # reuse the matrix already built for the index
    )

    # Shared coverage denominator — MUST match Phase 0's for a fair comparison.
    catalog_size = train_events["item_id"].n_unique()
    results = recall_at_k(
        ranker=phase1_ranker,
        train_events=train_events,
        test_events=test_events,
        k=20,
        max_users=5000,   # same cap as Phase 0 run
        catalog_size=catalog_size,
    )

    # ── Summary: Phase 0 vs Phase 1 ─────────────────────────────────
    phase0 = _load_phase0_baseline()
    PHASE0_RECALL   = phase0["recall_at_k"]
    PHASE0_COVERAGE = phase0["catalog_coverage"]
    PHASE0_WARM     = phase0["warm_user_recall"]
    PHASE0_COLD     = phase0["cold_start_recall"]

    print("\n" + "=" * 55)
    print("  PHASE 0 vs PHASE 1 COMPARISON")
    print("=" * 55)
    print(f"  {'Metric':<25} {'Phase 0':>10} {'Phase 1':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Recall@20':<25} {PHASE0_RECALL:>10.4f} {results['recall_at_k']:>10.4f}  "
          + ("[+] improvement" if results['recall_at_k'] > PHASE0_RECALL else "[-] regression — debug features"))
    print(f"  {'Catalog coverage':<25} {PHASE0_COVERAGE:>10.4f} {results['catalog_coverage']:>10.4f}  "
          + ("[+]" if results['catalog_coverage'] > PHASE0_COVERAGE else "[-]"))
    print(f"  {'Warm user recall':<25} {PHASE0_WARM:>10.4f} {results['warm_user_recall']:>10.4f}")
    print(f"  {'Cold-start recall':<25} {PHASE0_COLD:>10.4f} {results['cold_start_recall']:>10.4f}")
    print("=" * 55)

    # Persist for the scoreboard generator + HTML report.
    save_results(PHASE_DIR, results)
    print(f"[run] Saved Phase 1 metrics to {PHASE_DIR / 'results.json'}")

    print("\nNext steps:")
    print("  - View the training run: mlflow ui --port 5000")
    print("  - If warm user recall > Phase 0: the model is learning from history.")
    print("  - If coverage improved: the model is surfacing long-tail items.")
    print("  - Phase 2 adds a logistic regression RANKER on top of these 500 candidates")
    print("    + a feature store to eliminate training-serving skew.")
    print()

    return results


if __name__ == "__main__":
    main()
