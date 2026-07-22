"""
Phase 18 -- Entry Point: SASRec vs GRU4Rec vs Co-Visitation vs Popularity
=========================================================================
The follow-up Phase 10 asked for. Phase 10 found GRU4Rec *lost* to co-visitation on
KuaiRand. Self-attention (SASRec) can look at any earlier item directly instead of
squeezing history through one recurrent state -- does that close the gap?

We train SASRec on the SAME training sessions and grade it on the IDENTICAL
leave-one-out protocol, split, and test cases as Phases 7 and 10. Popularity and
co-visitation are recomputed here (they match Phase 10); GRU4Rec's number is read
from Phase 10's saved results (same protocol) for reference. Only the model changes.

Usage:
    cd phase18 && python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase7"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase10"))
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events
from evaluate import temporal_split
from metrics import recall_at_k, ndcg_at_k, mean
from results_io import save_results
from repro import set_global_seed

from covisitation import CoVisitationRecommender, sessionize, session_item_lists
from session_eval import build_test_cases, reciprocal_rank   # Phase 7 protocol (DRY)
import seq_data as ds                                          # Phase 10 encoding (DRY)

from sasrec import SASRec, PAD_IDX
from model import GRU4Rec           # Phase 10 model, trained here under the same budget

PHASE_DIR = Path(__file__).parent

SEED = 42
MAX_LEN = 20
EMB_DIM = 64
EPOCHS = 20
BATCH = 512
LR = 2e-3               # transformers like a gentler LR than the GRU's 3e-3
MAX_TRAIN_PAIRS = 300_000
MAX_TEST_SESSIONS = 20_000
K = 20


def train_model(model, pairs, lr):
    """Train any next-item model (GRU4Rec or SASRec) under an identical budget."""
    torch.manual_seed(SEED)
    X = torch.tensor([p for p, _ in pairs], dtype=torch.long)
    y = torch.tensor([t for _, t in pairs], dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        print(f"    epoch {epoch}/{EPOCHS}  avg loss={total / len(X):.4f}")
    return model


def evaluate(recommend_fn, cases) -> dict:
    recalls, mrrs, ndcgs = [], [], []
    for context, target in cases:
        recs = recommend_fn(context)
        relevant = {target}
        recalls.append(recall_at_k(recs, relevant, K))
        ndcgs.append(ndcg_at_k(recs, relevant, K))
        mrrs.append(reciprocal_rank(recs, target, K))
    return {"recall": mean(recalls), "mrr": mean(mrrs), "ndcg": mean(ndcgs)}


def main():
    set_global_seed()   # reproducible torch weight init + shuffling
    print("\n=== Phase 18: SASRec vs GRU4Rec vs Co-Visitation vs Popularity ===\n")

    print("Step 1/5: Loading + temporal split...")
    events = load_events()
    train_events, test_events, _ = temporal_split(events, train_fraction=0.8)

    print("\nStep 2/5: Fitting co-visitation (+ popularity baseline)...")
    covis = CoVisitationRecommender().fit(train_events)

    print("\nStep 3/5: Building vocab + training pairs (shared by both models)...")
    train_item_lists = session_item_lists(sessionize(train_events))
    vocab = ds.build_vocab(train_item_lists)
    idx2item = {v: k for k, v in vocab.items()}
    pairs = ds.make_training_pairs(train_item_lists, vocab, MAX_LEN, MAX_TRAIN_PAIRS)
    print(f"  vocab={len(vocab):,} items | training pairs={len(pairs):,} (max_len={MAX_LEN})")

    print("\nStep 4/5: Training GRU4Rec then SASRec (identical budget = fair)...")
    print("  [GRU4Rec]")
    gru_model = train_model(GRU4Rec(len(vocab), EMB_DIM, EMB_DIM), pairs, lr=3e-3)
    print("  [SASRec]")
    sas_model = train_model(SASRec(len(vocab), MAX_LEN, EMB_DIM), pairs, lr=LR)

    cases = build_test_cases(test_events)
    if len(cases) > MAX_TEST_SESSIONS:
        cases = cases[:MAX_TEST_SESSIONS]
    print(f"\nStep 5/5: Evaluating on {len(cases):,} leave-one-out cases...")

    def rec_pop(context):
        return covis.popular(n=K, exclude=set(context))

    def rec_covis(context):
        return covis.recommend(context, n=K, exclude=set(context))

    def _rec_seq(model, context):
        idx = ds.encode_context(context, vocab, MAX_LEN)
        if all(i == PAD_IDX for i in idx):
            return covis.popular(n=K, exclude=set(context))   # cold: back off
        exclude_idx = {i for i in idx if i != PAD_IDX}
        out = model.recommend(idx, K, exclude=exclude_idx)
        return [idx2item[i] for i in out if i in idx2item]

    pop = evaluate(rec_pop, cases)
    cov = evaluate(rec_covis, cases)
    gru = evaluate(lambda c: _rec_seq(gru_model, c), cases)
    sas = evaluate(lambda c: _rec_seq(sas_model, c), cases)

    print("\n" + "=" * 74)
    print("  SESSION-BASED NEXT-ITEM PREDICTION  (higher is better)")
    print("=" * 74)
    print(f"  {'Metric':<11}{'Popularity':>12}{'Co-vis':>10}{'GRU4Rec':>10}{'SASRec':>10}{'SAS vs covis':>14}")
    print(f"  {'-' * 70}")
    for key, label in [("recall", "Recall@20"), ("mrr", "MRR@20"), ("ndcg", "NDCG@20")]:
        lift = (sas[key] / cov[key] - 1) * 100 if cov[key] > 0 else 0.0
        print(f"  {label:<11}{pop[key]:>12.4f}{cov[key]:>10.4f}{gru[key]:>10.4f}{sas[key]:>10.4f}{lift:>+13.0f}%")
    print("=" * 74)

    sas_vs_cov = (sas["recall"] / cov["recall"] - 1) * 100 if cov["recall"] else 0.0
    sas_vs_gru = (sas["recall"] / gru["recall"] - 1) * 100 if gru["recall"] else 0.0
    print(f"\n  SASRec vs co-visitation: {sas_vs_cov:+.0f}% recall. "
          f"SASRec vs GRU4Rec: {sas_vs_gru:+.0f}% recall.")
    print("  Reading the numbers: self-attention can read any earlier item directly.")
    print("  Whether that beats co-occurrence on THIS catalog is measured, not assumed --")
    print("  the same honesty test Phases 7 and 10 applied. Both neural models trained")
    print("  on the identical pairs/epochs, so SASRec vs GRU4Rec is a clean comparison.")

    results = {"popularity": pop, "covisitation": cov, "gru4rec": gru, "sasrec": sas,
               "vocab_size": len(vocab), "train_pairs": len(pairs)}
    save_results(PHASE_DIR, results)

    print("\nNext steps:")
    print("  - Blend the winner's candidates into the Phase 3 service candidate union.")
    print("  - Sampled-softmax / in-batch negatives if the catalog grows large.")
    print("  - Try longer max_len / more blocks only if the win justifies the cost.\n")
    return results


if __name__ == "__main__":
    main()
