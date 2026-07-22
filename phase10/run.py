"""
Phase 10 -- Entry Point: GRU4Rec vs co-visitation vs popularity
================================================================
The head-to-head Phase 7 asked for. We train a GRU4Rec sequence model on the
training sessions and grade all three methods -- popularity, co-visitation
(Phase 7), and GRU4Rec -- on the IDENTICAL leave-one-out next-item protocol and
the IDENTICAL test cases. Same split, same sessions, same metrics: the only thing
that varies is the model, so the comparison is honest.

The question: does modelling session *order* with a GRU beat item-item
co-occurrence on KuaiRand? (Sequence models usually win, but the margin depends on
how much order actually carries -- another "measure it, don't assume it" moment.)

Usage:
    cd phase10 && python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase7"))
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_events
from evaluate import temporal_split
from metrics import recall_at_k, ndcg_at_k, mean
from results_io import save_results
from repro import set_global_seed

from covisitation import CoVisitationRecommender, sessionize, session_item_lists
from session_eval import build_test_cases, reciprocal_rank   # reuse Phase 7's protocol (DRY)

from model import GRU4Rec, PAD_IDX
import seq_data as ds

PHASE_DIR = Path(__file__).parent

SEED = 42
MAX_LEN = 20
EMB_DIM = 64
HID_DIM = 64
EPOCHS = 12
BATCH = 512
LR = 3e-3
MAX_TRAIN_PAIRS = 500_000
MAX_TEST_SESSIONS = 30_000
K = 20


def train_model(pairs, vocab_size) -> GRU4Rec:
    torch.manual_seed(SEED)
    X = torch.tensor([p for p, _ in pairs], dtype=torch.long)
    y = torch.tensor([t for _, t in pairs], dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True)

    model = GRU4Rec(vocab_size, EMB_DIM, HID_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
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
    print("\n=== Phase 10: GRU4Rec vs Co-Visitation vs Popularity ===\n")

    print("Step 1/5: Loading + temporal split...")
    events = load_events()
    train_events, test_events, _ = temporal_split(events, train_fraction=0.8)

    print("\nStep 2/5: Fitting co-visitation (also gives us the popularity baseline)...")
    covis = CoVisitationRecommender().fit(train_events)

    print("\nStep 3/5: Building vocab + GRU4Rec training pairs...")
    train_item_lists = session_item_lists(sessionize(train_events))
    vocab = ds.build_vocab(train_item_lists)
    idx2item = {v: k for k, v in vocab.items()}
    pairs = ds.make_training_pairs(train_item_lists, vocab, MAX_LEN, MAX_TRAIN_PAIRS)
    print(f"  vocab={len(vocab):,} items | training pairs={len(pairs):,} (max_len={MAX_LEN})")

    print("\nStep 4/5: Training GRU4Rec...")
    model = train_model(pairs, len(vocab))

    cases = build_test_cases(test_events)
    if len(cases) > MAX_TEST_SESSIONS:
        cases = cases[:MAX_TEST_SESSIONS]
    print(f"\nStep 5/5: Evaluating on {len(cases):,} leave-one-out cases...")

    def rec_pop(context):
        return covis.popular(n=K, exclude=set(context))

    def rec_covis(context):
        return covis.recommend(context, n=K, exclude=set(context))

    def rec_gru(context):
        idx = ds.encode_context(context, vocab, MAX_LEN)
        if all(i == PAD_IDX for i in idx):
            return covis.popular(n=K, exclude=set(context))   # cold: back off
        exclude_idx = {i for i in idx if i != PAD_IDX}
        out = model.recommend(idx, K, exclude=exclude_idx)
        return [idx2item[i] for i in out if i in idx2item]

    pop = evaluate(rec_pop, cases)
    cov = evaluate(rec_covis, cases)
    gru = evaluate(rec_gru, cases)

    print("\n" + "=" * 66)
    print("  SESSION-BASED NEXT-ITEM PREDICTION  (higher is better)")
    print("=" * 66)
    print(f"  {'Metric':<11}{'Popularity':>12}{'Co-vis':>10}{'GRU4Rec':>10}{'GRU vs covis':>14}")
    print(f"  {'-' * 62}")
    for key, label in [("recall", "Recall@20"), ("mrr", "MRR@20"), ("ndcg", "NDCG@20")]:
        lift = (gru[key] / cov[key] - 1) * 100 if cov[key] > 0 else 0.0
        print(f"  {label:<11}{pop[key]:>12.4f}{cov[key]:>10.4f}{gru[key]:>10.4f}{lift:>+13.0f}%")
    print("=" * 66)
    print("  Reading the numbers: GRU4Rec models session ORDER, not just co-occurrence.")
    print("  Whether that extra structure pays off -- and by how much -- is exactly the")
    print("  kind of thing you measure on a fixed protocol rather than assume.")

    results = {"popularity": pop, "covisitation": cov, "gru4rec": gru,
               "vocab_size": len(vocab), "train_pairs": len(pairs)}
    save_results(PHASE_DIR, results)

    print("\nNext steps:")
    print("  - Try SASRec (self-attention) on this same protocol; compare margins.")
    print("  - Blend GRU4Rec candidates into the Phase 3 service candidate union.")
    print("  - Add sampled-softmax / negatives if the catalog grows large.\n")
    return results


if __name__ == "__main__":
    main()
