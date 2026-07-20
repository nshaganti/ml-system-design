"""
Phase 1 — Training Loop
=========================
Trains the two-tower model and tracks experiments with MLflow.

MLflow gives us:
  - A permanent record of every run (hyperparams, metrics, model artifact)
  - The ability to answer "which model is in production?" two months from now
  - Comparison across runs (e.g., embedding_dim=64 vs 128)

Google's Rule 16: Plan to launch and iterate. This infra is what makes
safe iteration possible — you never lose track of what you changed.

Run the MLflow UI with: mlflow ui --port 5000
Then open http://localhost:5000 to see all runs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from two_tower import TwoTowerModel, bpr_loss, get_all_item_embeddings
from dataset import BPRDataset, ItemVocab


EMBEDDING_DIM = 64
BATCH_SIZE    = 512
EPOCHS        = 20
LR            = 5e-3
WEIGHT_DECAY  = 1e-5   # L2 regularization — prevents embedding collapse


def train(
    dataset: BPRDataset,
    vocab: ItemVocab,
    experiment_name: str = "phase1-two-tower",
    run_name: str | None = None,
) -> TwoTowerModel:
    """
    Train the two-tower model, log to MLflow, return trained model.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[train] Device: {device}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # 0 = main process (safe on macOS with MPS)
        pin_memory=False,
    )

    model = TwoTowerModel(n_items=vocab.size, embedding_dim=EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    # MLflow experiment — creates it if it doesn't exist
    mlflow.set_experiment(experiment_name)

    hparams = {
        "embedding_dim": EMBEDDING_DIM,
        "batch_size":    BATCH_SIZE,
        "epochs":        EPOCHS,
        "lr":            LR,
        "weight_decay":  WEIGHT_DECAY,
        "n_items":       vocab.size,
        "n_training_triples": len(dataset),
        "device":        device,
        "positive_events": "STRONG signals (target actions)",
        "user_repr":     "mean-pool item embeddings",
        "negative_sampling": getattr(dataset, "negative_sampling", "uniform"),
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(hparams)
        print(f"\n[train] MLflow run: {run.info.run_id}")
        print(f"  Experiment : {experiment_name}")
        print(f"  Triples    : {len(dataset):,}")
        print(f"  Batches/ep : {len(loader):,}")
        print(f"  Vocab size : {vocab.size:,} items\n")

        best_loss = float("inf")
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss  = 0.0
            epoch_start = time.time()

            for batch in loader:
                history = batch["history"].to(device)
                pos     = batch["pos"].to(device)
                neg     = batch["neg"].to(device)

                optimizer.zero_grad()
                pos_scores, neg_scores = model(history, pos, neg)
                loss = bpr_loss(pos_scores, neg_scores)
                loss.backward()
                # Gradient clipping — stabilizes early training
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss   = epoch_loss / len(loader)
            elapsed    = time.time() - epoch_start

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"  Epoch {epoch:2d}/{EPOCHS} | loss: {avg_loss:.4f} | {elapsed:.1f}s")

            if avg_loss < best_loss:
                best_loss  = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Restore best checkpoint
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        mlflow.log_metric("best_train_loss", best_loss)

        # Log model artifact so we can reload it later
        mlflow.pytorch.log_model(model.cpu(), artifact_path="model")
        print(f"\n[train] Best loss: {best_loss:.4f} | Model logged to MLflow run {run.info.run_id}")

    return model.cpu()
