"""
Phase 1 — Two-Tower Model
===========================
Item embeddings + mean-pool user representation.

Architecture:
  - ONE embedding table: items only (no user ID table).
  - User vector = mean of their history item embeddings.
  - Score(user, item) = dot product.
  - Loss: BPR — maximize margin between positive and negative scores.

Why no user embedding table?
  A user ID table can't generalize to users absent from training (cold-start).
  Mean-pooling item embeddings gives us a representation for ANY user with history,
  including users the model has never seen. This is the standard production approach
  (YouTube DNN, BERT4Rec, etc.).

Why dot product (not cosine)?
  Dot product encodes both direction (preference direction) and magnitude
  (how strong the preference is). L2-normalized embeddings + dot = cosine.
  We'll leave normalization optional — unnormalized dot product often works
  better for BPR because item embedding magnitude encodes popularity signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    """
    Single shared item embedding table.
    User is represented as the mean of their history item embeddings.

    embedding_dim:
        64 is a good start. Diminishing returns after 128 on this dataset size.
        Rule 21: feature count should scale with data volume.

    pad_idx:
        Index of the padding token. Embedding for pad_idx is always zeroed
        (padding_idx in nn.Embedding handles this automatically).
    """

    def __init__(self, n_items: int, embedding_dim: int = 64, pad_idx: int = None):
        super().__init__()

        # +1 for the padding token (pad_idx = n_items)
        _pad_idx = n_items if pad_idx is None else pad_idx
        self.pad_idx     = _pad_idx
        self.embedding_dim = embedding_dim

        self.item_emb = nn.Embedding(
            num_embeddings = n_items + 1,  # +1 for pad token
            embedding_dim  = embedding_dim,
            padding_idx    = _pad_idx,     # gradients don't flow through pad positions
        )

        # Initialize with small random values (avoid saturation at start)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)

    def get_user_vector(self, history: torch.Tensor) -> torch.Tensor:
        """
        Compute user representation as mean of history item embeddings.

        history: (batch, seq_len) — padded item indices, pad positions = pad_idx
        returns: (batch, embedding_dim)

        This is what runs at SERVING TIME for every recommendation request.
        Given a user's recent interaction history, compute a query vector and
        search the item index.
        """
        # (batch, seq_len, dim)
        emb = self.item_emb(history)

        # Build mask: 1 where valid (not padding), 0 where padded
        # (batch, seq_len, 1) for broadcasting
        mask = (history != self.pad_idx).unsqueeze(-1).float()

        # Sum valid positions, divide by count of valid positions
        n_valid = mask.sum(dim=1).clamp(min=1.0)  # (batch, 1) — clamp avoids div-by-zero
        user_vec = (emb * mask).sum(dim=1) / n_valid  # (batch, dim)
        return user_vec

    def get_item_vector(self, item_idx: torch.Tensor) -> torch.Tensor:
        """Item embedding lookup. (batch,) -> (batch, dim)"""
        return self.item_emb(item_idx)

    def forward(
        self,
        history: torch.Tensor,   # (batch, seq_len)
        pos_idx: torch.Tensor,   # (batch,)
        neg_idx: torch.Tensor,   # (batch,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (pos_scores, neg_scores) for BPR loss.
        Both are (batch,) tensors of dot products.
        """
        user_vec = self.get_user_vector(history)           # (batch, dim)
        pos_emb  = self.get_item_vector(pos_idx)           # (batch, dim)
        neg_emb  = self.get_item_vector(neg_idx)           # (batch, dim)

        # Dot product: element-wise multiply then sum along embedding dim
        pos_scores = (user_vec * pos_emb).sum(dim=1)       # (batch,)
        neg_scores = (user_vec * neg_emb).sum(dim=1)       # (batch,)
        return pos_scores, neg_scores


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    BPR loss: -mean(log(sigmoid(pos_score - neg_score)))

    Intuition: we don't need the model to predict absolute scores — just to
    rank positives above negatives. BPR directly optimizes this ordering.
    For a perfectly ranked pair: pos >> neg → sigmoid ≈ 1 → log ≈ 0 (no loss).
    For a wrongly ranked pair: neg > pos → sigmoid < 0.5 → loss is high.
    """
    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))


def get_all_item_embeddings(model: TwoTowerModel, device: str = "cpu") -> torch.Tensor:
    """
    Extract the full item embedding matrix (n_items, dim), excluding the pad token.
    Used to build the ANN index after training.
    """
    model.eval()
    with torch.no_grad():
        # All item indices except the pad token (last row)
        all_idx = torch.arange(model.item_emb.num_embeddings - 1, device=device)
        return model.item_emb(all_idx).cpu()
