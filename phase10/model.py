"""
Phase 10 -- Sequence model core: GRU4Rec (Part I extension)
============================================================
Phase 7 showed co-visitation beats popularity on session next-item prediction by
reading pairwise co-occurrence. A sequence model asks for more: it reads the
*order* of a session and learns a representation of "where this session is going."
GRU4Rec (Hidasi et al., 2016) is the canonical, still-competitive baseline: embed
each item, run a GRU over the session, and predict the next item from the final
hidden state.

This module is the pure model (Rule 5): it takes padded item-id sequences and
returns next-item logits. `dataset.py` turns sessions into training tensors and
`run.py` trains it and grades it on the SAME leave-one-out protocol as Phase 7, so
the co-visitation vs. sequence-model comparison is apples-to-apples.

Index 0 is reserved as the PAD token so we can left-pad short sessions; embeddings
for index 0 are zeroed and frozen so padding contributes nothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PAD_IDX = 0


class GRU4Rec(nn.Module):
    """
    Minimal GRU4Rec. `vocab_size` counts real items; index 0 is PAD, so the
    embedding and output layers span vocab_size + 1 rows.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        n = vocab_size + 1  # +1 for PAD at index 0
        self.embedding = nn.Embedding(n, embedding_dim, padding_idx=PAD_IDX)
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, n)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        sequences: (batch, seq_len) LongTensor of item indices (0 = PAD).
        Returns:   (batch, vocab_size + 1) logits over the next item.
        """
        emb = self.embedding(sequences)          # (B, T, E)
        _, h = self.gru(emb)                      # h: (num_layers, B, H)
        last = self.dropout(h[-1])               # (B, H) -- final layer's state
        return self.output(last)                 # (B, n)

    @torch.no_grad()
    def recommend(self, sequence_idx: list[int], k: int, exclude: set[int] | None = None) -> list[int]:
        """
        Score the next item for a single session and return the top-k item indices
        (never PAD, never anything in `exclude`). Pure inference helper for eval.
        """
        self.eval()
        exclude = set(exclude or set()) | {PAD_IDX}
        x = torch.tensor([sequence_idx], dtype=torch.long)
        logits = self.forward(x).squeeze(0)      # (n,)
        logits[list(exclude)] = float("-inf")
        # Cap k so topk can't fall back onto excluded (-inf) entries when few
        # items remain allowable.
        k_eff = min(k, logits.numel() - len(exclude))
        if k_eff <= 0:
            return []
        top = torch.topk(logits, k_eff).indices.tolist()
        return top
