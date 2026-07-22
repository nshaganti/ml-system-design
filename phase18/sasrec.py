"""
Phase 18 -- Sequence model core: SASRec (Self-Attentive Sequential Recommendation)
==================================================================================
Phase 10 asked whether modelling session ORDER with a GRU beats item-item
co-occurrence (Phase 7). On KuaiRand the honest answer was no -- GRU4Rec *lost* to
co-visitation by ~15%. This phase asks the natural follow-up: does **self-attention**
(SASRec, Kang & McAuley 2018) -- which can look at any earlier item directly instead
of squeezing history through a single recurrent state -- close that gap?

SASRec embeds each item, adds a learned positional embedding, runs a stack of
**causal** self-attention blocks (each position may only attend to itself and
earlier ones), and predicts the next item from the most-recent position's
representation. It is graded on the SAME leave-one-out protocol, SAME split, SAME
test cases as Phases 7 and 10 -- the only thing that changes is the model.

Pure model (Rule 5): padded item-id sequences in, next-item logits out. Reuses
Phase 10's `seq_data.py` for vocab/encoding so "a session" means exactly the same
thing across all three phases -- that's what makes the head-to-head fair.

Index 0 is the PAD token (matches Phase 10's convention); its embedding is frozen to
zero and it is masked out of attention so padding contributes nothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PAD_IDX = 0


class SASRec(nn.Module):
    """
    Minimal SASRec. `vocab_size` counts real items; index 0 is PAD, so the embedding
    and output layers span vocab_size + 1 rows. Left-padded inputs put the most
    recent item at the last position, which is where we read the prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embedding_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        n = vocab_size + 1  # +1 for PAD at index 0
        self.item_emb = nn.Embedding(n, embedding_dim, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(max_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=embedding_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.output = nn.Linear(embedding_dim, n)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        sequences: (batch, seq_len) LongTensor of item indices (0 = PAD).
        Returns:   (batch, vocab_size + 1) logits over the next item, read from the
                   most-recent (last) position.

        Inputs are LEFT-padded (pads first, most-recent item last). Absolute position
        ids would then depend on how much padding precedes the real items -- an
        inconsistent signal that cripples a transformer. So we index positions from
        the END: the most-recent item is position 0, the one before it position 1,
        etc. Recency, not absolute offset, is what the positional embedding encodes.
        """
        b, t = sequences.shape
        rev_positions = (t - 1) - torch.arange(t, device=sequences.device)   # last -> 0
        positions = rev_positions.unsqueeze(0).expand(b, t)
        x = self.item_emb(sequences) + self.pos_emb(positions)
        x = self.dropout(x)

        # Causal mask: position i may attend only to positions <= i.
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=sequences.device), diagonal=1)
        pad_mask = sequences == PAD_IDX                      # (B, T) True where PAD

        # Correctness > speed (Rule 32: train == serve). PyTorch's TransformerEncoder
        # "fast path" (taken under eval() + no_grad) returns all-NaN for fully-masked
        # query rows -- which left-padding ALWAYS creates: a leading PAD position, under
        # the causal + key-padding masks, is allowed to attend to nothing, and the fused
        # kernel propagates that NaN across the whole row. Because recommend() runs under
        # eval()/no_grad, topk() then saw NaN logits and returned items in index order --
        # NOT what the model scored. (train() forward is finite and learns fine, so the
        # old shape/finite contract tests never caught it.) Disabling the fused kernel
        # makes train and serve take the identical, correct math path. CPU-cheap here.
        if hasattr(torch.backends, "mha"):
            torch.backends.mha.set_fastpath_enabled(False)

        out = self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)   # (B, T, E)
        last = out[:, -1, :]                                 # most recent position
        return self.output(last)                             # (B, n)

    @torch.no_grad()
    def recommend(self, sequence_idx: list[int], k: int, exclude: set[int] | None = None) -> list[int]:
        """
        Score the next item for one session; return top-k item indices (never PAD,
        never in `exclude`). Mirrors GRU4Rec.recommend so eval code is shared.
        """
        self.eval()
        exclude = set(exclude or set()) | {PAD_IDX}
        x = torch.tensor([sequence_idx], dtype=torch.long)
        logits = self.forward(x).squeeze(0)                  # (n,)
        logits[list(exclude)] = float("-inf")
        k_eff = min(k, logits.numel() - len(exclude))
        if k_eff <= 0:
            return []
        return torch.topk(logits, k_eff).indices.tolist()
