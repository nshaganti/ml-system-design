"""
Phase 1 — Embedding Index (ANN Search)
=========================================
Brute-force dot-product search over the item embedding matrix.

In production this becomes Qdrant or Milvus — same interface, orders of magnitude
faster for 50M+ items. For 235K items with 64-dim embeddings, numpy matmul on CPU
runs in ~5-10ms per query, which works fine for evaluation.

The interface is intentionally minimal: build once, search many times.
"""

from __future__ import annotations

import numpy as np
import torch


class EmbeddingIndex:
    """
    Wraps the trained item embedding matrix for fast nearest-neighbor lookup.

    At serving time (production):
      1. User history → mean-pool item embeddings → query vector (dim,)
      2. This index: query vector @ item_matrix.T → top-K item IDs

    In production (Qdrant/Milvus):
      - Same query → ANN search with HNSW graph → sub-millisecond at 50M items
      - Here: exact search via numpy matmul → ~5ms at 235K items (acceptable for eval)
    """

    def __init__(
        self,
        item_embeddings: torch.Tensor,   # (n_items, dim)
        idx_to_item_id: list[str],
    ):
        # Normalize for cosine similarity search
        # (makes scores interpretable and prevents high-magnitude items dominating)
        self._matrix      = self._normalize(item_embeddings.numpy())  # (n_items, dim)
        self.idx_to_item_id = idx_to_item_id
        self.n_items      = len(idx_to_item_id)

        print(
            f"[index] Built index: {self.n_items:,} items × {self._matrix.shape[1]}-dim "
            f"| matrix size: {self._matrix.nbytes / 1024**2:.1f} MB"
        )

    def search(
        self,
        query_vec: np.ndarray,    # (dim,)
        k: int = 500,
        exclude_item_ids: set[str] | None = None,
    ) -> list[str]:
        """
        Return top-k item IDs by dot product with query_vec.
        Optionally exclude items the user has already interacted with.

        This is Stage 1 (Candidate Generation) in the two-stage architecture —
        50M items → 500 candidates in production, 235K → 500 here.
        """
        query_norm = self._normalize(query_vec.reshape(1, -1)).flatten()

        # Dot product with all items: (dim,) @ (dim, n_items) = (n_items,)
        scores = self._matrix @ query_norm

        # Sort descending — argpartition is faster than full argsort for large N
        top_k_idx = np.argpartition(scores, -min(k * 2, self.n_items))[-(k * 2):]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]

        # Decode and filter exclusions
        results = []
        exclude = exclude_item_ids or set()
        for idx in top_k_idx:
            item_id = self.idx_to_item_id[idx]
            if item_id not in exclude:
                results.append(item_id)
            if len(results) >= k:
                break

        return results

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """L2-normalize rows. Handles zero vectors safely."""
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        return np.where(norms > 0, x / norms, x)
