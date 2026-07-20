"""
Phase 10 -- Turning sessions into GRU4Rec training tensors.
===========================================================
Reuses Phase 7's sessionization (DRY) so "a session" means exactly the same thing
here as in the co-visitation phase -- that's what makes the head-to-head fair.

Vocabulary: item_id -> integer index in 1..V (index 0 is the PAD token, see
model.py). Items unseen in training map to nothing and are dropped at encode time,
exactly as a real serving system would treat an unknown id.

Training examples are next-item windows: for each session [a, b, c, d] we emit
(a -> b), (a b -> c), (a b c -> d). Each input is left-padded/truncated to
`max_len`. This is the standard "predict the next item given the prefix" setup and
lines up 1:1 with the leave-one-out eval protocol.
"""

from __future__ import annotations

from model import PAD_IDX


def build_vocab(item_lists: list[list[str]], min_count: int = 1) -> dict[str, int]:
    """Map item_id -> index in 1..V (0 reserved for PAD). Keeps items seen >= min_count."""
    counts: dict[str, int] = {}
    for items in item_lists:
        for it in items:
            counts[it] = counts.get(it, 0) + 1
    vocab: dict[str, int] = {}
    for it, c in counts.items():
        if c >= min_count:
            vocab[it] = len(vocab) + 1   # start at 1; 0 == PAD
    return vocab


def _dedup_keep_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def encode_context(context: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    """
    Encode a context session into a left-padded index list of length `max_len`.
    Out-of-vocab items are dropped; keeps the most recent `max_len` items.
    Returns all-PAD if nothing survives (caller should back off to popularity).
    """
    idx = [vocab[it] for it in context if it in vocab]
    idx = idx[-max_len:]
    return [PAD_IDX] * (max_len - len(idx)) + idx


def make_training_pairs(
    item_lists: list[list[str]], vocab: dict[str, int], max_len: int,
    max_pairs: int | None = None,
) -> list[tuple[list[int], int]]:
    """
    Build (padded_prefix, target_idx) pairs from sessions. Both prefix items and
    the target must be in-vocab. Optionally cap the total number of pairs (keeps
    training snappy on CPU).
    """
    pairs: list[tuple[list[int], int]] = []
    for items in item_lists:
        seq = [vocab[it] for it in _dedup_keep_order(items) if it in vocab]
        for t in range(1, len(seq)):
            prefix = seq[:t][-max_len:]
            padded = [PAD_IDX] * (max_len - len(prefix)) + prefix
            pairs.append((padded, seq[t]))
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs
