"""
Reproducibility helper (Rule: pin every source of randomness)
=============================================================
A single place to seed Python, NumPy and PyTorch so a fresh install reproduces
the neural numbers. The review of this repo correctly flagged that we seeded the
data pipeline (NumPy/`random`) but never called `torch.manual_seed`, so weight
init and DataLoader shuffling were free to drift between machines. This closes
that seam.

Import from any phase that already puts `phase0/` on `sys.path`:

    from repro import set_global_seed
    set_global_seed(42)
"""

from __future__ import annotations

import os
import random

import numpy as np

SEED = 42


def set_global_seed(seed: int = SEED, deterministic: bool = True) -> int:
    """Seed every RNG we touch. Returns the seed for logging.

    `deterministic=True` asks PyTorch for deterministic kernels (best-effort:
    `warn_only` so a missing deterministic CUDA kernel degrades to a warning
    rather than crashing a CPU-only teaching run).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # warn_only: keep CPU-only / MPS teaching runs alive if a kernel
            # has no deterministic implementation.
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass  # numpy-only phases don't need torch
    return seed
