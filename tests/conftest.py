"""
Shared pytest configuration for the ml-system-design test suite.

Puts both phase0/ and phase1/ on sys.path so the phase modules import the same
way they do when you run `python run.py` from inside each phase directory.

All tests use small synthetic DataFrames -- none of them touch the multi-hundred-MB
CSVs in data/. That keeps the suite fast and runnable in CI without the dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "phase0"))
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT / "phase2"))
sys.path.insert(0, str(ROOT / "phase3"))
sys.path.insert(0, str(ROOT / "phase4"))
sys.path.insert(0, str(ROOT / "phase5"))
sys.path.insert(0, str(ROOT / "phase6"))
sys.path.insert(0, str(ROOT / "phase7"))
sys.path.insert(0, str(ROOT / "phase8"))
sys.path.insert(0, str(ROOT / "phase9"))
sys.path.insert(0, str(ROOT / "phase10"))

DAY_MS = 86_400 * 1000


@pytest.fixture
def synthetic_events() -> pl.DataFrame:
    """
    A tiny, deterministic event log in the canonical schema.

    Two users:
      - u1: exposures + engagements + a target action on items i1/i2 (warm, category A)
      - u2: a single exposure (effectively cold once we filter to positive signals)
    Timestamps increase so temporal_split is meaningful.
    """
    base = 1_600_000_000_000
    rows = [
        # user_id, event_type, item_id, offset_days
        ("u1", "weak",   "i1", 0),
        ("u1", "weak",   "i2", 1),
        ("u1", "medium", "i1", 2),
        ("u1", "medium", "i2", 3),
        ("u1", "strong", "i1", 4),
        ("u2", "weak",   "i3", 5),
        ("u1", "medium", "i2", 6),
        ("u1", "strong", "i2", 7),
    ]
    return pl.DataFrame(
        {
            "user_id":      [r[0] for r in rows],
            "event_type":   [r[1] for r in rows],
            "item_id":      [r[2] for r in rows],
            "timestamp_ms": [base + r[3] * DAY_MS for r in rows],
        }
    ).with_columns(
        pl.col("user_id").cast(pl.Utf8),
        pl.col("item_id").cast(pl.Utf8),
        pl.col("timestamp_ms").cast(pl.Int64),
    )


@pytest.fixture
def synthetic_item_properties() -> pl.DataFrame:
    """Item -> categoryid property records in the canonical schema."""
    base = 1_600_000_000_000
    rows = [
        ("i1", "categoryid", "A", 0),
        ("i2", "categoryid", "A", 0),
        ("i3", "categoryid", "B", 0),
    ]
    return pl.DataFrame(
        {
            "item_id":      [r[0] for r in rows],
            "property":     [r[1] for r in rows],
            "value":        [r[2] for r in rows],
            "timestamp_ms": [base + r[3] * DAY_MS for r in rows],
        }
    ).with_columns(
        pl.col("item_id").cast(pl.Utf8),
        pl.col("timestamp_ms").cast(pl.Int64),
    )
