"""
data_sources.synthetic -- the DEFAULT, domain-neutral dataset
==============================================================
A seeded, in-memory generator so the whole repo runs with ZERO downloads. This is
what makes the canon demonstrable anywhere (and, later, makes off-policy evaluation
verifiable -- Part II adds a known logging policy on top of this).

It is deliberately *not* e-commerce. It models the generic recommender funnel:

    exposure (WEAK)  ->  engagement (MEDIUM)  ->  target action (STRONG)

with structure the phases actually need:
  * skewed item popularity (zipf-ish)          -> popularity baselines mean something
  * per-user category preference               -> the cross feature can personalize
  * sessions (a user's day)                     -> co-visitation signal (Phase 7)
  * a funnel                                    -> WEAK impressions + rarer positives
  * a timeline                                  -> temporal split works

Size/seed via env: SYNTH_USERS, SYNTH_ITEMS, SYNTH_DAYS, SYNTH_SEED.
No files are read; `data_dir` is ignored.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

from signals import WEAK, MEDIUM, STRONG

KNOWN_EVENT_TYPES = {WEAK, MEDIUM, STRONG}

_N_CATEGORIES = 20
_BASE_MS = 1_500_000_000_000  # arbitrary epoch-ms anchor


def _params() -> tuple[int, int, int, int]:
    return (
        int(os.environ.get("SYNTH_USERS", 2000)),
        int(os.environ.get("SYNTH_ITEMS", 800)),
        int(os.environ.get("SYNTH_DAYS", 90)),
        int(os.environ.get("SYNTH_SEED", 7)),
    )


@lru_cache(maxsize=4)
def _generate(n_users: int, n_items: int, n_days: int, seed: int):
    rng = np.random.default_rng(seed)

    item_ids = np.array([f"item_{i:05d}" for i in range(n_items)])
    item_cat = rng.integers(0, _N_CATEGORIES, n_items)
    # Skewed popularity (zipf-ish), shuffled so it's not correlated with id order.
    pop = 1.0 / (1.0 + np.arange(n_items))
    rng.shuffle(pop)
    pop /= pop.sum()

    user_ids = [f"user_{u:05d}" for u in range(n_users)]
    user_pref_cat = rng.integers(0, _N_CATEGORIES, n_users)

    ts, uid, etype, iid = [], [], [], []
    for u, user in enumerate(user_ids):
        n_sessions = 1 + rng.poisson(3)
        for _ in range(n_sessions):
            day = int(rng.integers(0, n_days))
            t = _BASE_MS + day * 86_400_000 + int(rng.integers(0, 86_400_000))
            slate = rng.choice(n_items, size=1 + rng.poisson(4), replace=False, p=pop)
            for idx in slate:
                # Every shown item is an exposure (WEAK).
                ts.append(t); uid.append(user); etype.append(WEAK); iid.append(item_ids[idx])
                # Engagement is likelier when the item matches the user's taste.
                match = (item_cat[idx] == user_pref_cat[u])
                p_click = 0.55 if match else 0.08
                if rng.random() < p_click:
                    ts.append(t + 1); uid.append(user); etype.append(MEDIUM); iid.append(item_ids[idx])
                    # A fraction of engagements convert to the target action.
                    if rng.random() < 0.35:
                        ts.append(t + 2); uid.append(user); etype.append(STRONG); iid.append(item_ids[idx])

    events = pl.DataFrame(
        {"timestamp_ms": ts, "user_id": uid, "event_type": etype, "item_id": iid}
    ).with_columns(pl.col("timestamp_ms").cast(pl.Int64)).sort("timestamp_ms")

    item_props = pl.DataFrame(
        {"item_id": item_ids, "value": [str(c) for c in item_cat]}
    ).with_columns([
        pl.lit(0).cast(pl.Int64).alias("timestamp_ms"),
        pl.lit("categoryid").alias("property"),
    ]).select(["timestamp_ms", "item_id", "property", "value"])

    # ~5% of items are ineligible to show -- the policy layer's job (Rule 15).
    # Domain-neutral: could be availability, compliance, safety, embargo, etc.
    eligible_flag = (rng.random(n_items) > 0.05).astype(int)
    eligible_props = pl.DataFrame(
        {"item_id": item_ids, "value": [str(e) for e in eligible_flag]}
    ).with_columns([
        pl.lit(0).cast(pl.Int64).alias("timestamp_ms"),
        pl.lit("eligible").alias("property"),
    ]).select(["timestamp_ms", "item_id", "property", "value"])

    return events, pl.concat([item_props, eligible_props])


def load_events(data_dir: Path | None = None) -> pl.DataFrame:
    events, _ = _generate(*_params())
    print(f"[load_data] (synthetic) Generated {len(events):,} events "
          f"(WEAK/MEDIUM/STRONG funnel). No download needed.")
    return events


def load_item_properties(data_dir: Path | None = None) -> pl.DataFrame:
    _, props = _generate(*_params())
    return props
