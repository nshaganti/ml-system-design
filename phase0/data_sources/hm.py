"""
data_sources.hm -- H&M Personalized Fashion loader
====================================================
Maps the H&M Kaggle dataset to our canonical schema, so every phase runs on it
unchanged. Download (accept the competition rules first):
    https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
Drop into data/:
    transactions_train.csv   (~3.5 GB: t_dat, customer_id, article_id, price, sales_channel_id)
    articles.csv             (article_id + rich metadata)
    customers.csv            (customer_id + age, club status, ...)

Key differences from Retail Rocket (documented in docs/dataset-hm.md):
  * PURCHASES ONLY -- no views/carts. So `event_type` is always "purchase".
  * DAILY timestamp granularity (t_dat is a date). Same-day purchases share a
    timestamp; Phase 7 sessionization treats a customer's day as one session,
    which is a sensible basket definition for this data.
  * STATIC item metadata (articles.csv). We emit it as item_properties stamped at
    time 0 (known for all time), mapping product_type_name -> "categoryid" so the
    feature store's category-affinity cross feature works with no code change.

Env knobs:
  * HM_MAX_ROWS -- keep only the most recent N transactions (for smaller machines).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl

KNOWN_EVENT_TYPES = {"purchase"}

# Article metadata columns we surface as item_properties (long format).
# The first one is mapped to "categoryid" for the feature store cross feature.
_CATEGORY_COL = "product_type_name"
_EXTRA_META_COLS = [
    "product_group_name",
    "colour_group_name",
    "department_name",
    "garment_group_name",
    "index_name",
]


def load_events(data_dir: Path) -> pl.DataFrame:
    """transactions_train.csv -> canonical [timestamp_ms, user_id, event_type, item_id]."""
    path = data_dir / "transactions_train.csv"
    _assert_file_exists(path)

    df = (
        pl.read_csv(path, infer_schema_length=1000)
        .select(["t_dat", "customer_id", "article_id"])
        .with_columns([
            pl.col("t_dat").str.to_datetime("%Y-%m-%d", strict=False).dt.epoch("ms").alias("timestamp_ms"),
            pl.col("customer_id").cast(pl.Utf8).alias("user_id"),
            pl.col("article_id").cast(pl.Utf8).alias("item_id"),
            pl.lit("purchase").alias("event_type"),
        ])
        .select(["timestamp_ms", "user_id", "event_type", "item_id"])
        .sort("timestamp_ms")
    )

    max_rows = os.environ.get("HM_MAX_ROWS")
    if max_rows:
        n = int(max_rows)
        if len(df) > n:
            df = df.tail(n)   # keep the most recent N -> preserves temporal structure
            print(f"[load_data] (hm) HM_MAX_ROWS set -> using most recent {n:,} transactions")

    _validate_events(df)
    return df


def load_item_properties(data_dir: Path) -> pl.DataFrame:
    """
    articles.csv -> long [timestamp_ms, item_id, property, value].
    Static metadata stamped at time 0 (available before every event). The
    category column is renamed to "categoryid" for the feature store.
    """
    path = data_dir / "articles.csv"
    _assert_file_exists(path)

    cols = [_CATEGORY_COL] + _EXTRA_META_COLS
    articles = pl.read_csv(path, infer_schema_length=1000)
    present = [c for c in cols if c in articles.columns]

    wide = articles.select(
        [pl.col("article_id").cast(pl.Utf8).alias("item_id")]
        + [pl.col(c).cast(pl.Utf8) for c in present]
    ).rename({_CATEGORY_COL: "categoryid"} if _CATEGORY_COL in present else {})

    value_cols = [("categoryid" if c == _CATEGORY_COL else c) for c in present]
    long = wide.unpivot(
        index="item_id", on=value_cols,
        variable_name="property", value_name="value",
    ).with_columns([
        pl.lit(0).cast(pl.Int64).alias("timestamp_ms"),
        pl.col("value").cast(pl.Utf8),
    ]).select(["timestamp_ms", "item_id", "property", "value"])

    return long.drop_nulls("value").sort("timestamp_ms")


def _assert_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing: {path}\n"
            "Download (accept rules first): "
            "https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations\n"
            "Place transactions_train.csv, articles.csv, customers.csv in data/."
        )


def _validate_events(df: pl.DataFrame) -> None:
    unknown = set(df["event_type"].unique().to_list()) - KNOWN_EVENT_TYPES
    if unknown:
        raise ValueError(f"Unknown event types found: {unknown}")
    nulls = {c: df[c].null_count() for c in ["user_id", "item_id", "timestamp_ms"]}
    if any(v > 0 for v in nulls.values()):
        raise ValueError(f"Null values in critical columns: {nulls}")
    print(f"[load_data] (hm) Loaded {len(df):,} events (purchases). Schema OK.")


# ---------------------------------------------------------------------------
# Synthetic H&M-shaped data generator (for tests + smoke runs ONLY).
# This is NOT real data -- it produces the SAME schema with baked-in structure
# (session baskets, category-correlated co-purchases) so the pipeline can be
# exercised end-to-end before the real 3.5 GB CSVs are downloaded. Any numbers
# it produces are illustrative, never headline results.
# ---------------------------------------------------------------------------

def generate_synthetic(
    data_dir: Path,
    n_customers: int = 800,
    n_articles: int = 400,
    n_days: int = 60,
    avg_baskets_per_customer: int = 4,
    seed: int = 42,
) -> None:
    """Write synthetic transactions_train.csv / articles.csv / customers.csv."""
    rng = np.random.default_rng(seed)
    data_dir.mkdir(parents=True, exist_ok=True)

    product_types = ["Trousers", "Dress", "Sweater", "T-shirt", "Jacket",
                     "Shoes", "Skirt", "Shorts", "Hoodie", "Socks"]
    groups = ["Garment Lower body", "Garment Full body", "Garment Upper body",
              "Garment Upper body", "Outerwear", "Shoes", "Garment Lower body",
              "Garment Lower body", "Garment Upper body", "Socks & Tights"]
    colours = ["Black", "White", "Blue", "Red", "Beige", "Green"]
    departments = ["Womenswear", "Menswear", "Kids", "Divided"]

    # Articles: each gets a product type (-> category) with a correlated group.
    art_ids = [f"{100000000 + i}" for i in range(n_articles)]
    art_type_idx = rng.integers(0, len(product_types), n_articles)
    articles = pl.DataFrame({
        "article_id": art_ids,
        "product_type_name": [product_types[i] for i in art_type_idx],
        "product_group_name": [groups[i] for i in art_type_idx],
        "colour_group_name": [colours[i] for i in rng.integers(0, len(colours), n_articles)],
        "department_name": [departments[i] for i in rng.integers(0, len(departments), n_articles)],
        "garment_group_name": [groups[i] for i in art_type_idx],
        "index_name": [departments[i] for i in rng.integers(0, len(departments), n_articles)],
    })
    articles.write_csv(data_dir / "articles.csv")

    # Customers.
    cust_ids = [f"cust_{i:06d}" for i in range(n_customers)]
    customers = pl.DataFrame({
        "customer_id": cust_ids,
        "age": rng.integers(16, 75, n_customers),
        "club_member_status": rng.choice(["ACTIVE", "PRE-CREATE"], n_customers),
        "fashion_news_frequency": rng.choice(["NONE", "Regularly"], n_customers),
    })
    customers.write_csv(data_dir / "customers.csv")

    # Transactions: each customer has a few "baskets" (a day). Within a basket,
    # articles are drawn with a bias toward one product type -> real co-vis signal.
    # Popularity is skewed (zipf-ish) so popularity baselines are meaningful.
    pop_weights = 1.0 / (1.0 + np.arange(n_articles))
    rng.shuffle(pop_weights)
    pop_weights /= pop_weights.sum()
    art_arr = np.array(art_ids)

    rows = []
    base_day = 1_500_000_000_000  # arbitrary epoch-ms anchor
    for cust in cust_ids:
        n_baskets = 1 + rng.poisson(avg_baskets_per_customer)
        for _ in range(n_baskets):
            day = int(rng.integers(0, n_days))
            ts_day = base_day + day * 86_400_000
            fav_type = rng.integers(0, len(product_types))
            type_mask = (art_type_idx == fav_type)
            basket_size = 1 + rng.poisson(2)
            for _ in range(basket_size):
                # 70% from the customer's favored type, 30% by global popularity.
                if rng.random() < 0.7 and type_mask.any():
                    idx = rng.choice(np.where(type_mask)[0])
                else:
                    idx = rng.choice(n_articles, p=pop_weights)
                rows.append((ts_day, cust, art_arr[idx]))

    tx = pl.DataFrame(
        {"t_ms": [r[0] for r in rows], "customer_id": [r[1] for r in rows],
         "article_id": [r[2] for r in rows]}
    ).with_columns(
        pl.from_epoch("t_ms", time_unit="ms").dt.strftime("%Y-%m-%d").alias("t_dat"),
        pl.lit(1).alias("sales_channel_id"),
        pl.lit(0.02).alias("price"),
    ).select(["t_dat", "customer_id", "article_id", "price", "sales_channel_id"])
    tx.write_csv(data_dir / "transactions_train.csv")
