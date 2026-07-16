# Phase 2 -- The Ranker and the Feature Store

> **Google's Rules 29-37** are all about one enemy: **training-serving skew.**
> This phase is where you meet it, measure it, and defeat it. If you take one
> idea from this whole repo, take this one.

Phase 1 gave us candidates. Phase 2 does two things:

1. Builds a **point-in-time feature store** -- the most under-taught concept in
   production ML -- and *quantifies* the skew bug it prevents.
2. Trains an **interpretable logistic-regression ranker** (Stage 2) that reorders
   candidates and finally beats popularity.

---

## Part 1: The bug that silently kills ML systems

Here is the mistake almost everyone makes when generating training data:

```python
# You have click events from the last 30 days.
# You join them to item stats... from TODAY.
item_stats = spark.read.table("item_stats_current")   # <- today's numbers
training_data = events_30d.join(item_stats, on="item_id")
```

Spot the bug? An event from 30 days ago gets today's `item_popularity`. But 30
days ago, that item's popularity was *different* -- maybe it hadn't gone viral
yet. **The model trains on feature values that did not exist at the time of the
event, and will never exist at serving time.** Offline eval looks fantastic
(the features leaked the answer); production quietly underperforms; nobody can
explain the gap.

This is **training-serving skew**, and it is invisible -- no exception, no crash.

### The fix: point-in-time correctness

A feature store guarantees that when you join a feature to a historical event,
you get the value **as it existed at that event's timestamp** -- never later.

Our `phase2/feature_store.py` implements this with polars `join_asof`, which is
purpose-built for it: for an event at time `T`, grab the most recent feature-
timeline row with `timestamp <= T`.

```python
# For each entity, a timeline of PRIOR cumulative counts.
# The k-th event (0-based) of an item has "prior count" = k.
# A backward as-of join at time T therefore returns the count of events
# STRICTLY BEFORE T -- no leakage of the current or any future event.
timeline = (strong
    .sort("timestamp_ms")
    .with_columns(pl.int_range(0, pl.len()).over(keys).alias(feat)))

features = entity_df.join_asof(timeline, on="timestamp_ms", by="item_id",
                               strategy="backward")
```

### Seeing the skew with your own eyes

`run.py` computes the same features **both ways** on the same rows -- point-in-
time correct (`get_historical_features`) vs the naive "join today's totals"
(`get_skewed_features`) -- and prints the difference:

```
TRAINING-SERVING SKEW (point-in-time vs leaked)
  item_pop           point-in-time mean= 3.18 | leaked mean= 6.92 | inflation x2.2
  user_pop           point-in-time mean=24.99 | leaked mean=50.91 | inflation x2.0
  user_cat_affinity  point-in-time mean= 0.63 | leaked mean= 1.76 | inflation x2.8
```

**The leaked features are inflated 2-2.8x.** Read that again: if you'd built this
the naive way, every training row would have carried roughly *double* the true
historical signal. Your model would learn relationships that don't hold at
serving time. This number -- x2.2 -- is training-serving skew made concrete.

> **Why the cross feature (x2.8) leaks worst:** category affinity accumulates
> over a user's whole life. Crediting a month-old event with a user's *lifetime*
> affinity is the most anachronistic of all. Rich, personal features are exactly
> the ones point-in-time correctness protects most.

### The same store, three doors

| Method | Used at | Correctness |
|---|---|---|
| `get_historical_features` | **training** | point-in-time correct (the right way) |
| `get_online_features` | **serving** | latest values as of the cutoff |
| `get_skewed_features` | **teaching only** | deliberately wrong, to measure skew |

In a real system these are Feast's offline store (Iceberg/S3) and online store
(Redis). The concept is identical; only the storage changes.

---

## Part 2: The ranker (Stage 2)

Now the second stage of the two-stage architecture. Given ~500 candidates, score
each `(user, item)` pair with `P(strong interaction)` and take the top 20.

### Why logistic regression, not XGBoost or a DNN?

> **Rules 4 & 14:** Start with an interpretable model.

When an LR ranker makes a mistake, you read the weights and understand why. When
XGBoost makes a mistake, you debug a forest of 500 trees. At this stage the
accuracy gap is small and the debuggability gap is enormous. Earn complexity
later.

And interpretability paid off immediately -- here are our learned weights:

```
weights = {item_pop: 3.08, user_cat_affinity: 2.93, user_pop: -0.35}
```

You can *read the model's mind*:

- **`item_pop` (+3.08):** popular items convert. No surprise.
- **`user_cat_affinity` (+2.93):** a user's history in the item's category is
  *almost as predictive as global popularity*. This is the personalization signal.
- **`user_pop` (-0.35):** slightly **negative** -- very active users are pickier
  per-item (they've already bought a lot). You'd never notice this in a black box.

### The cross feature: where personalization actually comes from

Our first ranker had only `item_pop` and `user_pop`, and it **tied popularity
exactly.** That's not a bug -- it's math. For a single user, `user_pop` is a
constant, and `item_pop` is the same value popularity already sorts by. So
per-user, ranking by the model *is* ranking by popularity. **Features that are
constant along one axis cannot personalize.**

> **Rule 20:** *Combine and modify existing features to create new ones in
> human-understandable ways.*

The fix is a **cross feature** -- one that varies per `(user, item)` pair:
`user_cat_affinity` = how many of *this user's* prior strong events were in
*this item's* category. Now the same popular pool gets reordered differently for
each user, based on what they actually care about.

---

## Results

Same candidate pool for both; the only difference is whether the LR ranker
reorders it:

| Metric | Popularity order | LR ranker | Lift |
|---|---|---|---|
| Recall@20 | 0.0260 | **0.0270** | +3.8% |
| NDCG@20 | 0.0181 | **0.0185** | +2% |

Modest, but **real and positive** -- and it comes purely from personalized
reordering. With a better candidate pool (Phase 1's two-tower instead of raw
popularity) and more cross features (price-vs-user-average, brand affinity,
recency), this is where most of a production system's lift is won.

> **Honest framing:** +2% NDCG isn't glamorous, but it's *trustworthy* -- it's
> measured temporally, on leakage-free features, with an interpretable model
> whose weights make sense. In production, a trustworthy +2% you can explain
> beats a suspicious +20% you can't.

---

## What Phase 2 taught us

1. **Training-serving skew is real and measurable** -- 2-2.8x on this dataset.
   A feature store isn't bureaucracy; it's the thing standing between you and a
   silently-wrong model.
2. **Point-in-time correctness has a name and a tool** (`join_asof`). You now
   know how to build it, not just cite it.
3. **Interpretable models earn their keep** -- the negative `user_pop` weight is
   an insight you only get from a model you can read.
4. **Personalization requires cross features.** Single-axis features can't
   personalize, no matter how fancy your model.

Next up ([roadmap](../README.md#roadmap)): serving (Ray Serve), monitoring
(drift detection), and A/B testing -- turning this offline pipeline into a live
system.
