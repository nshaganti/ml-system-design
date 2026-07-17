# ML System Design: Real-Time Recommendation Engine

A practical end-to-end guide for ML engineers transitioning from prototype notebooks to production systems. Uses a real-time product recommendation system as the vehicle -- the same class of problem operated at scale by large e-commerce platforms.

---

## The Scenario

| Constraint | Value |
|---|---|
| Daily active users | 10M |
| Product catalog size | 50M items |
| Recommendation events/day | ~500M |
| p99 latency budget | 100ms end-to-end |
| Freshness requirement | User actions affect recommendations within ~15 min |
| Cold-start rate | ~30% of sessions are new/anonymous users |

**Goal:** When a user lands on the homepage or a product page, show them 20 ranked product recommendations that maximize add-to-cart rate.

---

## Core Insight: The Model Is Not the Hard Part

The biggest mistake prototype ML engineers make in production is thinking the hard part is the model. The real hard parts are:

1. Getting clean, consistent data to the model -- at training time *and* serving time
2. Detecting when something silently breaks
3. Knowing which version of your model is causing which outcome

> **Google's Rule 4:** "The first model provides the biggest lift, so make it easy to get out the door." The infrastructure you build in Phase 1 is what you'll use forever. The model inside it will change every week.

---

## Phase 0: Launch Without ML (Rule 1)

Before touching ML, ship something. This is not a placeholder -- it has two real purposes:

1. **Collects training data.** You cannot train a personalized model until users have interacted with something.
2. **Establishes a baseline.** You will use this to measure your first model's actual lift.

### Heuristic Ranker

```python
def heuristic_recommendations(user_id, catalog):
    # Rule 7: Encode domain knowledge as features, not discarded logic
    top_items = catalog \
        .filter(in_stock=True) \
        .filter(launched_within_days=30) \
        .sort_by("purchases_last_7d", descending=True) \
        .limit(100)

    # Personalization via category affinity (still no ML)
    user_cats = get_user_purchase_history_categories(user_id)
    if user_cats:
        top_items = top_items.filter(category__in=user_cats)

    return top_items[:20]
```

### Instrument Everything Now (Rule 2)

This is the single most important investment of Phase 0. Log every impression, click, add-to-cart, and purchase with a `recommendation_request_id` that ties each event back to the exact set of items shown.

**Minimum event schema:**

```json
{
  "event_id": "uuid",
  "recommendation_request_id": "uuid",
  "user_id": "hashed_id",
  "session_id": "uuid",
  "event_type": "impression | click | add_to_cart | purchase",
  "item_id": "string",
  "position": 3,
  "timestamp_ms": 1720000000000,
  "context": {
    "page_type": "homepage | pdp | cart",
    "device": "mobile | desktop"
  }
}
```

> **Why `position` matters:** You will need it later to remove position bias from your training data (Rule 36).

**OSS stack for Phase 0:**
- **Apache Kafka** -- stream events from the application
- **Apache Iceberg on S3/GCS** -- durable event store for training data

> **In practice (this repo).** We implemented Phase 0 on the Retail Rocket
> dataset in [`phase0/`](phase0/). The heuristic scores **Recall@20 = 0.0310**
> with an honest temporal split -- that is the baseline every later model must
> beat. We also learned that catalog coverage (1.4%) matters as much as recall:
> a popularity ranker is a "popularity trap" that never surfaces the long tail.
> Full walkthrough: [`docs/phase0.md`](docs/phase0.md).

---

## Phase 1: First ML Pipeline (Rules 4, 5, 14)

### The Two-Stage Architecture

At 50M items, no model can score every item for every user in 100ms. All production recommendation systems split into two stages:

```
Request
  |
  v
+----------------------------------------------------------+
|  Stage 1: Candidate Generation (~10ms)                   |
|  Goal: 50M items --> 500-1000 candidates                 |
|  Model: Cheap retrieval (embedding ANN search or rules)  |
+---------------------------+------------------------------+
                            | 500-1000 candidates
                            v
+----------------------------------------------------------+
|  Stage 2: Ranking (~50ms)                                |
|  Goal: 500-1000 candidates --> 20 ranked results         |
|  Model: Expensive pointwise ranker                       |
+---------------------------+------------------------------+
                            | 20 items
                            v
                        Response
```

This separation matters not just for latency -- the two stages can evolve independently.

### Stage 1: Candidate Generation -- Two-Tower Model

```
User Tower:    user_id --> embedding(user_id) --> 128-dim vector
Item Tower:    item_id --> embedding(item_id) --> 128-dim vector

Training signal: did user interact with item? (implicit feedback)
Loss: sampled softmax or BPR (Bayesian Personalized Ranking)
```

At serving time, pre-compute all item embeddings and load into a vector index. A user query = compute user embedding, run ANN search, return top-500 items.

**OSS choices:**
- Training: **PyTorch** -- dominant, huge ecosystem, production-proven
- Vector index: **Qdrant** or **Milvus** -- both handle 50M vectors. Qdrant is simpler to operate; Milvus is more battle-tested at larger scale.

### Stage 2: Ranker -- Start with Logistic Regression (Rule 14)

```python
features = {
    # User features
    "user_purchase_count_30d": float,
    "user_avg_order_value": float,
    "user_category_affinity_match": float,

    # Item features
    "item_purchase_rate_7d": float,
    "item_click_through_rate": float,
    "item_days_since_launched": int,
    "item_in_stock": bool,
    "item_price_tier": int,   # 1-5 bucket

    # Context features
    "hour_of_day": int,
    "day_of_week": int,
    "is_mobile": bool,

    # Cross features (Rule 20)
    "user_has_bought_same_brand": bool,
    "item_price_vs_user_avg_order": float,
}

# Model: LogisticRegression predicting P(click | user, item, context)
```

**Why logistic regression first, not XGBoost or a neural net?**
When the ranker makes a mistake, logistic regression tells you *why* -- you can inspect the weights. When XGBoost makes a mistake you debug a forest of 500 trees. The lift difference at this stage is small; the debuggability difference is enormous.

### Training Pipeline

```
+----------------+    +---------------+    +---------------+    +---------------+
|  Event Store   |--->|  Spark job    |--->|  Feature eng  |--->|  Training job |
|  (Iceberg)     |    |  (join logs)  |    |  (features    |    |  (PyTorch/    |
+----------------+    +---------------+    |  above)       |    |  sklearn)     |
                                           +---------------+    +-------+-------+
                                                                        |
                                                               +--------v--------+
                                                               |  MLflow Model  |
                                                               |  Registry      |
                                                               +-----------------+
```

**OSS:**
- **Apache Airflow** -- schedules the daily retrain
- **MLflow** -- tracks every run: hyperparameters, metrics, model artifacts, lineage

> **In practice (this repo).** Our two-tower candidate generator lives in
> [`phase1/`](phase1/) with MLflow tracking. The humbling result: it **lost** to
> the Phase 0 heuristic (Recall@20 0.0228 vs 0.0310), even after popularity-
> weighted negatives (+27% warm recall) and fixing a train/serve dot-product
> mismatch. That is a normal, legitimate Phase 1 outcome -- a strong heuristic is
> a hard baseline, and beating it needs side features, not just embeddings. The
> lasting deliverable is the *pipeline*, not the model. Full story (with the
> debugging steps): [`docs/phase1.md`](docs/phase1.md).

---

## Phase 2: The Feature Store -- The Most Under-Taught Concept

### The Training-Serving Skew Problem (Rules 29-37)

Here is the bug that is silently killing most ML systems in production:

**At training time (broken approach):**

```python
# You join events with item stats FROM TODAY
item_stats = spark.read.table("item_stats_current")
events = spark.read.table("click_events_last_30d")
training_data = events.join(item_stats, on="item_id")
# item_stats is TODAY's data, but events are from 30 days ago.
# The model learns on a different distribution than it will see at serving time.
```

**The result:** The model performs great in offline eval, then degrades in production. Nobody knows why. This is training-serving skew.

### The Solution: A Feature Store

A feature store provides two guarantees:

1. **Point-in-time correctness at training time** -- when you join features to historical events, you get feature values *as they existed at the moment of each event*.
2. **Low-latency serving at inference time** -- the same features are available in under 5ms via an online store.

```
                              +------------------------------------------+
                              |           Feature Store (Feast)          |
                              |                                          |
  Batch pipelines ----------->|  Offline Store (Iceberg/S3)             |
  (Spark, daily)              |  Point-in-time correct joins for        |
                              |  training data generation               |
                              |                                          |
  Streaming pipelines ------->|  Online Store (Redis)                   |
  (Flink/Kafka, ~1min lag)    |  <5ms feature lookup at serving time   |
                              |                                          |
                              |  Shared feature definitions             |
                              |  (one codebase, Rule 32)               |
                              +------------------------------------------+
```

**OSS choice:** [Feast](https://feast.dev/) is the most mature open-source feature store.

### Feature Definition (write once, use everywhere -- Rule 32)

```python
from feast import Feature, FeatureView, Entity

user_features = FeatureView(
    name="user_features",
    entities=["user_id"],
    ttl=timedelta(days=30),
    features=[
        Feature(name="purchase_count_30d", dtype=ValueType.INT64),
        Feature(name="avg_order_value", dtype=ValueType.FLOAT),
        Feature(name="top_category", dtype=ValueType.STRING),
    ],
    batch_source=BigQuerySource(table="user_feature_table"),
    stream_source=KafkaSource(topic="user-events"),
)

# At training time -- point-in-time correct historical values
training_df = store.get_historical_features(
    entity_df=click_events_df,   # has user_id + event_timestamp
    features=["user_features:purchase_count_30d", ...]
).to_df()

# At serving time -- current values in <5ms
features = store.get_online_features(
    features=["user_features:purchase_count_30d", ...],
    entity_rows=[{"user_id": "u123"}]
).to_dict()
```

> **In practice (this repo).** [`phase2/`](phase2/) implements a minimal
> point-in-time feature store over polars `join_asof` (same API shape as the
> Feast snippet above: `get_historical_features` / `get_online_features`). The
> payoff is a *measured* skew number: building features the naive "join today's
> totals" way inflates them **2.0-2.8x** vs point-in-time correct. On top of it,
> an interpretable logistic-regression ranker with a user x item **cross feature**
> (category affinity) finally beats popularity (+2% NDCG) -- and its weights are
> readable (e.g. a *negative* weight on user activity). Full walkthrough:
> [`docs/phase2.md`](docs/phase2.md).

---

## Phase 3: The Serving Architecture

What happens in those 100ms when a user request arrives:

```
User Request (HTTP)
      |
      v
+-----------------------------------------------------+
|  Recommendation Service (100ms SLA)                 |
|                                                     |
|  1. Fetch user features from Redis/Feast (~5ms)     |
|  2. Compute user embedding (~2ms)                   |
|  3. ANN search in Qdrant for 500 candidates (~10ms) |
|  4. Batch fetch item features for 500 (~15ms)       |
|  5. Run ranker on 500 (user, item) pairs (~30ms)    |
|  6. Apply business rules (filter OOS, etc.) (~2ms)  |
|  7. Return top 20                                   |
+-----------------------------------------------------+
      |
      v
   Response + async log: features used, items shown, request_id
```

**For model serving:** [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) or [BentoML](https://www.bentoml.com/). Both handle Python model serving, autoscaling, and model versioning.

> **Critical (Rule 29):** Log the *exact features* passed to the model at inference time. When you generate training data for the next version, join user actions to this log -- not to the current feature store state. This eliminates training-serving skew entirely.

> **In practice (this repo).** [`phase3/`](phase3/) assembles Phases 1-2 into a
> live `RecommendationService` with all six stages timed. Over 2,000 real-user
> requests it holds **p50 6.2ms / p99 14.4ms, 100% within the 100ms budget**
> (flattering, since it's in-process -- but every stage is measured, so real
> infra costs are easy to locate). Fault injection proves graceful fallback
> (Rule 10): a broken ranker still returns 20 items with `fallback_used=True`,
> no 500. Business rules filter 357k out-of-stock items. And the Rule 29 feature
> log captured 40k rows of exactly-what-was-served -- the skew-free seed for v2.
> Full walkthrough: [`docs/phase3.md`](docs/phase3.md).

---

## Phase 4: Monitoring -- Rule 10

> "Watch for silent failures."

Production ML systems do not crash -- they degrade. Items go out of stock and keep getting recommended. A pipeline stalls and the model serves week-old features. A category goes viral and the model's prior is wrong. None of these throw exceptions.

### Layer 1: Data Health

```python
assertions = [
    # Volume check: row count drops >20% triggers alert
    RowCountAssertion(table="click_events", min_ratio=0.8, lookback_days=7),

    # Null rate on critical feature
    NullRateAssertion(column="item_in_stock", max_null_rate=0.01),

    # Distribution drift
    DistributionAssertion(column="user_purchase_count_30d",
                          method="KL_divergence", threshold=0.1)
]
```

**OSS:** [Great Expectations](https://greatexpectations.io/) for data quality assertions, [Evidently AI](https://www.evidentlyai.com/) for ML-specific drift detection.

### Layer 2: Model Health

Track daily:
- Mean predicted P(click) per category (should be stable)
- Actual CTR vs. predicted CTR -- calibration check
- % of requests hitting fallback (model timeout -&gt; heuristic)
- Recommendation diversity: avg number of unique categories in top-20

### Layer 3: Business Metrics (Rule 25)

Track in real-time from the event stream:
- CTR on recommended items (direct signal)
- Add-to-cart rate from recommendations (better signal)
- Revenue attributed to recommendations (best signal, longer lag)

### Monitoring Architecture

```
Kafka events
    |
    +-->  ClickHouse (real-time aggregations)
    |           |
    |           +--> Grafana dashboard (business metrics)
    |
    +--> Evidently AI (drift detection)
              |
              +--> PagerDuty alert if drift > threshold
```

> **In practice (this repo).** [`phase4/`](phase4/) implements the three health
> layers as check functions with a single pass/fail **pipeline gate**. The numeric
> core (PSI, KL, ECE) is pure and unit-tested. On real data the gate goes **FAIL**
> -- the feature-drift check fires at **PSI 0.47** (major shift). That is not a
> bug: it's the *same* popularity drift that produced the 2-2.8x training-serving
> skew in Phase 2, now surfaced as a monitorable, deploy-blocking signal. Model
> health stays green (0% fallback, 18.9 categories of diversity, calibration ECE
> 0.07). Full walkthrough: [`docs/phase4.md`](docs/phase4.md).

---

## Phase 5: Experimentation -- A/B Testing (Rule 16)

**Never compare two models by running them sequentially.** Traffic patterns change (day of week, seasonality, events). You need both models running simultaneously on different user splits.

### Experiment Configuration

```python
experiment = {
    "id": "ranker_xgboost_v1",
    "type": "A/B",
    "allocation": {
        "control":   {"model": "lr_ranker_v3",      "traffic_pct": 50},
        "treatment": {"model": "xgboost_ranker_v1", "traffic_pct": 50}
    },
    "metrics": {
        "primary":    "add_to_cart_rate",     # Rule 13: observable, attributable
        "guardrails": ["page_load_p99_ms", "recommendation_diversity"]
    },
    "min_detectable_effect": 0.02,   # 2% lift is worth shipping
    "required_power": 0.8,
    "significance_level": 0.05
}
```

### Deterministic, Sticky User Assignment

```python
def assign_experiment_variant(user_id: str, experiment_id: str) -> str:
    # Same user always gets same variant across all requests in the experiment
    hash_val = hashlib.md5(f"{user_id}:{experiment_id}".encode()).hexdigest()
    bucket = int(hash_val[:8], 16) % 100   # 0-99
    return "treatment" if bucket < 50 else "control"
```

> **Avoid novelty bias (Rule 35):** New experiences get more clicks initially just because they are different. Run experiments for at least 2 full weeks before concluding.

**OSS:** [GrowthBook](https://www.growthbook.io/) is fully open-source and handles feature flags and experiment analysis in one tool.

> **In practice (this repo).** [`phase5/`](phase5/) implements sticky, salted
> `assign_variant` (SHA-256), a pure-Python two-proportion z-test (via `math.erf`,
> no scipy), and an up-front `required_sample_size` power calculation. The replay
> A/B test (popularity vs the LR ranker, hit@20) is a deliberate teaching result:
> the +2% NDCG offline win from Phase 2 comes back **inconclusive** (p=0.84, CI
> straddles zero) -- and the power analysis explains why, calling for ~14.7k users
> per arm when we had ~1.2k (12x underpowered). The lesson: decide sample size
> first, and never ship on a noisy number. Full walkthrough:
> [`docs/phase5.md`](docs/phase5.md).

---

## Phase 6: Near-Realtime Freshness (Rule 8)

Your 15-minute freshness requirement means a daily-retrained model is not enough. A user who just bought running shoes should not see more running shoes immediately after.

### Approach A: Streaming Feature Updates (covers most freshness needs)

```
User buys item
      |
      v
Kafka topic: "purchase-events"
      |
      v
Flink streaming job: update user features in Redis
      | (~30 second lag)
      v
Next recommendation request sees updated features
```

You do not need to retrain the model -- you just need fresh features. For most freshness requirements (15 min), streaming features to Redis is sufficient. This is simpler than online learning and far more reliable.

### Approach B: Online Learning (for <1 min freshness)

Retrain on streaming data as it arrives. Only introduce this complexity if streaming feature updates cannot meet your freshness SLA. Online learning introduces new failure modes: catastrophic forgetting, feature distribution instability, and time-dimension training-serving skew.

---

## The Complete Architecture

```
+------------------------------------------------------------------+
|                        DATA LAYER                                |
|                                                                  |
|  User actions --> Kafka --> Flink (stream) --> Redis (online)    |
|                         --> Spark (batch)  --> Iceberg (offline) |
+-----------------------------+------------------------------------+
                              |
+-----------------------------v------------------------------------+
|                    FEATURE STORE (Feast)                         |
|  Point-in-time correct for training                              |
|  <5ms lookup for serving                                         |
+----------+------------------+------------------+----------------+
           |                  |                  |
+----------v--------+ +-------v---------+ +------v-----------+
|  TRAINING PIPELINE| |  MODEL REGISTRY | |  EXPERIMENT SVC  |
|  Airflow DAG      | |  MLflow         | |  GrowthBook      |
|  Daily retrain    | |  versioning +   | |  A/B assignment  |
+----------+--------+ |  lineage        | |  metrics         |
           |          +-------+---------+ +------+-----------+
           |                  |                  |
           +------------------v------------------+
                              |
+-----------------------------v------------------------------------+
|                      SERVING LAYER                               |
|                                                                  |
|  +------------------------+   +-----------------------------+   |
|  |  Candidate Generation  |   |  Ranker                     |   |
|  |  Qdrant ANN search     |-->|  Ray Serve                  |   |
|  |  (item embeddings)     |   |  LR --> XGBoost --> DNN     |   |
|  +------------------------+   +-----------------------------+   |
+-----------------------------+------------------------------------+
                              |
+-----------------------------v------------------------------------+
|                       MONITORING                                 |
|  Great Expectations (data quality)                               |
|  Evidently AI (drift detection)                                  |
|  Grafana + ClickHouse (business metrics, real-time)             |
+------------------------------------------------------------------+
```

---

## The Prototype-to-Production Gap

These are the things you never had to care about in a notebook:

| Prototype Reality | Production Reality | Key Rule |
|---|---|---|
| Features computed once, at training time | Features must be identical at training AND serving time | Rule 29, 32 |
| Data is static (a CSV) | Data pipelines fail silently; you need assertions | Rule 10 |
| Model eval on a held-out set | Must eval temporally (train on Jan, eval on Feb) | Rule 33 |
| One model at a time | Multiple model versions in flight simultaneously | Rule 16 |
| No position bias (you showed it once) | Position in the list affects clicks -- model separately | Rule 36 |
| Accuracy = success | Accuracy does not equal business metric; A/B test | Rule 25 |
| Complex model = better | Simple model you can debug beats a complex one you cannot | Rule 4, 14 |

---

## OSS Tool Stack Summary

| Layer | Tool | Why |
|---|---|---|
| Event streaming | Apache Kafka | De-facto standard; Flink integrates natively |
| Batch processing | Apache Spark | Handles 500M events/day comfortably |
| Event / feature storage | Apache Iceberg | Time-travel queries for point-in-time correct training |
| Feature store | Feast | Open-source; enforces PIT correctness; single feature definition |
| Online feature serving | Redis | Sub-5ms p99 for feature lookups |
| Stream processing | Apache Flink | Stateful streaming; 30s lag for feature updates |
| Model training | PyTorch | Dominant for neural models; sklearn for LR baseline |
| Experiment tracking | MLflow | Tracks runs, artifacts, model versions |
| Training orchestration | Apache Airflow | Schedules and monitors daily retrain DAGs |
| Candidate generation | Qdrant / Milvus | ANN search on 50M item embeddings |
| Model serving | Ray Serve | Python-native; autoscaling; shadow deployment |
| Data quality | Great Expectations | Assertions on every pipeline run |
| Drift detection | Evidently AI | ML-specific: feature drift, prediction drift |
| Metrics / dashboards | ClickHouse + Grafana | Real-time business metrics from event stream |
| Experimentation | GrowthBook | Feature flags + experiment analysis, fully OSS |

---

## Recommended Build Order

| Timeline | What to Build | Why |
|---|---|---|
| Week 1-2 | Heuristic ranker + event logging (Kafka + Iceberg) | Ship something; start collecting data |
| Week 3-4 | Two-tower candidate generation (PyTorch + Qdrant) | A/B vs. heuristic |
| Week 5-6 | Logistic regression ranker (sklearn + MLflow) | A/B vs. embedding-only |
| Month 2 | Feature store (Feast) + streaming features (Flink + Redis) | Eliminate training-serving skew |
| Month 3 | Data monitoring (Great Expectations + Evidently) + A/B framework (GrowthBook) | Detect silent failures; safe iteration |
| Month 4+ | Upgrade ranker to XGBoost, then DNN if XGBoost plateaus | Only if you have >100M training examples |

The model upgrades in Month 4+ are optional and depend on observed ceiling. The infrastructure in Months 1-3 is not optional -- every production ML system needs it.

---

## Key Google Rules of ML Reference

| Phase | Most Relevant Rules |
|---|---|
| Before ML | Rule 1 (launch without ML), Rule 2 (design metrics first), Rule 7 (heuristics as features) |
| First pipeline | Rule 4 (keep first model simple), Rule 5 (test infra independently), Rule 10 (watch for silent failures), Rule 14 (interpretable models) |
| Feature engineering | Rules 17-22 (feature design), Rule 29 (log features at serving time), Rule 32 (share code between train and serve), Rule 33 (temporal holdout) |
| Serving skew | Rules 29-37 (training-serving skew -- read all of these) |
| Experimentation | Rule 13 (observable metrics), Rule 16 (plan to iterate), Rule 25 (business metrics > accuracy), Rule 35 (novelty bias) |
| Slowdown | Rule 38 (misaligned objectives), Rule 41 (qualitatively new data) |

---

*Guide based on Google's Rules of Machine Learning (developers.google.com/machine-learning/guides/rules-of-ml). OSS tool choices current as of mid-2026.*
