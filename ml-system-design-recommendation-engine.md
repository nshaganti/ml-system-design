# ML System Design: Real-Time Recommendation Engine

A practical end-to-end guide for ML engineers transitioning from prototype notebooks to production systems. Uses a real-time product recommendation system as the vehicle -- the same class of problem operated at scale by large e-commerce platforms.

> **About the scenario vs. the reference implementation.** The architecture below
> is framed around a hypothetical large-scale e-commerce system (10M DAU, 50M
> items) because it's a familiar way to reason about scale and latency. The
> **reference implementation in this repo runs on real data -- the KuaiRand-Pure
> short-video dataset** -- and each phase's "In practice (this repo)" box reports
> the *actual measured* result. The repo is in two parts: **Part I (Phases 0-7)**
> builds the classic recommender; **Part II (Phase 8)** confronts the fact that
> Part I's offline metrics are causally biased and fixes it with off-policy
> evaluation. See [`docs/datasets.md`](docs/datasets.md).

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

> **In practice (this repo).** We implemented Phase 0 on the KuaiRand-Pure
> dataset in [`phase0/`](phase0/). The heuristic scores **Recall@20 = 0.0695**
> with an honest temporal split -- the baseline every later model must beat. A
> twist worth noting: cold-start recall (0.118, pure popularity) actually *beats*
> warm-user recall (0.066, popularity + category filter), an early hint that the
> category signal is weak here (Phase 2 confirms it). Catalog coverage (7.1%)
> matters as much as recall: a popularity ranker is a "popularity trap" that never
> surfaces the long tail. Full walkthrough: [`docs/phase0.md`](docs/phase0.md).

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
> [`phase1/`](phase1/) with MLflow tracking. On KuaiRand it **beats** the Phase 0
> heuristic: **Recall@20 0.1099 vs 0.0695 (+58%)** and catalog coverage +117% --
> because the feedback is dense (a third of events are strong) and the catalog is
> small (~7.5k). That's the *opposite* of a sparse e-commerce log, where the same
> ID-only two-tower would likely lose to a strong heuristic; model value is a
> function of the data regime. Popularity-weighted negatives and train/serve
> dot-product consistency are doing quiet work under the hood. The lasting
> deliverable is still the *pipeline*, not the model. Full story:
> [`docs/phase1.md`](docs/phase1.md).

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
> totals" way inflates them **2.0-2.4x** vs point-in-time correct. The ranker,
> though, is an honest **negative result**: the user x item category cross feature
> gets a large positive *in-sample* weight yet **loses** to popularity out of
> sample (-12% NDCG), because KuaiRand's category tag is coarse and engagement is
> popularity-driven. The interpretable weights still pay off (a *negative* -0.61
> weight on user activity; a diagnosis of why the feature failed). Lesson: judge
> features by out-of-sample lift, not by whether the model likes them. Full
> walkthrough: [`docs/phase2.md`](docs/phase2.md).

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
> requests it holds **p50 4.4ms / p99 ~9ms, 100% within the 100ms budget**
> (flattering, since it's in-process -- but every stage is measured, so real
> infra costs are easy to locate; `feature_fetch` dominates at ~9.7ms). Fault
> injection proves graceful fallback (Rule 10): a broken ranker still returns 20
> items with `fallback_used=True`, no 500. The business-rule step is a
> domain-neutral **policy layer** (0 items filtered on KuaiRand, which has no
> eligibility signal -- but the seam is there). And the Rule 29 feature log
> captured 40k rows of exactly-what-was-served -- the skew-free seed for v2.
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
> -- but notably *not* on drift: feature drift is **stable (PSI 0.024)**, so that
> alarm correctly stays quiet. Instead the **row-count** check fires (serving
> window has ~half the reference rows) and **calibration** fails (ECE 0.115). The
> lesson: run every layer -- a monitor with a favorite failure mode is half-blind.
> Fallback stays at 0% and diversity is healthy (4.0 categories in the top-20 --
> KuaiRand tags are coarse). Full walkthrough: [`docs/phase4.md`](docs/phase4.md).

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
> A/B test (popularity vs the LR ranker, hit@20) is decisive: the LR ranker is
> **significantly worse -- -10%, p=0.0006**, 95% CI entirely below zero, on ~7,500
> users per arm (well powered). The online test *confirms* Phase 2's offline
> signal and gives the confidence to **not ship** the regression. The lesson: a
> significant negative is the tool working. Full walkthrough:
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

> **In practice (this repo).** [`phase6/`](phase6/) implements Approach A: a
> `StreamingFeatureStore` that wraps the frozen batch store and layers O(1) delta
> updates on top, exposing the *same* read interface so the Phase 3 service
> consumes it unchanged. Streaming a user's earliest in-session event and
> measuring hit@20 on their later items gives an honest **null result: +0.2%,
> p=0.91 (not significant)** with the model unchanged. On KuaiRand's short-video
> engagement, a 30-second delta layer just isn't the lever it would be on a bursty
> e-commerce cart. The mechanism is real and correct; whether freshness *pays* is
> a property of the workload -- build the lever, then measure. Full walkthrough:
> [`docs/phase6.md`](docs/phase6.md).

---

## Phase 7: Session Co-visitation (Community Benchmark)

The community's classic protocol for interaction logs is **session-based
next-item** prediction, evaluated leave-one-out. Before assuming you need a
sequence model, measure the simplest strong baseline: **co-visitation** (count
which items co-occur within a session, recommend the neighbors of the last item).

> **In practice (this repo).** [`phase7/`](phase7/) sessionizes the event stream
> and builds a pure-Python co-visitation recommender behind the same
> `.recommend()` interface. On the leave-one-out session task it beats popularity
> by **+61%** (Recall@20 0.0797 vs 0.0496). A real but *modest* win -- on
> KuaiRand's small catalog popularity is already a strong session baseline, so
> there's less headroom than on a sparse e-commerce log. Win size is a property of
> the data, which is why you benchmark on *your* data. Co-visitation slots into the
> Phase 3 candidate union for free. Full walkthrough: [`docs/phase7.md`](docs/phase7.md).

---

## Part I (continued): Sequence Model & Two-Stage Ranking (Phases 10, 12-13)

Three phases that finish the Part I recommender -- and each one is an honest lesson
in *earning* complexity.

- **Phase 10 -- GRU4Rec sequence model.** Before assuming attention/recurrence beats
  a count-based baseline, measure it on the *same* leave-one-out protocol as Phase 7.
  Honest result: GRU4Rec **underperformed co-visitation (-15%)** on KuaiRand's small
  catalog -- a neural sequence model is not automatically better than a brutally
  strong count-based one. Full walkthrough: [`docs/phase10.md`](docs/phase10.md).
- **Phase 12 -- Two-stage integration.** Compose the Phase 1 two-tower retriever
  (stage 1) with the Phase 2 LR ranker (stage 2) behind one `.recommend()`. Honest
  result: the naive two-stage pipeline **lost to two-tower alone (-18.7%)** because
  popularity-flavored LR features undid the retriever's personalization. Two stages
  aren't automatically better than one. Full walkthrough: [`docs/phase12.md`](docs/phase12.md).
- **Phase 13 -- Two-tower score as a ranking feature.** Feed the retriever's own
  score into stage 2 as a feature. Now the two-stage pipeline **beats two-tower
  alone (+3.3%)**: stage 2 earns its place only once it can *see what stage 1 knows*.
  Full walkthrough: [`docs/phase13.md`](docs/phase13.md).

---

## Part II -- Causality, Exploration & the Closed Loop (Phases 8-20)

Part I graded and trained models by *replaying logged data*. Part II is the reckoning:
those logs were written by the incumbent policy, so both the *evaluation* and the
*learning* built on them are biased -- and the fix is to treat data collection as a
causal, closed-loop problem, not a static dataset.

### Phase 8 -- Off-Policy Evaluation (Rules 23, 30, 36)

Everything in Part I graded models by *replaying the logged data*. But those logs
were written by the incumbent policy -- it only ever showed items it liked, to
users it liked. So an offline metric estimated from them is **biased**, not just
noisy. The fix requires knowing the probability each item was shown (the
propensity), which is why KuaiRand's **uniform-random exposure log** (`beta=1/N`)
is the key that unlocks honest evaluation.

> **In practice (this repo).** [`phase8/`](phase8/) implements IPS, SNIPS, the
> Direct Method, and Doubly Robust estimation, plus effective-sample-size. Grading
> a target policy on the **biased** log overstates its true value by **+100%**
> (0.523 vs a ground truth of 0.261 computed from the random log). Reweighting the
> random log by known propensities recovers the truth: **SNIPS lands at 0.6%
> error**, doubly robust at 6%. This is the capstone lesson -- even after all of
> Part I's discipline, the offline number underneath it can be a factor of two
> wrong. Full walkthroughs: [`docs/phase8.md`](docs/phase8.md) and
> [`docs/off-policy-evaluation.md`](docs/off-policy-evaluation.md).

### Phase 9 -- Off-Policy LEARNING

The bias isn't only in *measurement*: Part I's policy was also *trained* on
confounded logs, so it's genuinely suboptimal, not just mis-measured. Learn the
policy from the unbiased random log instead -- and **a little unbiased data beats a
lot of biased data** (a small exploration log trains a better policy than the huge
production log). Full walkthrough: [`docs/phase9.md`](docs/phase9.md).

### Phase 11 -- Position-Bias Debiasing (controlled simulation)

Naive CTR ranks *positions* as much as it ranks items -- click = relevance x
examination. Inverse-propensity weighting on an examination curve (learned from a
result-randomization bucket) recovers the true item ranking (**Spearman 0.86 ->
0.97**). The core Part II tradeoff shows up here: IPW is unbiased but higher
variance. Full walkthrough: [`docs/phase11.md`](docs/phase11.md).

### Phase 14 -- The Explore-and-Learn Loop

Where does the unbiased data Phases 8/9/11 assumed actually come from in a live
system? **Exploration.** A greedy (exploit-only) policy manufactures the Phase 8
selection bias in real time; **Thompson sampling gets ~48% less regret than greedy
and mints a full-support log** that off-policy methods can actually use. Full
walkthrough: [`docs/phase14.md`](docs/phase14.md).

### Phase 15 -- The Contextual Bandit (LinUCB)

The best item depends on *who* is asking. LinUCB conditions its choice and its
exploration on a user-context vector: on a world where the best arm flips per user
it gets **~92% less regret than context-free Thompson**. Context helps, and
exploration still helps on top of context. Full walkthrough: [`docs/phase15.md`](docs/phase15.md).

### Phase 16 -- Closing the Loop (capstone)

Wires the whole course into the cycle a recommender runs forever: **deploy -> explore
+ log -> learn off-policy -> redeploy**. Exploration lifts the deployed policy's TRUE
value from **41% to ~96% of the skyline**; the no-exploration variant stalls. Honest
wrinkle: with a *near*-well-specified linear model (the reward is clipped-linear, so
mildly non-linear), IPS was a near-wash (it only added variance) -- propensities earn
their keep under stronger misspecification or direct value estimation, not everywhere. Full walkthrough: [`docs/phase16.md`](docs/phase16.md).

### Phase 17 -- Real Context + Safety-Gated Redeploys

Removes Phase 16's two shortcuts: the loop now runs on **real per-user contexts**
(activity + signal-mix features for ~23.5k KuaiRand users), and **every redeploy is
gated by off-policy evaluation** on a fresh uniform-random bucket (unbiased for any
candidate). The gate helps even in the clean case (**67% vs 59%** of skyline) and,
when a simulated logging bug ships a corrupt candidate, blocks it -- the ungated
loop's worst deploy craters to 0.57 while the gated loop holds 0.61. A safety gate is
priced in the good case and cashed in the bad. Details:
[`docs/phase17.md`](docs/phase17.md).

### Phase 18 -- SASRec: an Honest Negative

Closes Phase 10's loose end: does **self-attention** beat the co-visitation baseline
that beat GRU4Rec? On the identical leave-one-out protocol, **no** -- SASRec finishes
last, below popularity (Recall@20 0.023 vs co-visitation's 0.080), despite a verified
sound model and a generous budget. A data-hungry transformer starves on a small
(~7.5k-item), dense catalog where co-occurrence is a brutally strong baseline. The
fourth honest negative in the project (with Phases 2, 6, 10): **complexity must earn
its place on YOUR data.** Details: [`docs/phase18.md`](docs/phase18.md).

### Phase 19 -- Contextual OPE/OPL

Generalizes Phases 8-9 from one global policy to **per-user** policies. The context-
free estimator is **24% off** for a contextual target (a *constructed* strawman -- the
Phase 8 estimator fed context-marginalized frequencies -- to isolate the principle,
not the mistake teams knowingly make); contextual IPS/SNIPS/Doubly-Robust recover the truth to
within 0.3%, with **DR the safest** (unbiased if either the model or the propensities
are right). Learning off a context-blind log, a contextual policy beats a context-free
one by **+28%** true value -- it recovers the personalization the logger discarded.
Details: [`docs/phase19.md`](docs/phase19.md).

### Phase 20 -- Joint EM Position-Bias Debiasing

Finishes the position-bias arc. Phase 11 needed the examination curve from a costly
randomization bucket; **tabular EM** (the EM form of Regression-EM, with a per-item
relevance table rather than a feature regressor) estimates the examination curve AND
per-item relevance *jointly* from ordinary confounded production logs. It matches the
randomization-based IPW (**Spearman 0.923 vs 0.937**) and nearly the oracle (0.941)
with **no randomization at all** -- you can debias production traffic in place, and the
recovered relevances are the *ranking* you'd feed the Phase 2 ranker (identified only
up to a global scale, so a well-ordered score, not a calibrated CTR). Details:
[`docs/phase20.md`](docs/phase20.md).

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
+-----------------------------+------------------------------------+
                              |
+-----------------------------v------------------------------------+
|            EXPLORE / OFF-POLICY LOOP  (Part II, Phases 14-17)     |
|                                                                  |
|  Serving policy EXPLORES (epsilon-soft / Thompson / LinUCB) and   |
|  logs (context, action, propensity, reward)  -- unbiased data.    |
|  Off-policy LEARNING (IPS/DR, Phases 8-9) trains a better policy  |
|  --> gated by monitoring + min-propensity floor --> REDEPLOY -----+--+
+------------------------------------------------------------------+  |
     ^  (redeploy closes the loop back into the training pipeline)    |
     +----------------------------------------------------------------+
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
| Part II | Off-policy eval/learning (Phase 8-9, contextual in 19), position debiasing (Phase 11, joint-EM in 20), then an explore-and-learn loop (Phases 14-17) | Fix biased metrics/learning; make data collection a closed, safety-gated causal loop |

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
