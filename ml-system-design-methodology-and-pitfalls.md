# ML System Design: Methodology, Decisions & Real-World Pitfalls

> **Audience:** ML engineers moving from prototype notebooks to production systems.
> **Vehicle:** A recommender built on the real [KuaiRand-Pure dataset](https://kuairand.com) (Kuaishou short-video interaction logs).
> **What this document is:** A defense of every decision made -- not just *what* we built, but *why*, and what breaks when you get it wrong.

---

## Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [Phase 0: Heuristic Baseline](#2-phase-0-heuristic-baseline)
3. [Phase 1: Two-Tower Model](#3-phase-1-two-tower-model)
4. [Real-World Pitfalls](#4-real-world-pitfalls)
5. [Results & Honest Interpretation](#5-results--honest-interpretation)
6. [Google's Rules of ML Applied](#6-googles-rules-of-ml-applied)

---

## 1. Problem Framing

### Why Product Recommendations?

Product recommendations were chosen as the learning vehicle because they compress every hard production ML problem into one system:

| Challenge | How recommendations expose it |
|---|---|
| Scale | 50M items -- can't score all of them per request |
| Latency | 100ms SLA -- model complexity is physically constrained |
| Cold-start | 30% of sessions are new/anonymous users |
| Training-serving skew | Item availability changes; user history grows |
| Implicit feedback | No explicit ratings -- only clicks, carts, purchases |
| Position bias | Items shown at position 1 always get more clicks |
| Freshness | A user who just bought running shoes shouldn't see more |

Any system that solves recommendations correctly has solved the core production ML problems. The patterns transfer directly to search ranking, ad CTR prediction, fraud detection, and content feeds.

---

### Why the Two-Stage Architecture?

The latency math makes the answer unavoidable.

**The constraint:** 100ms end-to-end recommendation response.

**The naive approach:** Score every item for every user with a rich model.

```
50M items x 1ms per scoring = 50,000 seconds per request
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                this is not a typo
```

Even at 0.001ms per item (GPU-optimized neural net):

```
50M items x 0.001ms = 50 seconds
```

Still 500x over budget. **No amount of hardware fixes this linearly.**

**The solution:** Split into two stages with radically different compute profiles.

```
Stage 1 -- Candidate Generation (~10ms)
  Goal:  50M items --> 500 candidates
  Model: Fast retrieval (ANN search on pre-computed embeddings)
  Cost:  One embedding lookup + dot products against pre-indexed matrix

Stage 2 -- Ranking (~50ms)
  Goal:  500 candidates --> 20 ranked results
  Model: Rich pointwise ranker with user x item features
  Cost:  500 forward passes (not 50M)
```

**Why this split is the right abstraction:**
- Stage 1 optimizes for **recall** -- "get the right items in the candidate set"
- Stage 2 optimizes for **precision** -- "rank the candidates correctly"
- They evolve independently: you can improve the ranker without retraining the retrieval model

This architecture is used by YouTube, Pinterest, Airbnb, Walmart, and virtually every large-scale recommendation system. It is not an optimization -- it is a physical requirement.

---

### Why Recall@20 as the Primary Offline Metric?

**What Recall@20 measures:** Of all items a user actually purchased in the test period, what fraction appear in the top-20 recommendations?

```
Recall@20 = |purchased_items &#8745; top_20_recommendations| / |purchased_items|
```

**Why this metric and not accuracy, NDCG, or MAP?**

| Metric | Problem |
|---|---|
| Accuracy | Useless for implicit feedback (most items are "0" -- not rated, not bought) |
| MAP | Requires knowing the full ranked list is exhaustive -- we don't know what users *would have* bought |
| NDCG | Requires graded relevance labels (we only have binary buy/no-buy) |
| Recall@20 | Directly answers: "Did we surface the item before the user needed it?" |

**Why Recall@20 is not enough alone:**

> &#9888;&#65039; **Real-world problem:** Recall@20 can be gamed. A model that recommends the 20 globally most popular items every time achieves decent recall on many datasets (popular items get engaged with). On a *sparse* e-commerce log this popularity baseline is famously hard to beat -- but on KuaiRand, where feedback is dense, our Phase 1 two-tower *beats* the popularity heuristic decisively (see Section 5). The point stands: always report coverage and business metrics alongside recall, because recall alone can flatter a bestseller list.

The metrics you also need in production:
- **Coverage** -- what % of catalog appears in any recommendation? (Low = popularity trap)
- **Novelty** -- are we recommending items the user didn't already know about?
- **Business metric** -- add-to-cart rate, revenue. Offline recall is a proxy. The true test is an A/B experiment.

**Rule 25: Utilitarian performance beats predictive accuracy.** Never ship a model based on offline metrics alone. Offline recall is the ticket to the A/B experiment -- not the final answer.

---

### Why Temporal Split Over Random Split?

**The random split bug:**

```python
# WRONG -- this is what most tutorials do
from sklearn.model_selection import train_test_split
train, test = train_test_split(events, test_size=0.2, random_state=42)

# What this creates:
# train = events from Jan 1, Jan 3, Jan 5, Jan 7, Jan 8...
# test  = events from Jan 2, Jan 4, Jan 6, Jan 9, Jan 10...

# The model sees Jan 3 during training.
# The test event from Jan 2 is BEFORE Jan 3.
# The model has "seen the future" of the test user.
```

**Why it's insidious:** The model learns that "user A buys item X" from Jan 3 training data. Then it's evaluated on Jan 2 test data for the same user. It appears to predict the past. Your offline metrics look great. Then you deploy and performance is 40% lower than expected.

**The correct approach:**

```python
# RIGHT -- split by time
def temporal_split(events, train_fraction=0.8):
    timestamps = events["timestamp_ms"].sort()
    cutoff_idx = int(len(timestamps) * train_fraction)
    cutoff_ms  = int(timestamps[cutoff_idx])
    
    train = events.filter(pl.col("timestamp_ms") < cutoff_ms)
    test  = events.filter(pl.col("timestamp_ms") >= cutoff_ms)
    return train, test, cutoff_ms
```

**Rule 33: Test on data *after* your training cutoff.** Temporal holdout better simulates production conditions than random splits. Always.

On the KuaiRand-Pure timeline we split at 80% -- training on the earlier portion, evaluating on the later. The model never sees any information from the test period during training.

---

## 2. Phase 0: Heuristic Baseline

### The "Launch Without ML" Principle Is Not Laziness

> **Rule 1: Don't be afraid to launch a product without machine learning.**

The heuristic baseline serves two purposes that are *prerequisites* for every ML phase that follows:

**1. It creates training data.**

You cannot train a personalized recommendation model without behavioral data -- clicks, carts, purchases, and crucially, *what was shown* (impressions). If you launch directly with an ML model, you have no data to train it on. The heuristic is what generates that data.

**2. It establishes a measurable baseline.**

Without Phase 0, you have no answer to "did Phase 1 actually improve anything?" You would be deploying models with no reference point. Every ML team that skipped this step has experienced the same outcome: months of model iterations with no clear signal of progress.

> &#9888;&#65039; **Real-world problem:** Teams that skip Phase 0 often end up running their first A/B experiment comparing "ML model" vs "nothing." This is a meaningless comparison -- "nothing" usually means a random or reverse-chronological feed. The model wins easily, the team celebrates, and then they spend the next year unable to improve further because they have no principled baseline.

---

### Why Those Specific Weights? (strong=3, medium=2, weak=1)

```python
SIGNAL_WEIGHTS = {
    "strong": 3.0,   # target actions: long-view / like / follow / forward / comment
    "medium": 2.0,   # engagement: click
    "weak":   1.0,   # exposure: impression without engagement
}
```

These are not arbitrary. They encode an intent-signal hierarchy:

| Signal | Intent signal | Reasoning |
|---|---|---|
| Strong | Highest -- active endorsement | User long-viewed, liked, followed, forwarded, or commented |
| Medium | Medium -- deliberate action | User clicked in |
| Weak | Low -- passive signal | Item was merely exposed; could be scroll-past |

The ratio (3:2:1) is a reasonable starting point. **Rule 7: Convert heuristics into features.** These weights encode domain knowledge that the ML model will later learn automatically from data. By making them explicit and tunable now, you create a feature that can be validated, adjusted, and eventually replaced by a learned signal.

**What not to do:** Equal weights (1:1:1) treats a strong action the same as a scroll-past. This produces a popularity score dominated by raw exposure counts -- which is exactly the noise the ML model will have to fight.

---

### The Event Schema: Why `recommendation_request_id`?

```json
{
  "event_id": "uuid",
  "recommendation_request_id": "uuid",
  "user_id": "hashed_id",
  "event_type": "impression | click | add_to_cart | purchase",
  "item_id": "string",
  "position": 3,
  "timestamp_ms": 1720000000000
}
```

**The `recommendation_request_id` field** ties every event back to the exact recommendation request that generated it. Without it:

- You know user A bought item X
- You do NOT know whether item X was recommended or organically found
- Your training labels are polluted with organic purchases (not causally connected to your model)
- Your A/B test metrics attribute all purchases to the model, including ones that would have happened anyway

**Rule 2: Design metrics first.** Instrument this from day one. Retrofitting it onto a production system after 6 months of data collection means throwing away 6 months of clean training signal.

---

### Why `position` in the Event Schema?

> &#9888;&#65039; **Real-world problem -- Position Bias (Rule 36)**

Items shown at position 1 are clicked more than items at position 5, not because they are better, but because users look at the top first. This is position bias.

If you train a model on click data without accounting for position, the model learns: "items with high click rates are good." But high click rate is partly caused by the model's own placement decision -- a feedback loop that amplifies whatever the model already believes.

```
Model likes item X
  --> Shows item X at position 1
    --> Item X gets more clicks (position 1 bias)
      --> Model thinks item X is even better
        --> Shows item X at position 1 more
          --> ...
```

**The `position` field in the event schema is the first step toward fixing this.** Later, you use it to:
1. Train a position-bias correction model (predict P(click | position) irrespective of item quality)
2. Debias click labels: `debiased_score = observed_click / P(click | position)`
3. In Phase 2's feature engineering: add position as a feature, but **zero it out at serving time** (you don't know position before showing results)

---

## 3. Phase 1: Two-Tower Model

### Why Mean-Pool Item Embeddings for User Representation?

The naive approach for user representation in a two-tower model is a dedicated user embedding table:

```python
# NAIVE APPROACH
self.user_emb = nn.Embedding(num_users, embedding_dim)
self.item_emb = nn.Embedding(num_items, embedding_dim)

# user_vec = self.user_emb(user_id)  # simple lookup
```

**Why this breaks in production:**

**Problem 1 -- Cold-start.** A large fraction of real sessions are from users with no prior history. The user embedding table has no entry for them. What does `self.user_emb(new_user_id)` return? Random noise (or a crash, if the ID is out of range). You silently serve garbage recommendations to those users.

**Problem 2 -- Training-serving skew (Rule 32).** The user embedding is learned during training. At serving time, you look up the same embedding. But if a user's preferences have changed since training (they just binged cooking videos; now they want travel), the stale embedding doesn't reflect this. The user vector is only updated on the next model retrain.

**Problem 3 -- Memory.** 10M daily active users x 128-dim float32 = 5GB embedding table. This grows with user count, not item count.

**The solution: Mean-pool of item embeddings**

```python
def get_user_vector(self, history: torch.Tensor) -> torch.Tensor:
    # history: (batch, seq_len) -- padded item indices
    emb    = self.item_emb(history)                              # (batch, seq_len, dim)
    mask   = (history != self.pad_idx).unsqueeze(-1).float()    # (batch, seq_len, 1)
    n_valid = mask.sum(dim=1).clamp(min=1.0)                    # (batch, 1)
    return (emb * mask).sum(dim=1) / n_valid                    # (batch, dim)
```

**Why this is better:**

| Property | User ID table | Mean-pool |
|---|---|---|
| Cold-start | Breaks (no entry) | Graceful (empty history -> fallback) |
| Freshness | Stale until retrain | Updates with each interaction (new item in history) |
| Memory | O(n_users x dim) | O(n_items x dim) -- item table only |
| Training = Serving (Rule 32) | Same lookup | Same mean-pool computation |

**This is precisely what YouTube DNN (2016) does.** The user is represented as the mean of their watch history video embeddings -- not a user ID. This was one of the key insights that made it work at scale.

---

### Why Strong Signals as Positives ONLY?

**The signal breakdown on KuaiRand-Pure (standard log, 1,436,609 events):**

| Signal | Count | % of total |
|---|---|---|
| Weak (exposure) | 770,085 | 53.6% |
| Medium (click) | 180,053 | 12.5% |
| Strong (target actions) | 486,471 | 33.9% |

**What happens if you use all events as positives?**

With weak exposures as positives, the majority of your training signal says "every item shown is a positive" -- and the model just relearns popularity. The strong-intent signal (long-view, like, follow, forward, comment) gets diluted. Even though KuaiRand is *dense* compared to e-commerce (a third of events are strong), the principle holds: train on the signal that reflects intent, not the signal that reflects exposure.

**The design: positives = strong events only**

```python
STRONG_SIGNAL = "strong"   # long_view / is_like / is_follow / is_forward / is_comment

# Vocab: only items that appear in strong events, with >= 3 strong interactions each
strong_events = train_events.filter(pl.col("signal") == STRONG_SIGNAL)
# Result on KuaiRand: 6,266 items (from 7,540 unique training items)
```

**Why this works so well here:**
- Each vocab item has *abundant* training signal: 535,785 BPR triples across 6,266 items is **~85 examples per item** -- far above the 3-5 minimum. This is the opposite of a sparse e-commerce log, and it's precisely why the two-tower learns good embeddings and beats the heuristic (Section 5).
- Loss converges smoothly (0.40 -> 0.29 over 20 epochs).
- Coverage lands at 15.7% -- the model surfaces diverse items rather than tunneling on the head.

**Rule 17: Prefer directly observed features over learned ones.** Strong behavioral signals (long-view, like, follow) are direct observations of intent. Exposures are indirect and noisy. Always prioritize direct signal -- and when you have a lot of it, the payoff is a model that genuinely learns.

---

### Why BPR Loss Over Cross-Entropy?

**Cross-entropy loss** frames recommendation as classification: given (user, item), predict P(interaction=1). This requires:
- Hard negatives (items the user explicitly didn't like)
- Calibrated probability outputs
- Large amounts of confirmed negative examples

**The problem:** In implicit feedback datasets (no explicit ratings), you don't have confirmed negatives. An item a user didn't interact with might be:
- Irrelevant (true negative)
- Relevant but not seen (false negative -- it was never shown)
- Relevant but seen in a different session (false negative -- attribution loss)

Treating unseen items as negatives with cross-entropy penalizes the model for recommending items the user might actually want.

**BPR loss** sidesteps this by optimizing pairwise ranking:

```
BPR loss = -mean(log(sigmoid(score(user, positive) - score(user, negative))))
```

This says: "the positive item should be ranked higher than the negative item." It does NOT say "the positive probability should be 1.0 and the negative should be 0.0." It only requires the ranking order to be correct.

**Why this matters with random negatives:**
- Random negative = item the user hasn't interacted with
- BPR says: "rank the item they bought above a random item"
- This is always a valid training signal regardless of whether the random item is a true negative

**The intuition:** If a user long-viewed a cooking video, we want `score(user, cooking_video) > score(user, random_item)`. We don't need to claim the random item is bad -- just that the engaged item is better.

---

### Why Raw Dot Product (Not L2-Normalized Cosine) in the Index?

```python
# index.py defaults to raw dot product -- matching how the model TRAINED
scores = item_matrix @ user_vec        # NOT cosine similarity
```

This is a train/serve *consistency* decision (Rule 32), and it's subtle enough to trip up almost everyone.

**The model trains with a raw dot product.** During BPR training, an item embedding's *magnitude* naturally grows for popular items -- and that magnitude is genuine signal the model learned to use. If the serving index then L2-normalizes everything to cosine similarity, it **throws that magnitude away**, so the model optimizes one objective and you serve a different one. Silent training-serving skew.

```
Unnormalized dot product = cosine similarity x magnitude product.
```

A tempting argument says "normalize, so only direction (preference alignment) matters and long-tail items get a fair shot." That can be a valid *modeling* choice -- but only if you also train with normalized embeddings. Mixing normalized serving with unnormalized training is the bug. We keep both sides on the raw dot product.

**The tradeoff:** if you *want* magnitude out of the scoring (e.g. to reduce popularity bias), normalize in *both* training and serving, or blend a normalized and unnormalized score deliberately. The rule is not "normalize" or "don't" -- it's "do the identical thing at train and serve."

---

## 4. Real-World Pitfalls

### Pitfall 1: Training-Serving Skew

**What goes wrong:**

```python
# NAIVE TRAINING JOB (very common in production codebases)
item_stats = spark.read.table("item_stats_current")  # today's stats
click_events = spark.read.table("events_last_30d")   # 30 days of history

training_data = click_events.join(item_stats, on="item_id")
# item_stats reflects TODAY, but events are from 30 days ago.
# The model learns: "when click_rate=0.15, user clicks"
# But at serving time, that same item might have click_rate=0.08.
```

**Why it's insidious:** The model performs well in offline evaluation (you evaluate it on the same skewed training data distribution). It degrades in production. No exceptions are thrown. Metrics drift slowly. Nobody knows why.

> &#9888;&#65039; **This is Rule 37's entire reason for existence.** "Actively measure training/serving skew. Track three gaps: train vs. holdout, holdout vs. next-day, next-day vs. live."

**How we handled it in Phase 1:**
```python
# Point-in-time correct feature lookup (in load_data.py)
def get_item_snapshot(item_properties, as_of_timestamp_ms):
    return (
        item_properties
        .filter(pl.col("timestamp_ms") <= as_of_timestamp_ms)  # only history up to event time
        .sort("timestamp_ms", descending=True)
        .unique(subset=["item_id", "property"], keep="first")  # most recent value before event
        .pivot(...)
    )
```

A well-formed item-property log has timestamped updates -- every category, price, or availability change is a new row. By filtering to `<= event_timestamp`, we see the item as it was at that moment, not as it is today. (KuaiRand ships static video features; the point-in-time discipline matters most for the *rolling* count features built in Phase 2's feature store.)

**Production solution (Phase 2):** A feature store (Feast) enforces this automatically. The `get_historical_features()` call does the point-in-time join for you. The `get_online_features()` call returns current values at serving time. Same feature definition, two modes.

**Rule 32: Share code between training and serving pipelines.** One Python function defines the feature. It runs at training time and at serving time. Two codebases = guaranteed skew.

---

### Pitfall 2: Random vs Temporal Split

**What goes wrong:** Described in Section 1, but here's the exact impact on metrics:

On a dataset with strong temporal patterns (items become popular then fade, seasonal trends), random split artificially inflates metrics by 15-40%. A model evaluated with random split that achieves Recall@20 = 8% might achieve only 3-5% with temporal split.

You ship it, it underperforms in A/B, you investigate for weeks, and the answer is that your evaluation methodology was wrong from day one.

**Our solution:**

```python
def temporal_split(events, train_fraction=0.8):
    timestamps = events["timestamp_ms"].sort()
    n = len(timestamps)
    cutoff_idx = int(n * train_fraction)
    cutoff_ms  = int(timestamps[cutoff_idx])
    
    train = events.filter(pl.col("timestamp_ms") < cutoff_ms)
    test  = events.filter(pl.col("timestamp_ms") >= cutoff_ms)
    return train, test, cutoff_ms
```

We split at the 80th percentile of the timeline, evaluating on the most recent slice. **The model never sees any information from the test period during training.**

**Rule 33: Test on data after your training cutoff.** No exceptions. Always check that your split is temporal before trusting any offline metric.

---

### Pitfall 3: Coverage vs Recall -- Two Axes, Not One

**What happened in our KuaiRand run:**

| Metric | Phase 0 (Heuristic) | Phase 1 (Two-Tower) |
|---|---|---|
| Recall@20 | 6.70% | **12.31%** |
| Catalog coverage | 7.14% | **15.70%** |

Here Phase 1 wins on **both** axes -- recall +84% and coverage +120% -- because KuaiRand's dense feedback and small catalog give the embeddings enough signal to learn (Section 5). So this dataset does *not* show the classic tradeoff. But the reason to always track *both* metrics is exactly that on a **different** dataset you often see them diverge:

> &#9888;&#65039; **On a sparse e-commerce log, a popularity ranker often "wins" Recall@K while a two-tower spreads recommendations across far more of the catalog (higher coverage) yet scores lower recall.** Then Recall@20 alone would tell you to throw the model away -- and you'd be discarding the discovery engine. Coverage is what stops a recommender from collapsing into a bestseller list.

**Why measuring coverage matters regardless of the sign:**

Phase 0 (popularity) concentrates recommendations on the head of the catalog. Phase 1 (two-tower) spreads across more than twice as much of it. On KuaiRand that breadth comes *with* better recall; on sparser data it may cost some recall -- and only by tracking both do you make that call with eyes open.

**What this means for two-stage evaluation:**

The candidate generator's natural metric is Recall@500 -- "did the right item make it into the 500 candidates?" The ranker's metric is Recall@20 -- "did the ranker pick the right 20 from those 500?" Judging Stage 1 purely on Recall@20 sets an unfair bar; we report it here only because on KuaiRand the two-tower clears it anyway.

---

### Pitfall 4: Evaluation Coupling

**The bug:**

```python
# In phase0/evaluate.py (original)
catalog_size = ranker.item_scores["item_id"].n_unique()
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#              This is a HeuristicRanker internal attribute.
#              TwoTowerRanker doesn't have it.
```

When we passed `TwoTowerRanker` to `recall_at_k()`, it crashed with:
```
AttributeError: 'TwoTowerRanker' object has no attribute 'item_scores'
```

**Why it's insidious:** Evaluation code that reaches into model internals becomes a coupling contract. If you add a third model (XGBoost ranker in Phase 2), you either duplicate the attribute name artificially or rewrite the evaluation code -- which means you can no longer compare across runs.

**The fix -- shared interface:**

```python
# Both rankers expose this property
@property
def catalog_size(self) -> int:
    """Number of items this ranker can recommend. Used by evaluate.py for coverage."""
    return ...

# In evaluate.py -- model-agnostic
catalog_size = ranker.catalog_size  # works for any ranker
```

This is the **interface segregation principle** applied to ML evaluation: define what the ranker must expose (a `catalog_size`, a `recommend()` method), not how it's implemented internally.

**The broader lesson:** Treat your evaluation harness as a contract that every model version must satisfy. If you need to rewrite the evaluation code to test a new model, you can no longer compare the new model against old ones. Apples-to-apples comparison requires an unchanging evaluation interface.

---

### Pitfall 5: Silent Data Failures

**What goes wrong:** Your item properties pipeline has a bug. The `categoryid` field starts returning NULL for new items. The ML model silently stops recommending any item added in the last 2 weeks. New product launches get zero exposure. Revenue impact is real, but no exception was raised and no alert fired.

**This is Rule 10 in full force:** "Watch for silent failures. ML systems degrade gradually when data sources go stale -- actively track data statistics."

```python
# PASSIVE MONITORING (not enough)
# "If Grafana shows revenue drop, investigate"
# -- too late. Revenue has already dropped.

# ACTIVE ASSERTIONS (correct approach)
assertions = [
    RowCountAssertion(
        table="click_events",
        min_ratio=0.8,          # alert if today's rows < 80% of 7-day average
        lookback_days=7
    ),
    NullRateAssertion(
        column="categoryid",
        max_null_rate=0.01      # alert if >1% of items missing category
    ),
    DistributionAssertion(
        column="user_purchase_count_30d",
        method="KL_divergence",
        threshold=0.1           # alert if feature distribution drifts significantly
    ),
]
```

Run these assertions at the START of every training job and every batch feature computation job. If any assertion fails, abort the pipeline and page the on-call engineer. **Never ingest bad data silently.**

**The three types of silent failure to instrument:**
1. Volume drop -- pipeline processed fewer rows than expected
2. Null rate increase -- critical feature going missing
3. Distribution shift -- feature values are present but their distribution changed

---

### Pitfall 6: Cold-Start

**The failure:**

```python
# User ID embedding table approach
self.user_emb = nn.Embedding(num_training_users, embedding_dim)

# At serving time for a new user:
user_vec = self.user_emb(new_user_id)
# If new_user_id >= num_training_users -> IndexError (crash)
# If new_user_id is within range (reused ID) -> random noise
# Either way, you're serving garbage to 30% of traffic
```

**Why it's insidious:** The model doesn't know it doesn't know. It returns an embedding confidently and the recommendation service returns a result. No error. 30% of users get random recommendations. You discover it when you break down A/B metrics by user-session-type and notice cold-start users have 70% lower CTR than warm users.

**Our solution -- two-layer fallback:**

```python
def recommend(self, user_id, user_events, n=20):
    # Layer 1: in-vocab history -> ML model
    history_idx = [vocab.encode(item_id) for item_id in user_events["item_id"]]
    history_idx = [i for i in history_idx if i is not None]
    
    if history_idx:
        user_vec = self._compute_user_vector(history_idx)
        return self.index.search(user_vec, k=n)
    
    # Layer 2: no in-vocab history -> Phase 0 heuristic fallback
    return self.fallback.recommend(user_id, user_events, n)
```

**The two-layer fallback pattern is standard production practice:**
- ML model handles warm users (where it has learned signal)
- Heuristic handles cold-start (where no signal exists)
- The boundary is explicit and auditable (you can measure what % of traffic each layer handles)

As users accumulate history, they naturally graduate from the heuristic to the ML model without any code change.

---

### Pitfall 7: Vocab Size vs Training Signal Tradeoff

**The tension:**

```
More vocab items -> more items the model can recommend -> better coverage
Fewer training examples per item -> worse embedding quality -> worse recall
```

**How it plays out on KuaiRand:**

```
Vocab from ALL signals (incl. weak exposures): 7,540 items, mostly thin signal
Vocab from STRONG signals only:                6,266 items
Training triples:                              535,785
Training examples per vocab item:              535,785 / 6,266 = ~85

Each item gets ~85 gradient updates -> genuinely learned embeddings.
ANN search on well-trained vectors returns relevant items.
Warm-user Recall@20: 0.1231 (BEATS Phase 0's 0.0656)
```

This is the healthy side of the tension: dense feedback means even a modest vocab has abundant signal per item. On a **sparse** log the same "use all events" choice can collapse to <1 example per item, producing near-random embeddings that lose to popularity -- the classic Phase 1 failure. The lever is the same either way: keep the vocab no larger than your training signal can support.

```python
# Vocab from STRONG signals only, with a minimum-interaction floor
strong_events = train_events.filter(pl.col("signal") == "strong")
# Items with >= 3 strong interactions get an embedding -> 6,266 items
```

**The general principle: your vocab should never be larger than your training signal can support.** A rough heuristic: each item should have at least 3-5 positive training examples. If your training data is sparse, shrink the vocab, not the model.

**Rule 21: Feature count should scale with data volume.** "Scale your learning to the size of your data -- roughly linear relationship between examples and useful weights." This applies equally to embedding vocab size.

---

## 5. Results & Honest Interpretation

### The Numbers

| Metric | Phase 0 (Heuristic) | Phase 1 (Two-Tower) |
|---|---|---|
| Recall@20 | 6.70% | **12.31%** |
| Warm user Recall@20 | 6.56% | **12.31%** |
| Cold-start Recall@20 | 11.75% | 12.13% (heuristic fallback) |
| Catalog coverage | 7.14% | **15.70%** |
| Training time | N/A | ~2 min (20 epochs, CPU) |
| Index size | N/A | ~1.6 MB (6.3K items x 64-dim) |

### Why Phase 1 BEAT Phase 0 -- and What That Tells You

On a *sparse* e-commerce log, the textbook result is that a two-tower loses to a
strong popularity+category heuristic. On KuaiRand it's the opposite: the two-tower
wins by +84% on recall and +120% on coverage. The difference is entirely the data
regime:

- **Dense feedback.** A third of events are strong signals, so users have long,
  informative histories -- mean-pooling many item embeddings yields a rich user
  vector, not a guess from 2-3 items.
- **Enough signal per item.** ~85 training examples per vocab item (vs the ~4 you
  might get on a sparse log) means the embeddings actually converge.
- **Small catalog.** ~7.5K items is tractable for ID embeddings to cover.

> The honest counterpoint: **this win is a property of the dataset, not proof the
> architecture is universally better.** Run the same ID-only, mean-pooled
> two-tower on short histories over millions of items with mostly weak feedback,
> and it can easily lose to the heuristic. "We used a two-tower" is never the
> result; "we beat the baseline by X%, measured temporally, and here's the data
> reason why" is.

### What This Means for the Ranker (Phase 2)

The two-stage frame still holds: Stage 1 (this two-tower) maximizes recall into a
candidate set; Stage 2 (the ranker) maximizes precision within it. But Phase 2
delivered an *honest negative*: a single coarse category cross feature actually
**hurt** the ranker (-12% NDCG vs popularity order), because on KuaiRand the tag
is weak and engagement is popularity-driven. The lesson compounds -- a strong
candidate stage plus a weak ranking feature is worse than the candidate stage
alone. Better features (recency, sequence, richer side data) or a better candidate
union (Phase 7's co-visitation) are the way forward, not a fancier ranker on thin
signal. See [`docs/phase2.md`](docs/phase2.md).

---

## 6. Google's Rules of ML Applied

| Decision | Rule | How it applied |
|---|---|---|
| Ship heuristic first | Rule 1 | Launched popularity ranker before building any ML |
| Design event schema first | Rule 2 | Instrumented impression/click/cart/purchase before training |
| Encode domain knowledge | Rule 7 | Signal weights (3/2/1) encode the intent hierarchy |
| Know freshness requirements | Rule 8 | 15-min freshness -> streaming feature updates, not daily retrain |
| Watch for silent failures | Rule 10 | RowCount/NullRate assertions on every pipeline run |
| Start with logistic regression | Rule 14 | Ranker starts as LR before moving to XGBoost/DNN |
| Plan to launch and iterate | Rule 16 | MLflow tracking every run; A/B framework from day one |
| Prefer direct features | Rule 17 | Strong signals as positives; not weak exposures (noisy indirect signal) |
| Feature count scales with data | Rule 21 | Vocab size constrained to items with sufficient training signal |
| Observable, attributable metrics | Rule 13 | Target-action rate (direct signal) over exposure rate (indirect) |
| Recall vs. business metric | Rule 25 | Offline Recall@20 is ticket to A/B, not final answer |
| Temporal holdout | Rule 33 | Split at 80th percentile of timeline, not random |
| Shared train/serve code | Rule 32 | Mean-pool at training = mean-pool at serving (same function) |
| Position features separate | Rule 36 | `position` logged at serving time; zeroed at inference |
| Measure training/serving skew | Rule 37 | Point-in-time item property lookups; feature store in Phase 2 |
| Objectives as proxies | Phase III | Recall metric plateau -> need qualitatively new signal (ranker) |

---

## Key Takeaways

1. **The model is almost never the bottleneck.** Infrastructure, data quality, and evaluation methodology are. Fix those first.

2. **A regression is information.** When Phase 2's category cross feature made the
   ranker *worse* (-12% NDCG), we didn't hide it -- we diagnosed why (a coarse,
   weak signal) and let Phase 5's A/B test confirm it (-10%, p=0.0006) and kill
   the change. Negative results that surface early (in development or a replay
   A/B) cost hours; the same regression shipped to production costs revenue.

3. **A strong baseline is a strong baseline. Measure coverage too.** Popularity is
   a genuinely hard baseline on many datasets. A recommender that serves the same
   20 items to everyone is not a recommender -- it's a bestseller list. Track
   coverage alongside recall so you know which one you've built.

4. **Cold-start is a first-class concern, not an edge case.** 30% of sessions being cold-start means your architecture must handle it from day one. A user ID embedding table is a trap.

5. **Evaluation interfaces must be stable.** The `catalog_size` bug would have been invisible until we tested Phase 1. Stable interfaces across model versions make comparison reliable.

6. **Two-stage systems need two-stage evaluation.** Evaluating candidate generation on a ranking metric (Recall@20) is like evaluating a search index on whether the top-1 result is correct. The right metric for retrieval is Recall@K where K is your candidate set size.

---

*Dataset: [KuaiRand-Pure](https://kuairand.com) | ~1.44M standard-log interactions, ~27K users, ~7.5K videos, plus a ~1.19M-row uniform-random exposure log that powers Part II (off-policy evaluation).*

*Rules reference: [Google's Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)*
