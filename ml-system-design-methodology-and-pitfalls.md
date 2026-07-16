# ML System Design: Methodology, Decisions & Real-World Pitfalls

> **Audience:** ML engineers moving from prototype notebooks to production systems.
> **Vehicle:** A real-time product recommendation engine built on the [Retail Rocket dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).
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

> &#9888;&#65039; **Real-world problem:** Recall@20 can be gamed. A model that recommends the 20 globally most popular items every time will achieve decent recall on sparse datasets (popular items get bought). This is exactly what happened in our Phase 0/Phase 1 comparison -- the popularity baseline "won" on recall because popular items overlap heavily with what users actually buy.

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

On the Retail Rocket dataset (138 days), we split at 80% of the timeline -- training on the first 110 days, evaluating on the last 28.

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

### Why Those Specific Weights? (purchase=3, cart=2, view=1)

```python
EVENT_WEIGHTS = {
    "purchase":    3.0,
    "add_to_cart": 2.0,
    "impression":  1.0,
}
```

These are not arbitrary. They encode a purchase intent signal hierarchy:

| Event | Intent signal | Reasoning |
|---|---|---|
| Purchase | Highest -- irreversible action | User spent money |
| Add to cart | High -- considered action | User mentally committed |
| View/Impression | Low -- passive signal | Could be accidental, search-driven, or curiosity |

The ratio (3:2:1) is a reasonable starting point. **Rule 7: Convert heuristics into features.** These weights encode domain knowledge that the ML model will later learn automatically from data. By making them explicit and tunable now, you create a feature that can be validated, adjusted, and eventually replaced by a learned signal.

**What not to do:** Equal weights (1:1:1) treats a purchase the same as a scroll-past. This produces a popularity score dominated by view counts -- which is exactly the noise the ML model will have to fight.

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

**Problem 1 -- Cold-start.** On the Retail Rocket dataset, 30% of sessions are from users with no prior history. The user embedding table has no entry for them. What does `self.user_emb(new_user_id)` return? Random noise (or a crash, if the ID is out of range). You silently serve garbage recommendations to 30% of your traffic.

**Problem 2 -- Training-serving skew (Rule 32).** The user embedding is learned during training. At serving time, you look up the same embedding. But if a user's preferences have changed since training (they just bought running shoes; now they want race nutrition), the stale embedding doesn't reflect this. The user vector is only updated on the next model retrain.

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

### Why Cart + Purchase as Positives ONLY?

**The data breakdown on Retail Rocket:**

| Event type | Count | % of total |
|---|---|---|
| Impressions (views) | 2,664,312 | 96.7% |
| Add to cart | 69,332 | 2.5% |
| Purchases | 22,457 | 0.8% |

**What happens if you use all events as positives?**

With views as positives, 96.7% of your training signal says "the model is doing great -- every item shown gets a 'positive' label." The remaining 3.3% (cart + purchase) are drowned out.

> &#9888;&#65039; **Real-world failure we hit:** In our first training run, we defined positives as all events. Here is what happened:

```
First run results:
  Vocab size: 78,115 items (all items with >=5 interactions)
  Training triples: 53,714
  Training examples per item: 0.7 (less than 1!)
  
  Result:
    Warm user Recall@20: 0.0103  <-- WORSE than Phase 0's 0.0466
    Catalog coverage:    6.7%    <-- barely better than heuristic
```

The model learned nothing useful. Near-random embeddings for 78K items, searched with ANN, returned near-random results. **Popularity ranker beat it easily.**

**The fix: positives = strong events only**

```python
STRONG_EVENT_TYPES = {"add_to_cart", "purchase"}

# Vocab: only items that appear in strong events
strong_events = train_events.filter(
    pl.col("event_type").is_in(list(STRONG_EVENT_TYPES))
)
# Result: 8,219 items (from 212,915 unique) with >= 3 strong interactions each
```

**With the fix:**
- Each vocab item has meaningful training signal (at least 3 strong-event examples)
- Loss converges from 0.69 to 0.06
- Coverage jumps from 6.3% to **47.1%** -- the model surfaces diverse items

**Rule 17: Prefer directly observed features over learned ones.** Strong behavioral signals (cart, purchase) are direct observations of intent. Views are indirect and noisy. Always prioritize direct signal, especially when data is sparse.

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

**The intuition:** If a user bought running shoes, we want `score(user, running_shoes) > score(user, random_item)`. We don't need to claim the random item is bad -- just that the purchased item is better.

---

### Why L2-Normalize Embeddings in the Index?

```python
@staticmethod
def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(norms > 0, x / norms, x)
```

**Unnormalized dot product = cosine similarity x magnitude product.**

If item embeddings have different magnitudes (popular items tend to get larger gradient updates and larger norms), then the dot product search is biased toward high-magnitude items regardless of directional similarity. This reintroduces popularity bias through the back door.

After L2 normalization, dot product = pure cosine similarity -- only the direction (preference alignment) matters, not the magnitude. This gives the embedding model a fair shot at surfacing long-tail items that are directionally similar to user preferences.

**The tradeoff:** Sometimes magnitude IS signal (popular items are good by definition). You can tune this by partial normalization or mixing normalized and unnormalized scores. For Phase 1, full normalization is the right starting point because our goal is explicitly to improve over the popularity baseline.

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

The Retail Rocket item_properties table has timestamped property updates -- every category, price, and availability change is a new row. By filtering to `<= event_timestamp`, we see the item as it was at that moment, not as it is today.

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

We split at the 80th percentile of the timeline. Training on the first 110 days of Retail Rocket, evaluating on the last 28. **The model never sees any information from the test period during training.**

**Rule 33: Test on data after your training cutoff.** No exceptions. Always check that your split is temporal before trusting any offline metric.

---

### Pitfall 3: Coverage vs Recall Tradeoff

**What happened in our run:**

| Metric | Phase 0 (Heuristic) | Phase 1 (Two-Tower) |
|---|---|---|
| Recall@20 | 3.08% | 2.21% |
| Catalog coverage | 6.3% | 47.1% |

**Naive interpretation:** Phase 1 is worse. Throw it away.

**Correct interpretation:** Phase 1 has a fundamentally different distribution of recommendations.

Phase 0 (popularity ranker) concentrates 100% of its recommendations on the globally top ~15K items. It achieves decent recall because popular items are also what people buy. But it gives zero exposure to the other 220K items in the catalog.

Phase 1 (two-tower) spreads recommendations across 47% of the catalog (110K items). It's learning personalized preferences, but on a sparse dataset (36K training triples) without enough signal to outperform raw popularity on the "what do people buy" metric.

**Why this is actually expected:**

> &#9888;&#65039; **Popularity baselines are notoriously hard to beat on Recall@K metrics on sparse datasets.** This is a well-known result in recommendation systems research. The reason: popular items are popular because many people buy them. A recall metric asks "did we put the right item in the top 20?" -- and for most users on most e-commerce datasets, at least one of the top-20 globally popular items IS something they'll buy.

**What this means for Phase 2:**

The two-stage system (candidate gen + ranker) is designed to be evaluated as a whole pipeline. The candidate generator's Recall@500 is the right metric for Phase 1 -- did the right item make it into the 500 candidates? The ranker's Recall@20 is the right metric for Phase 2 -- did the ranker pick the right 20 from those 500 candidates?

Evaluating Phase 1 on Recall@20 (a Phase 2 metric) set an unfair bar. We did it here intentionally so you can see this exact failure mode.

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

**Our actual failure (first run):**

```
Vocab: 78,115 items (all items with >=5 any-event interactions)
Training triples: 53,714 (from cart/purchase events)
Training examples per item: 53,714 / 78,115 = 0.69

The model gets less than 1 gradient update per item on average.
Embeddings are essentially random initialization.
ANN search on random vectors returns random items.
Warm user Recall@20: 0.0103 (WORSE than Phase 0's 0.0466)
```

**The fix:**

```python
# Vocab from STRONG events only
strong_events = train_events.filter(pl.col("event_type").is_in({"add_to_cart", "purchase"}))
# Items with >= 3 cart/purchase interactions get an embedding
# Result: 8,219 items (vs 78K)

# Training examples per vocab item: 36,367 / 8,219 = 4.4
# Each item gets ~4 training examples -> meaningful gradient signal
```

After the fix:
- Loss converges properly: 0.69 -> 0.06
- Coverage: 47.1% (model learns diverse representations within the smaller vocab)

**The general principle: your vocab should never be larger than your training signal can support.** A rough heuristic: each item should have at least 3-5 positive training examples. If your training data is sparse, shrink the vocab, not the model.

**Rule 21: Feature count should scale with data volume.** "Scale your learning to the size of your data -- roughly linear relationship between examples and useful weights." This applies equally to embedding vocab size.

---

## 5. Results & Honest Interpretation

### The Numbers

| Metric | Phase 0 (Heuristic) | Phase 1 (Two-Tower) |
|---|---|---|
| Recall@20 | 3.08% | 2.21% |
| Warm user Recall@20 | 4.66% | 1.66% |
| Cold-start Recall@20 | 2.44% | 2.44% (Phase 0 fallback) |
| Catalog coverage | 6.3% | 47.1% |
| Training time | N/A | ~12 seconds (MPS) |
| Index size | N/A | 2.0 MB (8K items x 64-dim) |

### Why Phase 1 Didn't Beat Phase 0 on Recall -- and Why That's OK

**The dataset reality:**
- 96.7% of events are views (very weak signal)
- Purchase rate: 0.8% (sparse)
- 138 days of data -- not a lot of temporal diversity
- 36K training triples for 8K vocab items = 4.4 examples per item (marginal)

On sparse datasets with dominant popularity effects, a well-tuned popularity ranker is very hard to beat on Recall@K. This is a known empirical result in the field. It does NOT mean the ML model failed -- it means:

1. **The right metric for candidate generation is Recall@500, not Recall@20.** The two-tower's job is to make sure the right item is in the 500 candidates. The ranker's job is to select the best 20.

2. **Coverage improved massively (6.3% -> 47.1%)**, which is a real business win. Recommending from 6% of catalog means new items never get visibility. Recommending from 47% means the system can surface novel items -- which drives discovery, not just conversion.

3. **Phase 2 (logistic regression ranker) will address recall.** By scoring 500 candidates with user-item features (price affinity, category match, recency), the ranker will outperform the popularity baseline on both recall and precision. That is the correct evaluation frame.

### What Beating the Baseline Actually Requires

The two-stage system needs to be evaluated end-to-end. The correct comparison is:

```
Phase 0: Popularity ranker -> top 20 -> Recall@20 = 3.08%

vs.

Phase 1 + Phase 2: Two-tower -> 500 candidates
                  -> LR Ranker -> top 20
                  -> Recall@20 = ?
```

The ranker uses features the popularity model can't access:
- User-item price affinity (is this item in the user's typical price range?)
- Category affinity match (does this item match the user's top categories?)
- Item recency (is this a new item the user hasn't seen?)
- Co-purchase patterns (items frequently bought together)

With these features, the Phase 2 ranker selects the right 20 from 500 meaningful candidates -- outperforming the Phase 0 model that selects 20 from globally popular items.

---

## 6. Google's Rules of ML Applied

| Decision | Rule | How it applied |
|---|---|---|
| Ship heuristic first | Rule 1 | Launched popularity ranker before building any ML |
| Design event schema first | Rule 2 | Instrumented impression/click/cart/purchase before training |
| Encode domain knowledge | Rule 7 | Event weights (3/2/1) encode purchase intent hierarchy |
| Know freshness requirements | Rule 8 | 15-min freshness -> streaming feature updates, not daily retrain |
| Watch for silent failures | Rule 10 | RowCount/NullRate assertions on every pipeline run |
| Start with logistic regression | Rule 14 | Ranker starts as LR before moving to XGBoost/DNN |
| Plan to launch and iterate | Rule 16 | MLflow tracking every run; A/B framework from day one |
| Prefer direct features | Rule 17 | Cart/purchase as positives; not views (noisy indirect signal) |
| Feature count scales with data | Rule 21 | Vocab size constrained to items with sufficient training signal |
| Observable, attributable metrics | Rule 13 | Add-to-cart rate (direct signal) over click rate (indirect) |
| Recall vs. business metric | Rule 25 | Offline Recall@20 is ticket to A/B, not final answer |
| Temporal holdout | Rule 33 | Split at 80th percentile of timeline, not random |
| Shared train/serve code | Rule 32 | Mean-pool at training = mean-pool at serving (same function) |
| Position features separate | Rule 36 | `position` logged at serving time; zeroed at inference |
| Measure training/serving skew | Rule 37 | Point-in-time item property lookups; feature store in Phase 2 |
| Objectives as proxies | Phase III | Recall metric plateau -> need qualitatively new signal (ranker) |

---

## Key Takeaways

1. **The model is almost never the bottleneck.** Infrastructure, data quality, and evaluation methodology are. Fix those first.

2. **A regression is information.** When Phase 1 didn't beat Phase 0, we didn't throw away the model -- we debugged it. The vocab size vs training signal failure was a clear, fixable problem. Regressions that surface early (during development) cost hours; regressions that surface in production cost revenue.

3. **Popular items win on Recall@K. That's a reason to measure coverage too.** A recommendation system that serves the same 20 items to everyone is not a recommendation system -- it's a bestseller list. Coverage ensures the system can discover, not just confirm.

4. **Cold-start is a first-class concern, not an edge case.** 30% of sessions being cold-start means your architecture must handle it from day one. A user ID embedding table is a trap.

5. **Evaluation interfaces must be stable.** The `catalog_size` bug would have been invisible until we tested Phase 1. Stable interfaces across model versions make comparison reliable.

6. **Two-stage systems need two-stage evaluation.** Evaluating candidate generation on a ranking metric (Recall@20) is like evaluating a search index on whether the top-1 result is correct. The right metric for retrieval is Recall@K where K is your candidate set size.

---

*Dataset: [Retail Rocket E-Commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) | 2.7M events, 138 days, 1.4M users, 235K items*

*Rules reference: [Google's Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)*
