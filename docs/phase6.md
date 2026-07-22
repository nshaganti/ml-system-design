# Phase 6 -- Near-Real-Time Freshness

> **Google's Rule 8:** *Know the freshness requirements of your system.* A model
> that's a day stale can quietly cost you a large fraction of its value.

This is the capstone of Part I. Everything so far -- baseline, model, feature
store, serving, monitoring, experimentation -- assumed features were computed at
the training cutoff and frozen. Phase 6 closes the loop: a user's behavior from
*five minutes ago* should be able to shape their *next* recommendation. Whether it
*helps* turns out to be dataset-dependent -- and on KuaiRand the honest answer is
"barely," which is a lesson in itself.

---

## The problem: a frozen store is blind to right now

Our batch feature store (Phase 2) is a snapshot at the train/serve cutoff. That's
correct for a daily-retrained model -- but it means the store has *no idea* what a
user did this session. Someone who just long-viewed three cooking videos looks
identical to someone who did nothing. Their `user_cat_affinity` for that category
is whatever it was at last night's batch job: possibly zero.

The naive fix -- "just retrain more often" -- is both expensive and slow. You
can't retrain a model every 30 seconds, and even hourly retraining misses the
session that's happening *now*.

## The design decision: stream features, not models

The design doc's answer (and the right one) is to separate two things that are
usually conflated:

- **The model** changes slowly (daily retrain). It's expensive and risky to
  update; you want it stable and well-tested.
- **The features** change fast (every event). They're cheap to update -- just
  counters.

So you keep the model frozen and **stream fresh features into the online store**:

```
User buys item -> Kafka -> Flink updates Redis (~30s lag) -> next request
                  sees the updated features.   NO retrain required.
```

This is dramatically simpler and more reliable than *online learning* (retraining
on the stream), which the doc explicitly reserves for sub-minute SLAs because it
introduces catastrophic forgetting, feature-distribution instability, and
time-dimension skew. For a 15-minute freshness target, streaming features wins.

### The implementation: a delta layer with the same interface

`phase6/streaming_store.py` wraps the frozen batch store and layers O(1)
incremental updates on top. Each streamed strong event bumps three counters:

```python
def ingest(self, user_id, item_id, event_type):
    self._session_items[user_id].add(item_id)          # for the freshness rule
    if event_type in STRONG_EVENT_TYPES:
        self._item_delta[item_id] += 1
        self._user_delta[user_id] += 1
        self._user_cat_delta[(user_id, category_of(item_id))] += 1

# reads return batch value + streamed delta
```

The crucial design choice: `StreamingFeatureStore` exposes the **exact same
`get_online_features_batch` interface** as the batch store. So the Phase 3
`RecommendationService` consumes it **with zero changes** -- you swap the store
and the whole serving stack just works. This is the interface discipline from
Phase 3 paying off a second time (the first was the two-tower slotting behind the
candidate-generator protocol).

It also tracks per-session items so serving can apply the freshness rule the doc
names directly: *don't re-show the video they just watched.*

## Code tour

| File | Job |
|---|---|
| `streaming_store.py` | Frozen batch features + a live delta layer; `ingest()` (the Flink job in miniature) + the batch-compatible read interface. |
| `run.py` | A before/after single-user demo + a quantified in-session experiment. |

---

## Results

`cd phase6 && python run.py`:

**Qualitative -- one user, before vs after streaming one event:**

```
a user streams a strong signal on a new item (a new category for them)
  user_pop          : n -> n+1
  user_cat_affinity : 0 -> 1   (the model now knows the in-session intent)
  session dedup     : that item now suppressed -> True
```

The mechanism works exactly as designed: one streamed event updates the online
features with no retrain. The question is whether it changes outcomes.

**Quantitative -- frozen vs fresh, hit@20 on the user's *later* items:**

For 18,883 users with >=2 strong events, we stream their earliest event as a
"seed" and measure whether their *later* items show up in the top-20. Both arms
exclude the seed, so the only difference is feature freshness.

```
frozen  batch features   : hit@20 = 0.2343 (4,425/18,883)
fresh streaming features : hit@20 = 0.2348 (4,434/18,883)
absolute lift=+0.0005   relative=+0.2%
z=0.109   p=0.9130   -> NOT significant
```

### Reading the numbers like an engineer

- **+0.2%, p=0.91 -- no measurable effect.** Streaming one in-session event moved
  the needle by nine users out of ~18,900. The confidence interval straddles
  zero; we cannot claim freshness helped here. And the model was byte-for-byte
  identical between arms, so this is a clean read on the *feature-freshness* lever
  alone.

- **Why freshness is flat *here* (and when it wouldn't be).** KuaiRand is
  short-video engagement: dense, exploratory, and not strongly session-intent
  driven. Knowing a user just long-viewed one video tells you little about their
  *next* one. Contrast e-commerce, where a user browsing a category is *about* to
  buy more in it -- there, a 30-second delta layer can be worth a large lift. The
  value of freshness is a property of the **workload**, not the technology.

- **Same machinery, opposite verdict from Phase 5 -- both honest.** Phase 5 found
  a *significant negative* (don't ship the LR ranker). Phase 6 finds *no effect*
  (don't invest in a streaming layer for this workload yet). Neither is a
  disappointment; both are the monitoring/experimentation discipline telling you
  where **not** to spend effort.

> **The honest caveat:** this is a replay -- we stream logged events and check
> logged later items. The mechanism (delta layer, same interface, no retrain) and
> the statistics are production-shaped; the *result* just says freshness isn't the
> lever on this dataset.

---

## What Phase 6 taught us

1. **Separate what changes fast from what changes slow.** Features stream;
   models retrain. Conflating them leads you to expensive online learning when
   cheap streaming features would do.
2. **Interface discipline compounds.** Because the streaming store mimics the
   batch store's read API, the entire Phase 3 serving stack absorbed it for free.
   Good seams pay off repeatedly.
3. **Whether fresh data helps is a property of the workload.** The streaming
   machinery is real and correct, but on short-video engagement it moved nothing
   (+0.2%, ns). Build the lever; measure before you assume it pays (Rule 8).

---

## The journey, end to end

That completes **Part I (Phases 0-7)** -- the full classic-recommender lifecycle:

| Phase | What it added | Standout result (KuaiRand) |
|---|---|---|
| 0 | Heuristic baseline | Recall@20 = 0.067; temporal eval discipline |
| 1 | Two-tower retrieval | **beat** the heuristic +75%, coverage +113% |
| 2 | Feature store + LR ranker | skew measured 2-2.4x; weak cross feature *hurt* (-12% NDCG) |
| 3 | Serving architecture | p50 4.1ms, graceful fallback, Rule 29 logging |
| 4 | Monitoring & drift | drift gate FAILs by design |
| 5 | A/B testing | LR ranker significantly worse (-10%, p=0.0006) -- don't ship |
| 6 | Freshness | +0.2%, not significant -- freshness isn't the lever here |
| 7 | Co-visitation | +61% on session next-item |

Then **Part II (Phase 8)** delivers the punchline: the offline metrics under all
of this were **+100% biased**, and off-policy evaluation on the random log fixes
it. The through-line: **the model is the easy part.** The value -- and the
danger -- lives in evaluation discipline, point-in-time correctness, serving
robustness, monitoring, honest experimentation, and *causally sound* metrics. See
[`results.md`](results.md) and [`lessons-learned.md`](lessons-learned.md).
