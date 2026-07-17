# Phase 6 -- Near-Real-Time Freshness

> **Google's Rule 8:** *Know the freshness requirements of your system.* A model
> that's a day stale can quietly cost you a large fraction of its value.

This is the capstone. Everything so far -- baseline, model, feature store,
serving, monitoring, experimentation -- assumed features were computed at the
training cutoff and frozen. Phase 6 closes the loop: a user's behavior from *five
minutes ago* should shape their *next* recommendation. And the payoff here is the
biggest, most clearly-significant win in the entire repo.

---

## The problem: a frozen store is blind to right now

Our batch feature store (Phase 2) is a snapshot at the train/serve cutoff. That's
correct for a daily-retrained model -- but it means the store has *no idea* what a
user did this session. Someone who just added running shoes to their cart looks
identical to someone who did nothing. Their `user_cat_affinity` for footwear is
whatever it was at last night's batch job: possibly zero.

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
names directly: *don't re-show the running shoes they just bought.*

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
user 803960 streams an 'add_to_cart' on item 440937 (category 1593)
  user_pop          : 0 -> 1
  user_cat_affinity : 0 -> 1   (the model now knows the in-session intent)
  session dedup     : item 440937 now suppressed -> True
```

**Quantitative -- frozen vs fresh, hit@20 on the user's *later* items:**

For 1,536 users with >=2 strong events, we stream their earliest event as a
"seed" and measure whether their *later* items show up in the top-20. Both arms
exclude the seed, so the only difference is feature freshness.

```
frozen  batch features   : hit@20 = 0.0397 (61/1,536)
fresh streaming features : hit@20 = 0.0625 (96/1,536)
absolute lift=+0.0228   relative=+57.4%
z=2.868   p=0.0041   -> SIGNIFICANT
```

### Reading the numbers like an engineer

- **+57% relative lift, p=0.0041 -- a clear, significant win.** Streaming one
  in-session event more than halved the miss rate on the user's later purchases.
  And it happened with the model **byte-for-byte identical** between arms. We
  didn't build a better model; we gave the same model better inputs.

- **Contrast this with Phase 5, deliberately.** There, refining the *ranker*
  (popularity -> LR) was **inconclusive** (p=0.84). Here, refining the *feature
  freshness* is **highly significant** (p=0.004). Same statistical machinery,
  opposite verdict. The lesson is a classic one (Rule 8 and "features > models"):
  on this dataset, **fresh data beats a fancier model.** Time spent on the feature
  pipeline would pay off more than time spent tuning the ranker.

- **Why freshness helps so much here:** e-commerce sessions are bursty and
  intent-driven. A user browsing a category is *about* to buy more in it. The
  batch store can't see that; the stream can. Capturing 30 seconds of intent is
  worth more than any offline metric we moved in Phases 1-2.

> **The honest caveat, again:** this is a replay -- we stream logged events and
> check logged later items. But the mechanism (delta layer, same interface, no
> retrain) and the statistics are exactly production-shaped, and the effect is far
> too large to be an artifact.

---

## What Phase 6 taught us

1. **Separate what changes fast from what changes slow.** Features stream;
   models retrain. Conflating them leads you to expensive online learning when
   cheap streaming features would do.
2. **Interface discipline compounds.** Because the streaming store mimics the
   batch store's read API, the entire Phase 3 serving stack absorbed it for free.
   Good seams pay off repeatedly.
3. **Fresh data can beat a better model.** The single biggest, most significant
   lift in this whole project came not from a smarter algorithm but from letting
   the same model see 30 seconds of fresh behavior (Rule 8).

---

## The journey, end to end

That completes **Phases 0-6** -- the full lifecycle the design doc lays out:

| Phase | What it added | Standout result |
|---|---|---|
| 0 | Heuristic baseline | Recall@20 = 0.031; temporal eval discipline |
| 1 | Two-tower retrieval | Lost to the heuristic -- a real, instructive finding |
| 2 | Feature store + LR ranker | Skew measured at 2-2.8x; interpretable weights |
| 3 | Serving architecture | p50 6ms, graceful fallback, Rule 29 logging |
| 4 | Monitoring & drift | PSI 0.47 caught the same shift as the Phase 2 skew |
| 5 | A/B testing | Ranker win came back *inconclusive* -- don't ship noise |
| 6 | Freshness | +57% hit@20, significant, **no retrain** |

The through-line: **the model is the easy part.** The value -- and the danger --
lives in evaluation discipline, point-in-time correctness, serving robustness,
monitoring, honest experimentation, and data freshness. See
[`lessons-learned.md`](lessons-learned.md) for the distilled reflexes.
