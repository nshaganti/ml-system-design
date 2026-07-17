# Phase 3 -- The Serving Architecture

> **Google's Rule 29:** *The best way to make sure you train like you serve is to
> save the set of features used at serving time, and then pipe those features to a
> log to use them at training time.*

Phases 0-2 were offline: load a CSV, train, evaluate. Phase 3 is the first time
the system faces a *live request*. A user hits the homepage; you have ~100ms to
fetch features, generate candidates, rank them, apply business rules, and
respond. This walkthrough builds that request handler and stress-tests it.

---

## The problem: a 100ms budget, assembled from parts

Everything you built so far now has to cooperate inside one function, under a
hard deadline. The design doc's target breakdown:

```
1. Fetch user features            ~5ms
2. Candidate generation (Stage 1) ~10ms
3. Batch fetch item features      ~15ms
4. Rank candidates (Stage 2)      ~30ms
5. Business rules                 ~2ms
6. Return + async feature log
                                  -----
                          total < 100ms
```

The offline mindset ("is my metric good?") is replaced by three new questions
you never asked in a notebook: **Is it fast enough? What happens when a part
breaks? How do I make the *next* model better without repeating my mistakes?**

## The design decisions

### 1. Program to an interface, not a model (Stage 1)

`service.py` depends on a `CandidateGenerator` *protocol* -- anything with a
`.generate(user_id, n)` method -- not on a concrete class. Today we ship
`PopularityCandidateGenerator` (no model to train, always available). Tomorrow
you drop in the Phase 1 two-tower ANN behind the same method and **nothing
downstream changes.**

> This is the two-stage architecture's real superpower: Stage 1 and Stage 2
> evolve independently. You can A/B test a new candidate generator without
> touching the ranker.

### 2. Degrade, don't crash (Rule 10)

Production ML systems don't 500 -- they degrade. If the ranker throws or
overruns, the service catches it and returns the candidate order instead:

```python
def _rank(self, features, candidates, n):
    if self.ranker is None:
        return candidates[:n], True          # no ranker -> candidate order
    try:
        return self.ranker.rank(features, item_col="item_id", n=n), False
    except Exception as e:                    # never let ranking 500 the user
        print(f"[service] ranker failed ({e}); falling back")
        return candidates[:n], True
```

The response carries `fallback_used` and flips `model_version` to `"fallback"`,
so monitoring can *count* degraded requests -- a silent fallback is its own kind
of outage.

### 3. Business rules are features you didn't discard (Rule 7)

Out-of-stock items must never be recommended, no matter how high the model scores
them. We derive the OOS set from the `available` item property (most-recent value
before the cutoff == `"0"`) and filter it *after* ranking. On this dataset that's
**357,094 items** filtered -- a huge, non-negotiable correctness rule that has
nothing to do with the model.

### 4. Log the exact features served (Rule 29 -- the big one)

This is the most important line of code in the whole phase:

```python
self.feature_log.append({
    "request_id": request_id,
    "user_id": request.user_id, "item_id": item_id,
    "features": {"item_pop": ..., "user_pop": ..., "user_cat_affinity": ...},
})
```

When you build training data for v2, you **join tomorrow's clicks to this log**,
not to the current feature store. Why does that matter? Because the log records
the features *as they actually were at serving time*. Joining to the current
store would reintroduce exactly the training-serving skew we measured at 2-2.8x
in [Phase 2](phase2.md). Rule 29 closes the loop: **serving generates its own
skew-free training data.**

## Code tour

| File | Job |
|---|---|
| `candidate_generator.py` | Stage 1 behind a `Protocol`; popularity implementation. |
| `service.py` | The request handler: 6 timed stages, fallback, feature logging. |
| `run.py` | Assembles it on real data; single-request trace + load test + fault injection. |

The `RecommendationResponse` returns not just items but `request_id`,
`model_version`, `fallback_used`, and a per-stage `latency_ms` dict -- everything
monitoring and debugging will need.

---

## Results

`cd phase3 && python run.py`:

**Single request, per-stage latency (cold):**

```
candidate_generation    0.008 ms
feature_fetch          22.011 ms
ranking                 0.711 ms
business_rules          0.072 ms
feature_log             9.201 ms
TOTAL                  32.000 ms   (budget 100ms -> OK)
```

**Load test over 2,000 real users:**

| Metric | Value |
|---|---|
| p50 latency | **6.2 ms** |
| p99 latency | **14.4 ms** |
| within 100ms budget | **100%** of requests |
| out-of-stock filtered | 357,094 items |
| inference-log rows | 40,020 |

**Fault injection:** we swapped in a ranker that always raises. The service
returned 20 items anyway, `fallback_used=True`. No 500. Rule 10 in action.

### Reading the numbers like an engineer

- **The first request is slow (32ms), steady-state is fast (p50 6ms).** That gap
  is warmup -- the first `feature_fetch`/`feature_log` pay one-time costs. This is
  why you measure p50/p99 over many requests, never a single call.
- **`feature_fetch` dominates.** Even in a toy in-memory store, feature I/O is
  the bottleneck -- exactly why production puts features in Redis for <5ms lookups
  and why the design doc budgets 15ms for it. Optimize where the time actually is.
- **`ranking` is cheap (0.7ms)** because logistic regression is a dot product.
  This is the latency dividend of the "start simple" decision (Rule 14) -- a DNN
  ranker would cost far more here.

> **Honest caveat:** our latencies are flattering because everything is in-process
> and in-memory. Real network hops to Redis/Qdrant add milliseconds. The *value*
> here isn't the absolute numbers -- it's that the architecture measures every
> stage, so when you move to real infra you'll know exactly where the budget goes.

---

## What Phase 3 taught us

1. **Serving is assembly under a deadline.** The hard part isn't any single
   component; it's making them cooperate in <100ms and knowing where the time
   goes.
2. **Degradation is a feature, not an afterthought.** `fallback_used` turns a
   silent failure into a countable, monitorable event.
3. **Rule 29 is the flywheel.** Logging served features is what lets each model
   generation train skew-free on the last generation's real traffic. This single
   habit is the difference between a system that improves and one that mysteriously
   rots.

Next up: **Phase 4 (monitoring)** consumes these inference logs to detect drift,
and **Phase 5 (A/B testing)** uses `model_version` to compare v1 vs v2 on live
traffic. See the [roadmap](../README.md#roadmap).
