# Phase 5 -- A/B Testing (Online Experimentation)

> **Google's Rule 16:** *Plan to launch and iterate.* And the corollary that
> matters most here: **measure the delta between models on live traffic, never
> their absolute offline numbers.**

Phases 0-4 can only tell you a model *might* be better. This phase is about the
only evidence that actually justifies a launch: a controlled experiment with a
p-value. It's also where you learn the most deflating lesson in applied ML --
**an offline improvement often evaporates when you test it properly.**

---

## The problem: offline metrics are necessary but not sufficient

In Phase 2 our LR ranker beat popularity by +2% NDCG offline. Ship it, right?

No. Offline eval replays *logged* behavior -- it can't see how users react to
recommendations they were never shown. Novelty effects, position bias, and plain
noise all live in the gap between "offline metric went up" and "users actually
behaved better." The only way to close that gap is to run both models live, at
the same time, on comparable users, and measure the difference.

Two things you cannot get wrong when you do that:

### 1. Deterministic, sticky assignment

A user must see the **same variant** every visit for the life of the experiment.
If assignment flip-flops, a single user contributes to both arms and the
comparison is contaminated. Our `assign_variant` (`phase5/experiment.py`) hashes
`experiment_name + user_id`:

```python
digest = hashlib.sha256(f"{experiment_name}:{unit_id}".encode()).hexdigest()
bucket = int(digest[:8], 16) / 0xFFFFFFFF     # uniform in [0, 1]
# ...walk the cumulative weights to pick a variant
```

- **Sticky:** same inputs -> same bucket, forever, with *no database*. Any server
  computes the same assignment independently.
- **Salted:** mixing in `experiment_name` means two concurrent experiments assign
  the same user independently -- no accidental correlation between tests.

### 2. Statistical significance (not just "the number went up")

A raw "+2%" is meaningless without asking: *could this be noise?* We implement a
**two-proportion z-test** in pure Python (via `math.erf` for the normal CDF -- no
scipy dependency) that returns a p-value, a 95% confidence interval, and a
ship/no-ship verdict. And critically, a **sample-size calculator** you run
*before* the experiment to answer "how long must this run to detect the effect I
care about?"

> The cardinal sin this prevents: peeking at an underpowered experiment, seeing a
> number tick up, and shipping noise. Deciding your sample size up front is what
> keeps you honest.

## Code tour

| File | Job |
|---|---|
| `experiment.py` | `assign_variant` (sticky/salted), `two_proportion_ztest`, `required_sample_size`. Pure, dependency-free, unit-tested. |
| `run.py` | An offline **replay** A/B test: control=popularity, treatment=LR ranker, metric=hit@20. |

> **Honest framing.** `run.py` is a *replay* on logged purchases, not a live test
> -- we can't observe counterfactual reactions. But the machinery (assignment,
> z-test, CIs, power analysis) is exactly what you run online. The method is the
> lesson, not the specific number.

---

## Results

`cd phase5 && python run.py`:

**Design (computed before looking at any outcome):**

```
to detect a  5% relative lift (baseline 10%): need ~57,763 users PER ARM
to detect a 10% relative lift (baseline 10%): need ~14,751 users PER ARM
to detect a 20% relative lift (baseline 10%): need ~ 3,841 users PER ARM
```

**Assignment:** `user_12345 -> control` twice (STABLE); split 50.5% / 49.5%.

**The experiment (hit@20 as the conversion proxy):**

```
control=0.0401 (n=1,248)   treatment=0.0385 (n=1,222)
absolute lift=-0.0016   relative=-4.0%
z=-0.205   p=0.8376   95% CI=[-0.0169, +0.0137]   -> not significant

Decision: INCONCLUSIVE -- the delta is inside the noise band.
```

### Reading the numbers like an engineer

This result is a *gift*, because it teaches three things at once:

1. **The offline win did not prove out.** The LR ranker's +2% NDCG (Phase 2)
   produced **no significant difference** in hit@20 (p=0.84). The confidence
   interval `[-0.017, +0.014]` straddles zero -- we genuinely cannot tell the
   models apart on this metric and sample. *This is the norm, not the exception:*
   most offline wins shrink or vanish under a proper test.

2. **We were doomed to be inconclusive, and the math said so up front.** The power
   analysis called for ~14,751 users per arm to catch a 10% lift; we had ~1,235.
   The experiment was **~12x underpowered before it started.** Running it was
   always going to yield "inconclusive." That's why you compute sample size
   *first* -- to know whether the experiment can even answer the question.

3. **"Not significant" is a real, actionable answer.** It does *not* mean "ship
   the one that's slightly higher." It means: gather more data, or accept the
   models are equivalent and prefer the simpler/cheaper one. Shipping treatment
   here because it's "close" would be shipping noise.

> **The discipline in one sentence:** decide the sample size before you start, run
> until you hit it, then let the p-value -- not your hope -- make the call.

---

## What Phase 5 taught us

1. **Offline lift is a hypothesis, not a result.** The A/B test is the experiment
   that confirms or kills it. Ours got killed (well, ruled inconclusive) -- and
   that's a successful use of the tool.
2. **Sticky, salted assignment is non-negotiable.** Contaminated assignment
   invalidates everything downstream, silently.
3. **Power analysis is the antidote to self-deception.** Knowing you need 14.7k
   users per arm stops you from over-reading 1.2k.
4. **"Inconclusive" protects you.** The whole apparatus exists to prevent one
   specific, expensive mistake: shipping a model because a noisy number went up.

Next up: **Phase 6 (near-real-time freshness)** -- closing the loop so the next
model trains on fresh data (and future experiments have the volume to actually
conclude). See the [roadmap](../README.md#roadmap).
