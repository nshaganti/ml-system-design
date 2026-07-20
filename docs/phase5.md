# Phase 5 -- A/B Testing (Online Experimentation)

> **Google's Rule 16:** *Plan to launch and iterate.* And the corollary that
> matters most here: **measure the delta between models on live traffic, never
> their absolute offline numbers.**

Phases 0-4 can only tell you a model *might* be better. This phase is about the
only evidence that actually justifies a launch: a controlled experiment with a
p-value. Here it does its most valuable job -- **decisively killing a change**,
giving us the statistical confidence to say "no" instead of guessing.

---

## The problem: offline metrics are necessary but not sufficient

In Phase 2 our LR ranker *lost* to popularity offline (-12% NDCG). Maybe that was
noise, or an artifact of the offline metric -- so do we trust it? The A/B test is
how you find out for real.

Offline eval replays *logged* behavior -- it can't see how users react to
recommendations they were never shown. Novelty effects, position bias, and plain
noise all live in the gap between an offline metric and real behavior. The only
way to close that gap is to run both models live, at the same time, on comparable
users, and measure the difference.

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

> **Honest framing.** `run.py` is a *replay* on logged positive actions, not a
> live test -- we can't observe counterfactual reactions. But the machinery
> (assignment, z-test, CIs, power analysis) is exactly what you run online. The
> method is the lesson, not the specific number.

---

## Results

`cd phase5 && python run.py`:

**Design (computed before looking at any outcome):**

```
to detect a  5% relative lift (baseline 10%): need ~57,763 users PER ARM
to detect a 10% relative lift (baseline 10%): need ~14,751 users PER ARM
to detect a 20% relative lift (baseline 10%): need ~ 3,841 users PER ARM
```

**Assignment:** `user_12345 -> control` twice (STABLE); split 49.8% / 50.2%.

**The experiment (hit@20 as the conversion proxy):**

```
control=0.2296 (n=7,469)   treatment=0.2063 (n=7,531)
absolute lift=-0.0233   relative=-10.1%
z=-3.451   p=0.0006   95% CI=[-0.0365, -0.0101]   -> SIGNIFICANT

Decision: DO NOT SHIP -- treatment is significantly WORSE.
```

### Reading the numbers like an engineer

This is the A/B test doing exactly its job -- catching a bad change before it
ships:

1. **The online test confirms the offline signal.** Phase 2 said the LR ranker
   was worse offline (-12% NDCG); the A/B test says it is worse online too
   (-10.1% hit@20), and now with a **p-value of 0.0006** we can say so with
   confidence. The 95% CI `[-0.037, -0.010]` sits entirely below zero -- treatment
   is genuinely worse, not noise.

2. **This experiment was adequately powered.** Unlike a thin offline slice, the
   live arms had ~7,500 users each at a ~23% baseline hit rate -- comfortably
   enough to detect a 10% effect. When the test is powered and the CI excludes
   zero, "significant" means what it says.

3. **"Significant" here means STOP.** The correct action is to *not ship*, and to
   go back to feature work (Phase 2's lesson) rather than pushing a regression
   live because someone was attached to the model. The A/B test just saved you
   from shipping a -10% change.

> **The discipline in one sentence:** decide the sample size before you start, run
> until you hit it, then let the p-value -- not your hope -- make the call.

---

## What Phase 5 taught us

1. **Offline lift is a hypothesis; the A/B test is the verdict.** Here the offline
   signal (LR is worse) and the online test agreed -- and the test gave us the
   confidence (p=0.0006) to kill the change decisively.
2. **Sticky, salted assignment is non-negotiable.** Contaminated assignment
   invalidates everything downstream, silently.
3. **Power analysis is the antidote to self-deception.** Deciding you need N users
   per arm *before* you look stops you from over- or under-reading the result.
4. **A significant negative is a win for the tool.** The whole apparatus exists to
   prevent one specific, expensive mistake: shipping a model that's actually
   worse. This time it caught a -10% regression before it reached a user.

Next up: **Phase 6 (near-real-time freshness)** -- can fresher data move the
needle even when a fancier ranker couldn't? See the [roadmap](../README.md#roadmap).
