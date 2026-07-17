# Phase 4 -- Monitoring and Drift Detection

> **Google's Rule 10:** *Watch for silent failures.*

Every prior phase asked "is the model good?" Phase 4 asks the question that
actually keeps production systems alive: **"is the model *still* good, right now,
and how would I know if it weren't?"** Because ML systems don't fail loudly --
they rot quietly.

---

## The problem: failures that don't raise exceptions

A crashed service pages you in seconds. These don't page anyone:

- A feature pipeline stalls; the model serves **week-old features**. No error.
- An item goes **out of stock** but keeps getting recommended. No error.
- A category **goes viral**; the model's learned prior is now wrong. No error.
- A data source silently starts sending **nulls** in a key column. No error.

Each one degrades recommendations for days before anyone notices a revenue dip.
Monitoring is the discipline of converting these silent rots into **loud,
countable signals** -- ideally ones that block a bad deploy before it ships.

## The design: three layers + a gate

`phase4/monitors.py` implements the design doc's three layers as small check
functions, each returning a `CheckResult` (passed / value / threshold / detail).
A `Monitor` aggregates them into one **pipeline gate**: PASS or FAIL.

| Layer | Watches | Example checks |
|---|---|---|
| **1. Data health** | the inputs | row-count volume, null rate, feature drift (PSI) |
| **2. Model health** | the model's behavior | fallback rate, recommendation diversity, calibration |
| **3. Business** | the outcome | add-to-cart rate, purchase rate (from the event stream) |

The philosophy: **catch problems as early in the chain as possible.** A data-
health failure is cheaper to fix than a business-metric failure you discover a
week later in a revenue report.

### The star of the show: drift detection with PSI

The **Population Stability Index** is the industry-standard drift metric. It bins
a reference distribution by quantiles, then measures how much the current
distribution's mass has moved between those bins:

```
PSI = sum over bins of  (current% - reference%) * ln(current% / reference%)

PSI < 0.1   stable
0.1 - 0.2   moderate shift  -> investigate
PSI >= 0.2  major shift     -> likely a problem
```

We compute PSI on the `item_pop` feature distribution between a 7-day window
*before* the train/serve cutoff (reference) and a 7-day window *after* it
(current). Equal-length windows keep it apples-to-apples.

### Calibration: ranking well isn't the same as being right

A ranker can order items perfectly yet output nonsense probabilities (e.g. "0.9"
for things that convert 10% of the time). **Expected Calibration Error** bins
predictions by score and measures the average gap between predicted probability
and observed frequency. It matters because business logic (bidding, thresholds,
expected-value calculations) trusts the *number*, not just the order.

## Code tour

| File | Job |
|---|---|
| `monitors.py` | Pure numeric core (`population_stability_index`, `kl_divergence`, `expected_calibration_error`) + `check_*` functions + the `Monitor` gate. |
| `run.py` | Fits the Phase 2/3 stack, runs all three layers on real data, prints the gate. |

The numeric functions are pure and heavily unit-tested -- a wrong drift metric is
worse than no metric (false alarms train people to ignore alerts; silent misses
defeat the point).

---

## Results

`cd phase4 && python run.py`:

**Layer 1 -- data health**

```
[PASS] row_count_ratio      value=1.0025 (thr 0.8000) -- 124,595 vs baseline 124,283
[PASS] null_rate            value=0.0000 (thr 0.0100) -- 0/124,595 null
[FAIL] feature_drift_psi    value=0.4681 (thr 0.2000) -- major shift
GATE: FAIL
```

**Layer 2 -- model health**

```
[PASS] fallback_rate            value=0.0000  (thr 0.05) -- 0/2,000 requests degraded
[PASS] recommendation_diversity value=18.87   (thr 3.0)  -- avg unique categories in top-20
[PASS] calibration_ece          value=0.0709  (thr 0.10) -- expected calibration error
GATE: PASS
```

**Layer 3 -- business metrics**

```
add_to_cart rate : 0.0260  (14,347 / 551,221 events)
purchase rate    : 0.0083  ( 4,593 / 551,221 events)
```

**Overall gate: FAIL** (driven by the drift check).

### Reading the numbers like an engineer

- **The drift check fired -- PSI 0.47, a major shift.** This is the most important
  result in the whole repo, because it *connects back to Phase 2*. Remember the
  training-serving skew we measured (features inflated 2-2.8x)? That skew exists
  precisely **because item popularity drifts over time.** Phase 2 measured the
  cause; Phase 4's monitor detects the symptom and **blocks the pipeline.** Same
  phenomenon, caught two different ways -- that's a healthy system.
- **This FAIL is "correct."** On a static historical dataset, train-vs-serve drift
  is expected. In production you'd respond by retraining on fresh data (Phase 6's
  freshness story), not by silencing the alarm. The lesson: **a red gate is
  information, not an insult.**
- **The model is well-behaved where it counts.** 0% fallback (the ranker never
  errored over 2,000 requests), 18.9 categories of diversity (no popularity
  tunnel-vision), and an ECE of 0.07 (the LR's probabilities are trustworthy --
  another dividend of choosing a simple, calibratable model in Phase 2).

> **Why block the deploy on drift?** Because shipping a new model on top of
> drifted data bakes the drift into the next generation. The gate forces a human
> decision: retrain, or acknowledge and proceed. Silent auto-deploy is how skew
> compounds.

---

## What Phase 4 taught us

1. **The scary failures are silent.** None of stale features, OOS items, or nulls
   throws an exception. If you're only watching for crashes, you're blind to how
   ML systems actually fail.
2. **Drift is measurable, and it ties the whole system together.** PSI 0.47 in
   Phase 4 is the same popularity shift that caused the 2-2.8x skew in Phase 2.
   Measuring a phenomenon two ways and having them agree is how you build trust.
3. **A gate turns metrics into decisions.** A dashboard nobody reads is theater.
   A check that *blocks a deploy* is a control. Wire the gate into CI/Airflow.
4. **Calibration is a first-class metric.** A model that ranks well but lies about
   probabilities breaks every downstream system that trusts the number.

Next up: **Phase 5 (A/B testing)** uses the `model_version` tag from Phase 3 to
compare two models on live traffic -- the only honest way to know if v2 actually
beats v1. See the [roadmap](../README.md#roadmap).
