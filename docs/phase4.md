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
- An item becomes **ineligible** (removed/blocked) but keeps getting recommended. No error.
- A category **goes viral**; the model's learned prior is now wrong. No error.
- A data source silently starts sending **nulls** in a key column. No error.
- The upstream window silently ships **half the usual rows**. No error.

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
| **3. Business** | the outcome | engagement rate (MEDIUM), target-action rate (STRONG), from the event stream |

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
[FAIL] row_count_ratio      value=0.5185 (thr 0.8000) -- 122,475 vs baseline 236,208
[PASS] null_rate            value=0.0000 (thr 0.0100) -- 0/122,475 null
[PASS] feature_drift_psi    value=0.0237 (thr 0.2000) -- stable
GATE: FAIL
```

**Layer 2 -- model health**

```
[PASS] fallback_rate            value=0.0000  (thr 0.05) -- 0/2,000 requests degraded
[PASS] recommendation_diversity value=4.02    (thr 3.0)  -- avg unique categories in top-20
[FAIL] calibration_ece          value=0.1151  (thr 0.10) -- expected calibration error
GATE: FAIL
```

**Layer 3 -- business metrics**

```
engagement rate (MEDIUM)   : 0.1292  (37,111 / 287,322 events)
target-action rate (STRONG): 0.3201  (91,985 / 287,322 events)
```

**Overall gate: FAIL** (driven by row-count and calibration).

### Reading the numbers like an engineer

- **The monitor catches what is *actually* wrong -- and it differs by dataset.**
  Feature drift is **stable here (PSI 0.024)**, so the drift alarm correctly stays
  quiet. Instead two other checks fire: the serving window has ~half the reference
  rows (`row_count_ratio` 0.52), and the ranker is **miscalibrated** (ECE 0.115 >
  0.10). A good monitor doesn't have a favorite failure; it surfaces whichever one
  is real.
- **Contrast this with what you might expect.** On a drift-heavy dataset the PSI
  check would be the star; here it's a non-event and the *volume* and
  *calibration* checks earn their keep. That's exactly why you run all three
  layers instead of betting on one metric.
- **This FAIL is "correct."** A row-count halving and a calibration slip are
  genuine reasons to block a deploy and make a human look. The lesson: **a red
  gate is information, not an insult.**
- **Where the model is well-behaved:** 0% fallback (the ranker never errored over
  2,000 requests) and 4.0 categories of diversity in the top-20 (KuaiRand's tags
  are coarse, so 4 distinct categories is healthy spread, not tunnel vision).

> **Why block the deploy?** Because shipping on top of anomalous data or a
> miscalibrated model bakes the problem into the next generation. The gate forces
> a human decision rather than a silent auto-deploy.

---

## What Phase 4 taught us

1. **The scary failures are silent.** None of stale features, OOS items, or nulls
   throws an exception. If you're only watching for crashes, you're blind to how
   ML systems actually fail.
2. **Run every layer -- don't bet on one metric.** Here PSI was quiet and the
   *row-count* and *calibration* checks did the catching. On another dataset it's
   the reverse. A monitor with a favorite failure mode is half-blind.
3. **A gate turns metrics into decisions.** A dashboard nobody reads is theater.
   A check that *blocks a deploy* is a control. Wire the gate into CI/Airflow.
4. **Calibration is a first-class metric.** A model that ranks well but lies about
   probabilities breaks every downstream system that trusts the number.

Next up: **Phase 5 (A/B testing)** uses the `model_version` tag from Phase 3 to
compare two models on live traffic -- the only honest way to know if v2 actually
beats v1. See the [roadmap](../README.md#roadmap).
