# ML System Design: A Recommender-Systems Field Guide

A hands-on, phase-by-phase build of a production-style recommender system,
following [Google's Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml).
It runs on **real data** -- the [KuaiRand-Pure](https://kuairand.com) short-video
interaction logs -- and grows from a no-ML heuristic baseline into two-tower
retrieval, a feature store, a serving path, monitoring, A/B testing, freshness,
session co-visitation, and finally **causality-aware off-policy evaluation** --
with the same evaluation harness at every step so comparisons stay honest.

The project is in two parts:

- **Part I -- Classic recommender (Phases 0-7):** build and ship a recommender the
  way most teams do, with production discipline (temporal eval, feature stores,
  serving SLAs, monitoring, honest A/B).
- **Part II -- Causality-aware evaluation & learning (Phases 8-9):** confront the
  fact that the offline metrics Part I trusts are *biased*, fix the *evaluation*
  with off-policy estimators on KuaiRand's uniform-random exposure log (Phase 8),
  then fix the *learning* itself by training a policy on unbiased data (Phase 9).

The design narrative lives in two documents:

- [`ml-system-design-recommendation-engine.md`](ml-system-design-recommendation-engine.md) -- the end-to-end architecture.
- [`ml-system-design-methodology-and-pitfalls.md`](ml-system-design-methodology-and-pitfalls.md) -- the methodology and common traps.

## Learning walkthroughs (start here)

The `docs/` folder turns this from a code dump into a guided course. Each
walkthrough follows the same arc: **the problem -> the design decision and why ->
a tour of the code -> the actual measured results -> the gotchas that surprised
us.** Written for an ML engineer moving from notebooks to production.

**Part I -- classic recommender:**

- [`docs/phase0.md`](docs/phase0.md) -- launch without ML; temporal evaluation; the baseline (Recall@20 = 0.067).
- [`docs/phase1.md`](docs/phase1.md) -- two-tower retrieval that **beats** the heuristic (+84% recall, +120% coverage) on dense feedback.
- [`docs/phase2.md`](docs/phase2.md) -- the feature store, training-serving skew, and why a weak cross feature made the ranker *lose* to popularity (Rules 17 & 20).
- [`docs/phase3.md`](docs/phase3.md) -- the serving architecture: the 100ms request path (p50 4.1ms), latency budgets, graceful fallback, Rule 29 feature logging.
- [`docs/phase4.md`](docs/phase4.md) -- monitoring and drift detection: three health layers and a blocking pipeline gate.
- [`docs/phase5.md`](docs/phase5.md) -- A/B testing: sticky assignment, a two-proportion z-test, and a *significant* verdict to not ship the weaker ranker.
- [`docs/phase6.md`](docs/phase6.md) -- near-real-time freshness: streaming features with no retrain (and an honest not-significant result here).
- [`docs/phase7.md`](docs/phase7.md) -- session co-visitation: +60% over popularity on the community's next-item task.
- [`docs/phase10.md`](docs/phase10.md) -- a GRU4Rec **sequence model** on the same protocol: beats popularity but **loses to co-visitation** by ~15% (fancier isn't automatically better).
- [`docs/phase12.md`](docs/phase12.md) -- the **two-stage integration** (two-tower retrieval -> LR rerank): the textbook architecture **loses to two-tower alone** (-19%) because the popularity-flavored ranker undoes personalization.
- [`docs/phase13.md`](docs/phase13.md) -- **stage 2 that earns its place**: feed the two-tower similarity score into the LR and the two-stage system **beats two-tower alone** (+3.3%). Two stages beat one only once stage 2 can see stage 1.
- [`docs/phase18.md`](docs/phase18.md) -- **SASRec** (self-attention) on the same protocol: finishes **last, below popularity** (-71% vs co-visitation). A verified-sound model that a small, dense catalog starves -- attention is data-hungry, and complexity must earn its place.

**Part II -- causality-aware evaluation:**

- [`docs/phase8.md`](docs/phase8.md) -- off-policy evaluation: the naive offline metric was **+100% biased**; SNIPS on the random log recovers truth to 0.6%.
- [`docs/phase9.md`](docs/phase9.md) -- off-policy **learning**: a policy learned from the random log has **+87%** the true value of one learned from the (larger) biased log. Bias doesn't average out.
- [`docs/phase11.md`](docs/phase11.md) -- **position-bias debiasing** (controlled sim): naive CTR ranks positions as much as items; IPW recovers the true ranking (Spearman 0.86 -> 0.97), with the examination curve learned from a randomization bucket.
- [`docs/phase14.md`](docs/phase14.md) -- the **explore-and-learn loop** (bandit sim on real KuaiRand rates): where unbiased data comes from. Thompson sampling gets ~48% less regret than greedy and mints a full-support log; exploration is the online source of Part II's causal ground truth.
- [`docs/phase15.md`](docs/phase15.md) -- the **contextual bandit** (LinUCB): personalized exploration. On a world where the best item depends on the user, LinUCB gets ~92% less regret than context-free Thompson -- context helps, and exploration still helps on top of context.
- [`docs/phase16.md`](docs/phase16.md) -- **closing the loop** (capstone): explore -> learn off-policy -> redeploy. Exploration lifts the deployed policy's TRUE value from 41% to ~96% of the skyline; the no-exploration trap stalls. IPS is a near-wash with a well-specified model (Phase 11's bias-variance lesson, one last time).
- [`docs/phase17.md`](docs/phase17.md) -- **real context + safety-gated redeploys**: the loop runs on real per-user contexts and gates every redeploy through off-policy evaluation on an unbiased bucket. The gate helps even clean (67% vs 59% of skyline) and blocks a simulated logging bug (worst deploy 0.61 vs 0.57).
- [`docs/phase19.md`](docs/phase19.md) -- **contextual OPE/OPL**: Part II's estimators go per-user. Context-free OPE is **24% off** for a contextual target; contextual IPS/SNIPS/DR recover the truth (<=0.3%), and a contextual learned policy beats context-free by **+28%** true value.
- [`docs/phase20.md`](docs/phase20.md) -- **joint EM debiasing**: recover the examination curve AND relevance jointly from confounded production logs. EM matches the randomization-based IPW (Spearman 0.923 vs 0.937) with **no randomization bucket** -- debias production traffic in place.
- [`docs/off-policy-evaluation.md`](docs/off-policy-evaluation.md) -- the deep dive on IPS / SNIPS / DM / DR and why known propensities matter.

**Cross-cutting:**

- [`docs/results.md`](docs/results.md) -- the consolidated scoreboard, grouped by what's *actually* comparable.
- [`docs/benchmarking-vs-literature.md`](docs/benchmarking-vs-literature.md) -- how we stack up against the community's session next-item task.
- [`docs/datasets.md`](docs/datasets.md) -- the canonical schema, the WEAK/MEDIUM/STRONG signal taxonomy, and how to plug in any dataset with no phase changes.
- [`docs/rules-of-ml.md`](docs/rules-of-ml.md) -- where each of Google's 43 Rules lives in this codebase.
- [`docs/lessons-learned.md`](docs/lessons-learned.md) -- the greatest-hits cheat sheet of production reflexes.

---

## Repository layout

```
.
├── data/                     # KuaiRand-Pure CSVs go here (gitignored; see data/README.md)
├── phase0/                   # Heuristic baseline (no ML)
│   ├── load_data.py          #   dataset dispatcher + canonical schema + random-log hook
│   ├── signals.py            #   WEAK/MEDIUM/STRONG signal taxonomy (Rule 7)
│   ├── data_sources/         #   pluggable adapters (kuairand.py)
│   ├── heuristic_ranker.py   #   popularity + category-affinity ranker
│   ├── evaluate.py           #   temporal split, Recall@K, coverage
│   └── run.py                #   phase 0 entry point -> writes results.json
├── phase1/                   # Two-tower candidate generation (first ML model)
├── phase2/                   # LR ranker + point-in-time feature store
├── phase3/                   # Serving architecture (the 100ms request path)
├── phase4/                   # Monitoring & drift detection (Rule 10)
├── phase5/                   # A/B testing / online experimentation (Rule 16)
├── phase6/                   # Near-real-time freshness (Rule 8)
├── phase7/                   # Session-based co-visitation (community benchmark)
├── phase8/                   # Off-policy evaluation -- Part II (Rules 23, 36)
├── phase9/                   # Off-policy learning -- Part II (Rules 23, 36)
│   ├── ope.py                #   IPS / SNIPS / Direct Method / Doubly Robust / ESS
│   └── run.py                #   biased-vs-random OPE experiment
├── phase10/                  # GRU4Rec sequence model (Part I extension)
├── phase11/                  # Position-bias debiasing -- Part II (controlled sim)
├── phase12/                  # Two-stage retrieval + ranking integration
├── phase13/                  # Two-tower score as a ranking feature (stage-2 fix)
├── phase14/                  # Explore-and-learn bandit loop -- Part II
├── phase15/                  # Contextual bandit (LinUCB) -- Part II
├── phase16/                  # Closing the loop: explore->learn->redeploy (capstone)
├── phase17/                  # Real context + safety-gated redeploys -- Part II
├── phase18/                  # SASRec sequence model (honest negative) -- Part I
├── phase19/                  # Contextual OPE/OPL (per-user off-policy) -- Part II
├── phase20/                  # Joint EM position-bias debiasing -- Part II
├── scripts/                  # build_results.py + build_report.py (scoreboard/report generators)
├── run_all.py                # run every phase end-to-end, then rebuild docs
├── tests/                    # pytest suite (tiny in-memory frames, no CSVs needed)
├── requirements.txt
└── .github/workflows/ci.yml  # runs pytest on push / PR
```

---

## Quickstart

### 1. Install dependencies

```bash
# Standard (public PyPI)
pip install -r requirements.txt

# Inside Walmart (Artifactory mirror)
pip install -i https://pypi.ci.artifacts.walmart.com/artifactory/api/pypi/external-pypi/simple \
    -r requirements.txt
```

### 2. Get the dataset

Download **KuaiRand-Pure** and place it in `data/KuaiRand-Pure/` (see
[`data/README.md`](data/README.md) for the exact layout). KuaiRand is used because
it ships both a *biased* production log (Part I) and a *uniform-random* exposure
log whose known propensities make honest off-policy evaluation possible (Part II).

```bash
cd phase0 && python run.py                      # kuairand (default)
KUAIRAND_MAX_ROWS=200000 python run.py          # cap rows on small machines
```

See [`docs/datasets.md`](docs/datasets.md) for the canonical schema, the signal
taxonomy, and how to add your own dataset (a module in `phase0/data_sources/` +
one line in the dispatcher; no phase code changes).

> The **test suite needs no download** -- `tests/` runs on tiny in-memory frames.

### 3-10. Run the phases

Run everything end-to-end (each phase writes a `results.json`, then the scoreboard
and HTML report regenerate from those numbers -- see below):

```bash
python run_all.py            # runs phases 0-9, then rebuilds docs/results.md + report
```

Or run them one at a time:

```bash
cd phase0 && python run.py   # heuristic baseline; Recall@20 = 0.068 (writes results.json)
cd phase1 && python run.py   # two-tower; +83% recall vs Phase 0 (MLflow-tracked)
cd phase2 && python run.py   # LR ranker + point-in-time feature store + skew audit
cd phase3 && python run.py   # serving: p50 4.2ms, load test, fault-injection fallback
cd phase4 && python run.py   # monitoring: 3 health layers + a blocking pipeline gate
cd phase5 && python run.py   # A/B replay: significant -6%, do NOT ship the weak ranker
cd phase6 && python run.py   # freshness: streamed features, no retrain (not significant here)
cd phase7 && python run.py   # session co-visitation: +60% over popularity
cd phase10 && python run.py  # GRU4Rec sequence model: beats popularity, loses to co-vis
cd phase11 && python run.py  # PART II -- position debiasing: IPW recovers ranking 0.86 -> 0.97
cd phase12 && python run.py  # two-stage retrieve+rank: honest -- loses to two-tower alone
cd phase13 && python run.py  # + two-tower score as a feature: two-stage now WINS (+3.3%)
cd phase14 && python run.py  # PART II -- bandit loop: Thompson -48% regret + full-support log
cd phase15 && python run.py  # PART II -- contextual bandit (LinUCB): -92% regret, personalized
cd phase16 && python run.py  # PART II -- CAPSTONE: closed loop, exploration 41% -> 96% of skyline
cd phase17 && python run.py  # PART II -- real contexts + OPE safety gate (blocks a poisoned batch)
cd phase18 && python run.py  # SASRec: self-attention finishes LAST here (attention is data-hungry)
cd phase8 && python run.py   # PART II -- OPE: naive metric +100% biased vs SNIPS 0.6%
cd phase9 && python run.py   # PART II -- OPL: policy learned on unbiased data +87% true value
cd phase19 && python run.py  # PART II -- contextual OPE/OPL: context-free eval 24% off; contextual OPL +28%
cd phase20 && python run.py  # PART II -- joint EM debiasing: matches randomization IPW, no random bucket
```

The scoreboard in [`docs/results.md`](docs/results.md) is **auto-generated** from
the per-phase `results.json` files by `python scripts/build_results.py` -- the
numbers in the docs can never silently drift from what the code produced. CI runs
it with `--check` to enforce this.

Inspect Phase 1 training runs with `mlflow ui --port 5000`.

---

## Running the tests

```bash
pytest tests/ -q     # 126 tests, tiny in-memory data, no dataset download required
```

CI runs the same suite on every push and pull request.

---

## Design principles baked in

- **Temporal evaluation only** (Rule 33) -- never a random split; the model never
  sees the future.
- **Same eval harness across phases** -- every ranker implements the same
  `.recommend()` interface, so `recall_at_k` compares them apples-to-apples.
- **Heuristic fallback for cold-start** (Rule 28) -- ML serves warm users; the
  Phase 0 heuristic covers users with no learnable history.
- **Interpretable-first** (Rules 4, 14) -- start simple, earn complexity.
- **Trust your metric, then distrust it** (Rules 23, 36) -- Part I builds evaluation
  discipline; Part II proves the offline metric was still biased and fixes it.

---

## Roadmap

**Part I -- classic recommender:**

- [x] Phase 0 -- heuristic baseline
- [x] Phase 1 -- two-tower candidate generation
- [x] Phase 2 -- logistic-regression ranker + point-in-time feature store
- [x] Phase 3 -- serving architecture (100ms request path, fallback, feature logging)
- [x] Phase 4 -- monitoring & drift detection (3 health layers, gate)
- [x] Phase 5 -- A/B testing (sticky assignment, z-test, power analysis)
- [x] Phase 6 -- near-real-time freshness (streaming features, no retrain)
- [x] Phase 7 -- session-based co-visitation benchmark
- [x] Phase 10 -- GRU4Rec sequence model (beats popularity, loses to co-visitation)
- [x] Phase 12 -- two-stage retrieval + ranking integration (honest: loses to two-tower alone; the ranker must add signal)
- [x] Phase 13 -- two-tower score as a ranking feature (stage 2 earns its place: two-stage beats two-tower alone +3.3%)
- [x] Phase 18 -- SASRec sequence model (self-attention finishes last, below popularity: attention is data-hungry, complexity must earn its place)

**Part II -- causality-aware evaluation:**

- [x] Phase 8 -- off-policy evaluation (IPS / SNIPS / DM / DR on the random log)
- [x] Phase 9 -- off-policy learning (train a policy on unbiased data; +87% true value)
- [x] Phase 11 -- position-bias debiasing (IPW + result-randomization propensities)
- [x] Phase 14 -- explore-and-learn bandit loop (greedy vs epsilon-greedy vs Thompson; exploration mints unbiased data)
- [x] Phase 15 -- contextual bandit (LinUCB): personalized exploration, -92% regret vs context-free
- [x] Phase 16 -- closing the loop (capstone): explore -> learn off-policy -> redeploy; exploration 41% -> 96% of skyline
- [x] Phase 17 -- real context + safety-gated redeploys (OPE gate on an unbiased bucket blocks a poisoned batch)
- [x] Phase 19 -- contextual OPE/OPL (context-free eval 24% off; DR recovers truth; contextual learned policy +28% over context-free)
- [x] Phase 20 -- joint EM position-bias debiasing (Regression-EM matches randomization IPW, Spearman 0.923 vs 0.937, with no randomization bucket)

**Backlog: all cleared.** Every phase originally sketched -- and the four follow-up
gaps (real-context safety gate, SASRec, contextual OPE/OPL, joint EM debiasing) -- is
now built, tested, and documented.

All implemented phases are tested and run on real KuaiRand-Pure data. See the
[learning walkthroughs](#learning-walkthroughs-start-here). For an honest look at
which additions actually paid off, see
[`docs/results.md` -> "Does complexity pay?"](docs/results.md).
