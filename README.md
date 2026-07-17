# ML System Design: Real-Time Recommendation Engine

A hands-on, phase-by-phase build of a production-style product recommendation
system, following [Google's Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml).
It uses the [Retail Rocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
as the vehicle and grows from a no-ML heuristic baseline into a two-tower
retrieval model -- with the same evaluation harness used at every step so
comparisons stay honest.

The design narrative lives in two documents:

- [`ml-system-design-recommendation-engine.md`](ml-system-design-recommendation-engine.md) -- the end-to-end architecture (Phases 0-6).
- [`ml-system-design-methodology-and-pitfalls.md`](ml-system-design-methodology-and-pitfalls.md) -- the methodology and common traps.

## Learning walkthroughs (start here)

The `docs/` folder turns this from a code dump into a guided course. Each
walkthrough follows the same arc: **the problem -> the design decision and why ->
a tour of the code -> the actual measured results -> the gotchas that surprised
us.** Written for an ML engineer moving from notebooks to production.

- [`docs/phase0.md`](docs/phase0.md) -- launch without ML; temporal evaluation; the baseline.
- [`docs/phase1.md`](docs/phase1.md) -- two-tower retrieval, and why our ML model *lost* to the heuristic (and what we did about it).
- [`docs/phase2.md`](docs/phase2.md) -- the feature store, training-serving skew measured at ~2x, and the ranker that finally wins.
- [`docs/phase3.md`](docs/phase3.md) -- the serving architecture: the 100ms request path, latency budgets, graceful fallback, and Rule 29 feature logging.
- [`docs/phase4.md`](docs/phase4.md) -- monitoring and drift detection: three health layers, PSI drift caught at 0.47, and a pipeline gate.
- [`docs/phase5.md`](docs/phase5.md) -- A/B testing: sticky assignment, a two-proportion z-test, and why our offline win came back inconclusive.
- [`docs/phase6.md`](docs/phase6.md) -- near-real-time freshness: streaming features (no retrain) for a significant +57% hit@20 lift.
- [`docs/results.md`](docs/results.md) -- the consolidated scoreboard: every phase's numbers, grouped by what's *actually* comparable, with the honest story (complexity bought robustness, freshness bought accuracy).
- [`docs/benchmarking-vs-literature.md`](docs/benchmarking-vs-literature.md) -- how we stack up against the community's actual task (session next-item). Co-visitation scores Recall@20=0.344, beating every learned model here.
- [`docs/dataset-hm.md`](docs/dataset-hm.md) -- swapping Retail Rocket for H&M with zero phase changes: the dataset dispatcher, what's different about H&M, and how to run it.
- [`docs/lessons-learned.md`](docs/lessons-learned.md) -- the greatest-hits cheat sheet of production reflexes.

---

## Repository layout

```
.
├── data/                     # dataset CSVs go here (gitignored, not committed)
├── phase0/                   # Heuristic baseline (no ML)
│   ├── load_data.py          #   load + validate + canonical schema
│   ├── heuristic_ranker.py   #   popularity + category-affinity ranker
│   ├── evaluate.py           #   temporal split, Recall@K, coverage
│   └── run.py                #   phase 0 entry point -> writes results.json
├── phase1/                   # Two-tower candidate generation (first ML model)
│   ├── dataset.py            #   BPR triples, item vocab, user histories
│   ├── two_tower.py          #   model, BPR loss, embedding extraction
│   ├── train.py              #   training loop + MLflow tracking
│   ├── index.py              #   brute-force ANN over item embeddings
│   ├── ranker.py             #   two-tower ranker w/ phase 0 fallback
│   └── run.py                #   phase 1 entry point (compares vs phase 0)
├── phase2/                   # LR ranker + point-in-time feature store
│   ├── feature_store.py      #   point-in-time correct + online + skewed features
│   ├── lr_ranker.py          #   interpretable logistic-regression ranker
│   └── run.py                #   phase 2 entry point (+ skew demonstration)
├── phase3/                   # Serving architecture (the 100ms request path)
│   ├── candidate_generator.py#   Stage 1 behind an interface (popularity impl)
│   ├── service.py            #   request handler: timed stages, fallback, feature log
│   └── run.py                #   phase 3 entry point (latency test + fault injection)
├── phase4/                   # Monitoring & drift detection (Rule 10)
│   ├── monitors.py           #   PSI/KL/ECE numeric core + check framework + gate
│   └── run.py                #   phase 4 entry point (3 health layers + gate)
├── phase5/                   # A/B testing / online experimentation (Rule 16)
│   ├── experiment.py         #   sticky assignment + two-proportion z-test + power
│   └── run.py                #   phase 5 entry point (replay A/B + significance)
├── phase6/                   # Near-real-time freshness (Rule 8)
│   ├── streaming_store.py    #   frozen batch features + live delta layer (no retrain)
│   └── run.py                #   phase 6 entry point (frozen vs fresh experiment)
├── phase7/                   # Session-based co-visitation (community benchmark)
│   ├── covisitation.py       #   sessionize + item-kNN co-visitation recommender
│   └── run.py                #   phase 7 entry point (leave-one-out next-item eval)
├── tests/                    # pytest suite (synthetic data, no CSVs needed)
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

**Default -- Retail Rocket.** Download the three CSVs from
[Kaggle: Retail Rocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
and drop them in `data/`:

```
data/events.csv
data/item_properties_part1.csv
data/item_properties_part2.csv
```

(They total ~1 GB and are gitignored -- never commit them.)

**Optional -- H&M.** The pipeline is dataset-agnostic. To run everything on the
[H&M Personalized Fashion](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
data instead, accept the competition rules, drop `transactions_train.csv`,
`articles.csv`, `customers.csv` into `data/`, and set `DATASET=hm`:

```bash
cd phase0 && DATASET=hm python run.py           # any phase, one env var
cd phase0 && DATASET=hm HM_MAX_ROWS=3000000 python run.py   # cap for small machines
```

See [`docs/dataset-hm.md`](docs/dataset-hm.md) for the migration guide and
what's different about H&M. Adding another dataset = one module in
`phase0/data_sources/` + one line in the dispatcher; no phase code changes.

### 3. Run Phase 0 (heuristic baseline)

```bash
cd phase0
python run.py
```

This loads the data, does a temporal 80/20 split, fits the popularity +
category-affinity ranker, evaluates Recall@20 / coverage, and writes the
baseline metrics to `phase0/results.json`.

### 4. Run Phase 1 (two-tower model)

```bash
cd phase1
python run.py
```

This trains the two-tower model (tracked in MLflow), builds the embedding
index, evaluates with the **same** `recall_at_k` used in Phase 0, and prints a
head-to-head comparison against the Phase 0 baseline it loads from
`results.json`.

Inspect the training runs:

```bash
mlflow ui --port 5000   # then open http://localhost:5000
```

### 5. Run Phase 2 (LR ranker + feature store)

```bash
cd phase2
python run.py
```

This fits a **point-in-time feature store**, trains an interpretable
logistic-regression ranker on leakage-free features, and -- the headline --
**quantifies training-serving skew** by showing how much the naive
"join today's totals" approach inflates historical feature values.

### 6. Run Phase 3 (the serving architecture)

```bash
cd phase3
python run.py
```

This assembles Phases 1-2 into a live recommendation **service** and exercises
it: a single request with a per-stage latency breakdown, a load test reporting
p50/p99 latency against the 100ms budget, fault injection proving graceful
fallback, and the **Rule 29 inference feature log** (the seed of skew-free
next-generation training data).

### 7. Run Phase 4 (monitoring & drift detection)

```bash
cd phase4
python run.py
```

This runs three health layers (data / model / business) over real data and
prints a single **pipeline gate**. The feature-drift check fires (PSI 0.47) --
the same popularity shift Phase 2 measured as skew, now caught as a blocking,
monitorable signal.

### 8. Run Phase 5 (A/B testing)

```bash
cd phase5
python run.py
```

This runs an offline **replay** A/B test (popularity vs the LR ranker) with the
real statistical machinery: sticky/salted assignment, a two-proportion z-test
with confidence intervals, and up-front sample-size planning. Spoiler: the
offline win comes back **inconclusive** -- a lesson in not shipping on noise.

### 9. Run Phase 6 (near-real-time freshness)

```bash
cd phase6
python run.py
```

This streams a user's in-session behavior into the online feature store and shows
how their recommendations change **without any retraining**. The quantified
experiment finds a significant **+57% hit@20 lift** (p=0.004) from freshness
alone -- on this data, fresh features beat a fancier ranker.

### 10. Run Phase 7 (session co-visitation benchmark)

```bash
cd phase7
python run.py
```

This benchmarks us against the community's actual task -- session-based next-item
prediction, leave-one-out. A simple co-visitation model scores **Recall@20=0.344**
(vs 0.008 for popularity), beating every learned model from Phases 1-2 on the task
they should have targeted. The honest gap analysis is in
[`docs/benchmarking-vs-literature.md`](docs/benchmarking-vs-literature.md).

---

## Running the tests

The suite uses small synthetic DataFrames -- no dataset download required.

```bash
pytest tests/ -q
```

CI runs the same suite on every push and pull request (Python 3.9 and 3.11).

---

## Design principles baked in

- **Temporal evaluation only** (Rule 33) -- never a random split; the model
  never sees the future.
- **Same eval harness across phases** -- `TwoTowerRanker` implements the same
  `.recommend()` interface as `HeuristicRanker`, so `recall_at_k` compares them
  apples-to-apples (same users, same seed, same coverage denominator).
- **Heuristic fallback for cold-start** (Rule 28) -- the ML model serves warm
  users; the Phase 0 heuristic covers users with no learnable history.
- **Interpretable-first** (Rules 4, 14) -- start simple, earn complexity.

---

## Roadmap

- [x] Phase 0 -- heuristic baseline
- [x] Phase 1 -- two-tower candidate generation
- [x] Phase 2 -- logistic-regression ranker + point-in-time feature store
- [x] Phase 3 -- serving architecture (100ms request path, fallback, feature logging)
- [x] Phase 4 -- monitoring & drift detection (3 health layers, PSI gate)
- [x] Phase 5 -- A/B testing (sticky assignment, z-test, power analysis)
- [x] Phase 6 -- near-real-time freshness (streaming features, no retrain)

All six phases from the design doc are implemented, tested, and run on the real
Retail Rocket dataset. See the [learning walkthroughs](#learning-walkthroughs-start-here).

**Bonus -- Phase 7 (`phase7/`):** a session-based co-visitation benchmark against
the community's actual task. It scores Recall@20=0.344 (leave-one-out next-item),
beating every learned model in Phases 1-2 -- see
[`docs/benchmarking-vs-literature.md`](docs/benchmarking-vs-literature.md).

See the design doc for the full multi-phase plan.
