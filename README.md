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

Download the three CSVs from
[Kaggle: Retail Rocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
and drop them in `data/`:

```
data/events.csv
data/item_properties_part1.csv
data/item_properties_part2.csv
```

(They total ~1 GB and are gitignored -- never commit them.)

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
- [ ] Phase 2 -- logistic-regression ranker + point-in-time feature store
- [ ] Phase 3+ -- serving, monitoring, A/B testing, near-real-time freshness

See the design doc for the full multi-phase plan.
