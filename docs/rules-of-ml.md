# Google's Rules of ML, mapped to this codebase

This repo is a *teaching* codebase for recommender systems and ML system design.
Its spine is **Google's [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)**
by Martin Zinkevich -- 43 field-tested rules for doing ML engineering well.

Rather than cite them and move on, we **implement** them and trace each rule to the
exact place in the code (or docs) that honors it -- or mark it *deferred* with a
reason. Rule statements below are paraphrased; read the original for the full text.

Status legend: **[done]** honored in code · **[doc]** taught in docs/design ·
**[part II]** implemented in Phase 8 (the causality track) · **[n/a]** not
applicable to an offline teaching repo (explained inline).

> Note: throughout the code, docstrings tagged `Rule N:` point back here. The
> `tests/test_ml_test_score.py` suite turns several of these into runnable checks.

---

## Before machine learning

| # | Rule (paraphrased) | Where in this repo | Status |
|---|---|---|---|
| 1 | Don't be afraid to launch a product without ML. | `phase0/heuristic_ranker.py` -- a non-ML popularity+affinity baseline shipped first. | done |
| 2 | First, design and implement metrics. | `phase0/metrics.py`, `phase0/evaluate.py` -- Recall/NDCG/MRR + temporal split defined before any model. | done |
| 3 | Choose ML over a complex heuristic. | `phase1/` two-tower replaces the heuristic *only once it's the bottleneck*. | done |

## Phase I -- your first pipeline

| # | Rule | Where | Status |
|---|---|---|---|
| 4 | Keep the first model simple; get the infrastructure right. | `phase1/two_tower.py` (small model) + the whole test/eval harness. | done |
| 5 | Test the infrastructure independently from the ML. | `tests/` runs every phase on tiny in-memory frames with no model quality assumptions. | done |
| 6 | Be careful about dropped data when copying pipelines. | `phase2/training.py::build_labelled_features` is the single shared path (no divergent copies). | done |
| 7 | Turn heuristics into features (or handle them externally). | Heuristic event weights + category affinity become features in `phase2/feature_store.py`. | done |

## Monitoring

| # | Rule | Where | Status |
|---|---|---|---|
| 8 | Know your freshness requirements. | `phase6/streaming_store.py` -- streams fast-changing features without retrain; freshness measured. | done |
| 9 | Detect problems before exporting models. | `phase4/monitors.py::Monitor` gate must pass before a model is "served". | done |
| 10 | Watch for silent failures. | `phase0/load_data.py` validation (fail loud) + `phase3/service.py` graceful, *logged* fallback. | done |
| 11 | Give feature columns owners and documentation. | `docs/feature-registry.md` (owner + description per feature); feature-store docstrings. | done |

## Your first objective

| # | Rule | Where | Status |
|---|---|---|---|
| 12 | Don't overthink the objective you directly optimize. | `phase0` uses a single simple signal weighting; no premature multi-objective. | done |
| 13 | Choose a simple, observable, attributable metric first. | Recall@20 / hit@20 -- observable and attributable to the shown list. | done |
| 14 | An interpretable model makes debugging easier. | `phase2/lr_ranker.py` -- logistic regression whose weights we literally read out. | done |
| 15 | Separate spam/quality filtering into a policy layer. | `phase3/service.py` eligibility/business-rule layer, distinct from the ranker. | done |

## Phase II -- feature engineering

| # | Rule | Where | Status |
|---|---|---|---|
| 16 | Plan to launch and iterate. | The phased roadmap itself (`README.md`); each phase ships then improves. | done |
| 17 | Start with directly observed/reported features. | Strong positive signals used directly as labels (see `feature-registry.md`). | done |
| 18 | Explore content features that generalize across contexts. | Category/co-visitation features in `phase2`/`phase7`. | done |
| 19 | Use very specific features when you can. | Item-ID and user-ID embeddings (`phase1`). | done |
| 20 | Combine/modify features in human-understandable ways. | `user_x_category_affinity` cross feature in `phase2/feature_store.py`. | done |
| 21 | Learnable weights scale with data volume. | `phase1/dataset.py` min-interaction threshold before an item gets an embedding. | done |
| 22 | Clean up features you no longer use. | Documented practice in `feature-registry.md` ("retire" column). | doc |

## Human analysis of the system

| # | Rule | Where | Status |
|---|---|---|---|
| 23 | You are not a typical end user. | Held-out temporal eval instead of eyeballing (`phase0/evaluate.py`); Part II (`phase8`) shows even that offline number is 2x biased. | done |
| 24 | Measure the delta between models. | `phase5/experiment.py` A/B replay + `docs/results.md` deltas. | done |
| 25 | Utilitarian performance trumps predictive power. | `docs/results.md` "what each phase *bought*" framing. | doc |
| 26 | Look for patterns in errors; make new features. | Cold-start error analysis motivating history features (`docs/phase1.md`). | doc |
| 27 | Quantify observed undesirable behavior. | `phase4/monitors.py` diversity + calibration (ECE) checks. | done |
| 28 | Identical short-term behavior != identical long-term. | Offline-vs-online caveats (`docs/phase5.md`, `phase6.md`). | doc |

## Training-serving skew

| # | Rule | Where | Status |
|---|---|---|---|
| 29 | Log the features used at serving time; train on those. | `phase3/service.py` inference feature logging. | done |
| 30 | Importance-weight sampled data; don't arbitrarily drop it. | `phase8/ope.py` -- IPS/SNIPS importance weights on the random-exposure log. | part II |
| 31 | Joined tables can change between train and serve. | `phase2/feature_store.py` point-in-time joins (as-of). | done |
| 32 | Re-use code between training and serving. | `phase2/training.py` shared feature builder used by train + eval + serve. | done |
| 33 | Test on data *after* the training cutoff. | `temporal_split` used in every phase. | done |
| 34 | For filtering, trade a little performance for clean data. | Policy layer keeps labels clean (`phase3`). | doc |
| 35 | Beware inherent skew in ranking problems. | `phase2/run.py` skew demonstration. | done |
| 36 | Avoid feedback loops with positional features. | `phase8/` off-policy evaluation breaks the logging-policy feedback loop with known propensities. | part II |
| 37 | Measure training/serving skew. | `phase4/monitors.py` PSI drift + `phase2` skew metric. | done |

## Phase III -- slowed growth, optimization, complex models

| # | Rule | Where | Status |
|---|---|---|---|
| 38 | Don't add features if unaligned objectives are the problem. | Discussed in `docs/results.md` (fix the objective, not the features). | doc |
| 39 | Launch decisions proxy long-term product goals. | The ship-gate discussion (A/B significance) in `docs/phase5.md`. | doc |
| 40 | Keep ensembles simple. | Candidate *union* (`phase7` -> `phase3`) is a simple additive blend, not a stack of stacks. | done |
| 41 | On plateau, add qualitatively new information. | Co-visitation (`phase7`) + freshness (`phase6`) as new signal sources. | done |
| 42 | Diversity/personalization/relevance aren't as tied to popularity as you think. | `phase4` diversity monitor; `docs/benchmarking-vs-literature.md`. | done |
| 43 | Friends are stable across products; interests are not. | Social-graph signal -- **deferred** (KuaiRand has no social-graph data). | n/a |

---

## How to use this map

- **Learning:** read a phase's `docs/phaseN.md`, then come here to see which rules
  it embodies and why.
- **Contributing:** if you add a feature, tag its docstring with the relevant
  `Rule N:` and update this table. If you *violate* a rule deliberately, say so
  and why -- honest debt beats silent debt (see Sculley et al. in
  [`ml-system-design-canon.md`](ml-system-design-canon.md)).
- **Auditing:** run `pytest tests/test_ml_test_score.py` to check the machine-
  verifiable subset.
