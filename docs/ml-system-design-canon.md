# The ML System Design Canon (the "bibles")

This codebase teaches recommender systems *and* the discipline of building ML
systems. It leans on a small set of field-defining references. This page
summarizes each, and -- more importantly -- shows **where the repo embodies it**.

If you read only one thing first, read
[Google's Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
and our mapping in [`rules-of-ml.md`](rules-of-ml.md).

---

## 1. Rules of Machine Learning -- Martin Zinkevich (Google)

43 rules on how to do ML engineering. The organizing spine of this repo. Full
per-rule traceability in [`rules-of-ml.md`](rules-of-ml.md).

**Headline ideas we live by:** ship a heuristic first (Rule 1); metrics before
models (Rule 2); keep the first model simple and the *infrastructure* solid
(Rule 4); log serving features to kill training/serving skew (Rule 29); prefer
directly-observed signal (Rule 17).

---

## 2. The ML Test Score -- Breck, Cai, Nielsen, Salib, Sculley (Google)

A 28-point rubric across four areas (Features/Data, Model Development,
Infrastructure, Monitoring) for scoring **production readiness**. We turn the
machine-checkable subset into an actual test:
[`tests/test_ml_test_score.py`](../tests/test_ml_test_score.py).

| Rubric area | How this repo scores | Where |
|---|---|---|
| Features & data | schema validated, point-in-time joins, feature ownership | `phase0/load_data.py`, `phase2/feature_store.py`, `docs/feature-registry.md` |
| Model development | reproducible training, meaningful baseline, interpretable model | `phase0` baseline, `phase2/lr_ranker.py`, fixed seeds |
| ML infrastructure | pipeline tested independently of model, train/serve code reuse | `tests/`, `phase2/training.py` |
| Monitoring | data drift, staleness/freshness, invariants, model quality | `phase4/monitors.py` |

---

## 3. Hidden Technical Debt in Machine Learning Systems -- Sculley et al. (Google)

The catalog of ML-specific debt. We *name* the anti-patterns and show our
countermeasures.

| Debt / anti-pattern | Our countermeasure | Where |
|---|---|---|
| Glue code & pipeline jungles | one shared feature builder; a dataset dispatcher | `phase2/training.py`, `phase0/load_data.py` |
| Training/serving skew | log-and-reuse serving features; point-in-time joins | `phase3/service.py`, `phase2/feature_store.py` |
| CACE ("Changing Anything Changes Everything") | comparable-groups scoreboard + per-phase A/B deltas | `docs/results.md`, `phase5` |
| Undeclared consumers / feedback loops | policy layer separation; exposure-bias work | `phase3/service.py`, **Part II** |
| Unmonitored features | feature registry with owners; drift monitors | `docs/feature-registry.md`, `phase4` |

---

## 4. Designing Machine Learning Systems -- Chip Huyen

System-design framing: iterative development, data/feature engineering, batch vs
stream, evaluation, and deployment/monitoring. Our phase arc mirrors this
lifecycle: baseline -> model -> features/store -> serving -> monitoring -> A/B ->
freshness.

---

## 5. Causality / off-policy canon (Part II)

For the causality-aware track we lean on the counterfactual-learning literature:

- **Counterfactual Reasoning and Learning Systems** -- Bottou et al. (the logging
  -> counterfactual estimation framing).
- **Unbiased Learning-to-Rank / IPS for LTR** -- Joachims, Swaminathan et al.
  (IPS, SNIPS, position-bias correction).
- **Doubly Robust policy evaluation** -- Dudik, Langford, Li.

These land as **Phase 8+** (logging & propensities -> IPS/SNIPS/DR off-policy
evaluation -> unbiased LTR -> contextual bandits -> uplift). The default synthetic
generator emits a **known logging policy** precisely so these estimators can be
validated against ground truth.

---

## How the repo enforces the canon (not just cites it)

1. **Traceability:** every rule/anti-pattern maps to a file (tables above +
   `rules-of-ml.md`). Docstrings carry `Rule N:` tags.
2. **Executable checks:** `tests/test_ml_test_score.py` runs the machine-verifiable
   subset of the ML Test Score.
3. **Honest debt:** where we *defer* a rule, we say so and why (better than silent
   debt -- the Sculley lesson).
4. **Runs anywhere:** a domain-neutral synthetic generator is the default, so every
   principle is demonstrable with zero downloads.
