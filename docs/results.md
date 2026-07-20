# Results & Performance Progression

This is the consolidated scoreboard for Phases 0-8 on **real KuaiRand-Pure data**
(seed 42, 80/20 temporal split). Read the **big caveat** first, because the single
most important thing about these numbers is that **most of them are not directly
comparable to each other** -- and pretending otherwise is exactly the kind of
self-deception this whole project is about avoiding.

> **How this file stays honest.** Every table between `<!-- AUTOGEN -->` markers is
> regenerated from each phase's `results.json` by `scripts/build_results.py` -- so
> the numbers in the tables are exactly what the code last produced, never
> hand-copied. The surrounding prose is human-written and quotes numbers
> *approximately* (they wobble a little run-to-run from negative sampling and user
> subsampling); trust the tables for the precise values.

---

## The big caveat: read groups, not one trend line

It is tempting to draw one line -- "we added complexity, the metric went up." For
this system that line would be a **lie**, for three reasons:

1. **Different metrics.** Phases 0-2 report `Recall@20`; Phases 5-6 report `hit@20`
   (a per-user binary). These measure different things.
2. **Different denominators / candidate sets.** Phase 0-1 recall is over the
   *full catalog*. Phase 2 reranks a fixed candidate pool. Phase 7 is next-item
   *within a session* -- an easier task entirely.
3. **Different populations.** Phase 5/6 evaluate subsets (test positives,
   multi-event users), not the same user set as Phase 0/1.

So the tables below are **grouped by what is actually comparable.** Cross-group
comparisons are explicitly flagged as invalid.

> **The honest through-line (KuaiRand):** on dense feedback and a small catalog,
> the ML retrieval model **does** beat the heuristic (Phase 1, ~+83% recall). But a
> weak single cross-feature **hurts** (Phase 2), the A/B test honestly rules that
> weak ranker out (Phase 5, significant ~-6%), and in-session freshness barely
> moves (Phase 6). Then **Part II (Phase 8) drops the real bomb: the offline metrics
> everyone trusts were biased by 2x.** Complexity bought robustness and honesty;
> causality-aware evaluation bought *truth*.

---

## Group A -- Full-catalog retrieval quality (Phase 0 vs Phase 1)

*Comparable.* Same metric, same full-catalog candidate universe, same test users,
same temporal split.

<!-- AUTOGEN:groupA -->
| Metric | Phase 0 (heuristic) | Phase 1 (two-tower) | Change |
|---|---|---|---|
| Recall@20 | 0.0682 | **0.1246** | +83% |
| Warm-user recall | 0.0665 | **0.1252** | +88% |
| Cold-start recall | **0.1179** | 0.1058 | -10% |
| Catalog coverage | 0.0744 | **0.1603** | +116% |
<!-- /AUTOGEN:groupA -->

**Verdict:** the learned two-tower **beats** the heuristic on the metrics that
measure learning -- roughly 1.8x recall and 2.2x coverage. (Cold-start recall is a
wash: cold users are served by the *same* heuristic fallback in both systems, so
any gap there is sampling noise, not signal.) This is the *opposite* of what a
sparse e-commerce log tends to show: KuaiRand's feedback is dense (a third of
events are strong) and the catalog is small (~7.5k videos), so ID-embedding
retrieval has enough signal to learn from. See [`phase1.md`](phase1.md).

---

## Group B -- Reranking a fixed candidate pool (Phase 2)

*Comparable within the group only.* Both rankers reorder the **same popularity
pool**; the only difference is the ranker.

<!-- AUTOGEN:groupB -->
| Metric | Popularity order | LR ranker | Change |
|---|---|---|---|
| Recall@20 | **0.0689** | 0.0635 | -8% |
| NDCG@20 | **0.0446** | 0.0386 | -13% |
<!-- /AUTOGEN:groupB -->

**Verdict:** the LR ranker with a single `user_cat_affinity` cross feature
**loses** to raw popularity order. On KuaiRand the category tag is coarse and
engagement is popularity-driven, so that one feature carries little signal -- and a
feature with no signal is dead weight (Rules 17 & 20). This is a genuine,
instructive negative result, not a bug. The point-in-time feature store and the
train/serve skew audit still matter regardless of the lift's sign.

---

## Group C -- Per-user hit@20 (Phases 5 & 6)

*Comparable within the group.* Metric = "did any of the user's actual later
positive actions land in the top-20?" (binary per user). **Not comparable to
Groups A/B** (different metric and populations).

<!-- AUTOGEN:groupC -->
| Experiment | Arm A | Arm B | Lift | Significance |
|---|---|---|---|---|
| **Phase 5:** popularity vs LR ranker | **0.2278 (popularity)** | 0.2134 (LR) | -6.3% | p=0.0327 -- **significant** |
| **Phase 6:** frozen vs fresh features | 0.2343 (frozen batch) | 0.2349 (fresh stream) | +0.2% | p=0.903 -- **not significant** |
<!-- /AUTOGEN:groupC -->

**Verdict:** the A/B test (Phase 5) confirms Group B's finding with proper
statistics -- the weak LR ranker is *significantly worse*, so **do not ship**.
Feature freshness (Phase 6) barely moves and is not significant: short-video
engagement here is less bursty-intent than an e-commerce cart, so a 30-second delta
layer adds little. Same statistical machinery, honest verdicts in both directions.

> Sanity check: Phase 5's popularity arm (0.2276) and Phase 6's frozen arm (0.2343)
> are ~equal, as they should be -- both are the batch/popularity setup measured the
> same way. That consistency is a small confidence signal that the harness behaves.

---

## Group D -- Session-based next-item (Phase 7, the community protocol)

*Comparable within the group.* Leave-one-out next-item within a session. **Not
comparable to Groups A-C** (different, easier task: next *item in session*).

<!-- AUTOGEN:groupD -->
| Metric | Popularity | Co-visitation | Lift |
|---|---|---|---|
| Recall@20 | 0.0500 | **0.0798** | +60% |
| MRR@20 | 0.0128 | **0.0204** | +60% |
| NDCG@20 | 0.0207 | **0.0332** | +60% |
<!-- /AUTOGEN:groupD -->

**Verdict:** a simple, untuned, pure-Python co-visitation model beats popularity by
~60% on the session task -- the session signal is real and cheap to exploit. See
[`benchmarking-vs-literature.md`](benchmarking-vs-literature.md).

---

## Group E -- Off-policy evaluation (Phase 8, Part II)

*Not comparable to anything above -- it grades a **policy's value**, and it grades
the honesty of offline evaluation itself.* Ground truth is computable only because
KuaiRand ships a uniform-random log.

<!-- AUTOGEN:groupE -->
| Estimator | Value | Error vs truth |
|---|---|---|
| **Ground truth** (π × random-log rewards) | 0.261 | -- |
| Naive / Direct Method (biased log) | 0.523 | **100.2%** |
| IPS | 0.244 | 6.4% |
| **SNIPS** | 0.259 | **0.6%** |
| Doubly Robust | 0.277 | 6.1% |
<!-- /AUTOGEN:groupE -->

**Verdict:** the naive offline metric -- the one most teams ship on -- overstates
the target policy's true value by **2x**, purely from confounding. Reweighting the
random log by known propensities recovers the truth (SNIPS to 0.6%). This is the
capstone lesson of the repo: *even after all of Part I's discipline, your offline
number can still be a factor of two wrong.* See [`phase8.md`](phase8.md) and
[`off-policy-evaluation.md`](off-policy-evaluation.md).

---

## Group F -- Off-policy LEARNING (Phase 9, Part II)

*Not comparable to Groups A-E -- it grades a **learned policy's true value**
(V(pi) = sum_a pi(a) r_true(a)), where r_true comes from the random log.* Phase 8
showed evaluating on biased logs lies; Phase 9 shows *learning* on them yields a
genuinely worse policy -- and that learning on unbiased data fixes it.

<!-- AUTOGEN:groupF -->
| Policy (how it was learned) | True value | vs naive |
|---|---|---|
| uniform random (no learning) | 0.179 | -18% |
| pi_naive = softmax(biased-log rates) | 0.217 | -- |
| **pi_learned = softmax(random-log rates)** | **0.407** | **+87%** |
| pi_greedy = argmax(random-log rates) | 0.587 | +170% |
<!-- /AUTOGEN:groupF -->

**Verdict:** a policy *learned* from the unbiased random log has ~**1.9x** the true
value of one learned from the (much larger) biased production log. The data-
efficiency sweep drives the point home: it takes ~100k unbiased rows to overtake a
policy fit on 1.44M biased rows, and unbiased data keeps pulling ahead after that.
**Bias doesn't average out with volume -- only unbiased data fixes it.** (Reward
rates are Bayesian-smoothed so thinly-sampled items can't fake a perfect score.)
See [`phase9.md`](phase9.md).

---

## Group G -- Sequence model vs co-visitation (Phase 10)

*Comparable within the group only -- same leave-one-out session protocol and the
SAME test cases as Group D (Phase 7).* The only thing that changes is the model:
popularity, co-visitation (co-occurrence), and GRU4Rec (a sequence model that reads
session order).

<!-- AUTOGEN:groupG -->
| Metric | Popularity | Co-visitation | GRU4Rec |
|---|---|---|---|
| Recall@20 | 0.0498 | **0.0801** | 0.0691 |
| MRR@20 | 0.0128 | **0.0203** | 0.0172 |
| NDCG@20 | 0.0207 | **0.0332** | 0.0283 |
<!-- /AUTOGEN:groupG -->

**Verdict:** an honest surprise. A basic, CPU-budget GRU4Rec **beats popularity**
(order carries real signal) but **loses to plain co-visitation** by ~15%. On a
small catalog with strong pairwise co-occurrence, the cheap item-kNN is a
remarkably hard baseline; closing the gap would need heavier tuning (negative
sampling, a bigger model, many more epochs) -- and might still not be worth it. This
is Phase 2's lesson again: **the fancier model is not automatically better -- you
measure it.** See [`phase10.md`](phase10.md).

---

## What each phase actually bought

| Phase | Primary currency | Headline result | Accuracy delta |
|---|---|---|---|
| 0 Heuristic | a **baseline** | Recall@20 = 0.068 | (defines zero) |
| 1 Two-tower | a **pipeline** + real retrieval win | +83% recall, +116% coverage | **up** |
| 2 Feature store + LR | **correctness** (skew audit) + interpretability | weak cross feature hurt (-13% NDCG) | down |
| 3 Serving | **latency & robustness** (p50 4.2ms, fallback) | no accuracy change | flat |
| 4 Monitoring | **trust** (drift gate FAIL, as designed) | no accuracy change | flat |
| 5 A/B testing | **honesty** (don't ship the worse ranker) | significant -6% | down (correctly) |
| 6 Freshness | **fresh data** (streamed, no retrain) | +0.2%, not significant | flat |
| 7 Co-visitation | **task framing** (session signal) | +60% on session next-item | up (diff task) |
| 8 OPE | **causal truth** | naive metric was +100% biased | -- |
| 9 Off-policy learning | **causal learning** | unbiased-learned policy +87% true value | up (true value) |
| 10 Sequence model | **order modelling** (GRU4Rec) | beats popularity, LOSES to co-vis by ~15% | up vs pop, down vs covis |

**The lesson in one line:** Part I's complexity bought robustness, correctness, and
honest experimentation; Part II's causal evaluation revealed that the offline
numbers underneath all of it were still 2x biased. In production ML, trustworthy
evaluation is the whole game (Rules 8, 23, 36).

---

## Reproducing these numbers

Every number is produced by a `run.py` on real KuaiRand-Pure (seed 42, 80/20
temporal split). To regenerate:

```bash
cd phase0 && python run.py     # Group A: heuristic baseline (writes results.json)
cd phase1 && python run.py     # Group A: two-tower vs Phase 0
cd phase2 && python run.py     # Group B: LR ranker vs popularity + skew audit
cd phase5 && python run.py     # Group C: A/B replay (significant -9%)
cd phase6 && python run.py     # Group C: frozen vs fresh (not significant)
cd phase7 && python run.py     # Group D: session co-visitation benchmark
cd phase8 && python run.py     # Group E: off-policy evaluation (Part II)
```

Small run-to-run variation is expected (negative sampling, user subsampling); the
*conclusions* -- the Phase 1 win, the weak-feature regression, the significant A/B
loss, the flat freshness result, and the 2x OPE bias -- are stable.
