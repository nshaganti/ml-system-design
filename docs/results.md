# Results & Performance Progression

This is the consolidated scoreboard for Phases 0-8 on **real KuaiRand-Pure data**
(seed 42, 80/20 temporal split). Read the **big caveat** first, because the single
most important thing about these numbers is that **most of them are not directly
comparable to each other** -- and pretending otherwise is exactly the kind of
self-deception this whole project is about avoiding.

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
> the ML retrieval model **does** beat the heuristic (Phase 1, +84%). But a weak
> single cross-feature **hurts** (Phase 2), the A/B test honestly rules that weak
> ranker out (Phase 5, significant -9%), and in-session freshness barely moves
> (Phase 6). Then **Part II (Phase 8) drops the real bomb: the offline metrics
> everyone trusts were biased by 2x.** Complexity bought robustness and honesty;
> causality-aware evaluation bought *truth*.

---

## Group A -- Full-catalog retrieval quality (Phase 0 vs Phase 1)

*Comparable.* Same metric, same full-catalog candidate universe, same test users,
same temporal split.

| Metric | Phase 0 (heuristic) | Phase 1 (two-tower) | Change |
|---|---|---|---|
| Recall@20 | 0.0670 | **0.1231** | **+84%** |
| Warm-user recall | 0.0656 | **0.1231** | +88% |
| Cold-start recall | 0.1175 | **0.1213** | +3% |
| Catalog coverage | 0.0714 | **0.1570** | +120% |

**Verdict:** the learned two-tower **beats** the heuristic on every axis -- roughly
1.8x recall and 2.2x coverage. This is the *opposite* of what a sparse e-commerce
log tends to show: KuaiRand's feedback is dense (a third of events are strong) and
the catalog is small (~7.5k videos), so ID-embedding retrieval has enough signal to
learn from. See [`phase1.md`](phase1.md).

---

## Group B -- Reranking a fixed candidate pool (Phase 2)

*Comparable within the group only.* Both rankers reorder the **same popularity
pool**; the only difference is the ranker.

| Metric | Popularity order | LR ranker | Change |
|---|---|---|---|
| Recall@20 | **0.0730** | 0.0657 | -10% |
| NDCG@20 | **0.0439** | 0.0380 | -13% |

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

| Experiment | Arm A | Arm B | Lift | Significance |
|---|---|---|---|---|
| **Phase 5:** popularity vs LR ranker | **0.2276 (popularity)** | 0.2072 (LR) | -9.0% | p=0.0024 -- **significant** |
| **Phase 6:** frozen vs fresh features | 0.2343 (frozen batch) | 0.2348 (fresh stream) | +0.2% | p=0.913 -- **not significant** |

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

| Metric | Popularity | Co-visitation | Lift |
|---|---|---|---|
| Recall@20 | 0.0496 | **0.0797** | +61% |
| MRR@20 | 0.0127 | **0.0203** | +60% |
| NDCG@20 | 0.0206 | **0.0331** | +61% |

**Verdict:** a simple, untuned, pure-Python co-visitation model beats popularity by
~60% on the session task -- the session signal is real and cheap to exploit. See
[`benchmarking-vs-literature.md`](benchmarking-vs-literature.md).

---

## Group E -- Off-policy evaluation (Phase 8, Part II)

*Not comparable to anything above -- it grades a **policy's value**, and it grades
the honesty of offline evaluation itself.* Ground truth is computable only because
KuaiRand ships a uniform-random log.

| Estimator | Value | Error vs truth |
|---|---|---|
| **Ground truth** (π × random-log rewards) | 0.261 | -- |
| Naive / Direct Method (biased log) | 0.523 | **+100%** |
| IPS | 0.244 | 6.4% |
| **SNIPS** | 0.259 | **0.6%** |
| Doubly Robust | 0.277 | 6.1% |

**Verdict:** the naive offline metric -- the one most teams ship on -- overstates
the target policy's true value by **2x**, purely from confounding. Reweighting the
random log by known propensities recovers the truth (SNIPS to 0.6%). This is the
capstone lesson of the repo: *even after all of Part I's discipline, your offline
number can still be a factor of two wrong.* See [`phase8.md`](phase8.md) and
[`off-policy-evaluation.md`](off-policy-evaluation.md).

---

## What each phase actually bought

| Phase | Primary currency | Headline result | Accuracy delta |
|---|---|---|---|
| 0 Heuristic | a **baseline** | Recall@20 = 0.067 | (defines zero) |
| 1 Two-tower | a **pipeline** + real retrieval win | +84% recall, +120% coverage | **up** |
| 2 Feature store + LR | **correctness** (skew audit) + interpretability | weak cross feature hurt (-13% NDCG) | down |
| 3 Serving | **latency & robustness** (p50 4.1ms, fallback) | no accuracy change | flat |
| 4 Monitoring | **trust** (drift gate FAIL, as designed) | no accuracy change | flat |
| 5 A/B testing | **honesty** (don't ship the worse ranker) | significant -9% | down (correctly) |
| 6 Freshness | **fresh data** (streamed, no retrain) | +0.2%, not significant | flat |
| 7 Co-visitation | **task framing** (session signal) | +61% on session next-item | up (diff task) |
| 8 OPE | **causal truth** | naive metric was +100% biased | -- |

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
