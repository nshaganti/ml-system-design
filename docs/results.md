# Results & Performance Progression

This is the consolidated scoreboard for Phases 0-6. Read the **big caveat** first,
because the single most important thing about these numbers is that **most of them
are not directly comparable to each other** -- and pretending otherwise is exactly
the kind of self-deception this whole project is about avoiding.

---

## The big caveat: complexity did not monotonically improve accuracy

It is tempting to draw one line -- "we added complexity, the metric went up." For
this system that line would be a **lie**, for three reasons:

1. **Different metrics.** Phases 0-2 report `Recall@20`; Phases 5-6 report `hit@20`
   (a per-user binary). These measure different things.
2. **Different denominators / candidate sets.** Phase 0-1 recall is over the
   *full catalog*. Phase 2 recall is over a *500-item popularity pool*. A smaller
   candidate set inflates recall -- so Phase 2's 0.027 is **not** better than
   Phase 0's 0.031; they're measured against different universes.
3. **Different populations.** Phase 5/6 evaluate subsets (test purchasers,
   multi-event users), not the same user set as Phase 0/1.

So the tables below are **grouped by what is actually comparable.** Cross-group
comparisons are explicitly flagged as invalid.

> **The honest through-line:** raw offline accuracy did *not* climb steadily with
> complexity. Phase 1 (our first ML model) *regressed* against the Phase 0
> heuristic. Phases 3-5 added production capability (serving, monitoring,
> experimentation) with essentially **no accuracy change** -- that was the point.
> The one large, statistically-significant accuracy win came in Phase 6 from
> **fresher data, not a more complex model.**

---

## Group A -- Full-catalog retrieval quality (Phase 0 vs Phase 1)

*Comparable.* Same metric, same full-catalog candidate universe, same test users,
same temporal split.

| Metric | Phase 0 (heuristic) | Phase 1 (two-tower) | Change |
|---|---|---|---|
| Recall@20 | **0.0310** | 0.0228 | **-26%** (regression) |
| Warm-user recall | **0.0474** | 0.0190 | -60% |
| Cold-start recall | 0.0244 | 0.0244 | tie (Phase 1 falls back to heuristic) |
| Catalog coverage | 0.0140 | 0.0118 | -16% |

**Verdict:** the learned model lost to the heuristic. A strong popularity +
category baseline is genuinely hard to beat with a pure ID-embedding two-tower on
sparse histories. See [`phase1.md`](phase1.md) for the debugging story.

---

## Group B -- Reranking a fixed candidate pool (Phase 2)

*Comparable within the group only.* Both rankers reorder the **same 500-item
popularity pool**; the only difference is the ranker. **Not comparable to Group A**
(different, much smaller candidate universe).

| Metric | Popularity order | LR ranker | Change |
|---|---|---|---|
| Recall@20 | 0.0260 | **0.0270** | +3.8% |
| NDCG@20 | 0.0181 | **0.0185** | +2.0% |

Ranker weights (interpretable, the reason we chose LR): `item_pop=3.08`,
`user_cat_affinity=2.93`, `user_pop=-0.35`.

**Verdict:** the ranker adds a small, real personalization lift over raw
popularity -- but note even 0.0270 is *below* Phase 0's 0.0310, because the
popularity pool is a weaker candidate source than the heuristic's category-aware
selection. Reranking can't recover relevant items the candidate stage never
surfaced.

---

## Group C -- Per-user hit@20 (Phases 5 & 6)

*Comparable within the group.* Metric = "did any of the user's actual later
purchases land in the top-20?" (binary per user). **Not comparable to Groups A/B**
(different metric and populations).

| Experiment | Arm A | Arm B | Lift | Significance |
|---|---|---|---|---|
| **Phase 5:** popularity vs LR ranker | 0.0401 (popularity) | 0.0385 (LR) | -4.0% | p=0.84 -- **not significant** |
| **Phase 6:** frozen vs fresh features | 0.0397 (frozen batch) | **0.0625 (fresh stream)** | **+57%** | p=0.004 -- **significant** |

**Verdict:** the two experiments use the *same statistics* and reach *opposite*
conclusions. Refining the ranker (Phase 5) was inconclusive noise. Refining data
**freshness** (Phase 6) was a large, significant win -- **with the model
byte-for-byte unchanged.**

> Sanity check: Phase 5's popularity arm (0.0401) and Phase 6's frozen arm
> (0.0397) are ~equal, as they should be -- both are the batch/popularity setup
> measured the same way. That consistency is a small confidence signal that the
> harness is behaving.

---

## What each phase actually bought

Accuracy is only one axis. Most phases traded in a *different* currency -- and
that's the real story of going from prototype to production.

| Phase | Primary currency | Headline result | Accuracy delta |
|---|---|---|---|
| 0 Heuristic | a **baseline** | Recall@20 = 0.031 | (defines zero) |
| 1 Two-tower | a **pipeline** (MLflow, index, eval) | regressed on accuracy | **down** |
| 2 Feature store + LR | **correctness** (skew measured 2-2.8x) + interpretability | small pool-rerank lift | ~flat |
| 3 Serving | **latency & robustness** (p50 6ms, fallback) | no accuracy change | flat |
| 4 Monitoring | **trust** (PSI 0.47 drift caught, gate) | no accuracy change | flat |
| 5 A/B testing | **honesty** (don't ship noise) | inconclusive | flat |
| 6 Freshness | **fresh data** | +57% hit@20, significant | **up (big)** |

**The lesson in one line:** we spent six phases adding complexity, and the complexity
mostly bought *robustness, correctness, and trust* -- not raw accuracy. The one
accuracy breakthrough came from feeding the same model **fresher data**. In
production ML, that is the rule, not the exception (Rules 8 & the "features > models"
mantra).

---

## Reproducing these numbers

Every number here is produced by a `run.py` on the real Retail Rocket dataset
(seed 42, 80/20 temporal split). To regenerate:

```bash
cd phase0 && python run.py     # Group A: heuristic (writes results.json)
cd phase1 && python run.py     # Group A: two-tower vs Phase 0
cd phase2 && python run.py     # Group B: LR ranker vs popularity + skew
cd phase5 && python run.py     # Group C: A/B replay (inconclusive)
cd phase6 && python run.py     # Group C: frozen vs fresh (significant)
```

Small run-to-run variation is expected (negative sampling, user subsampling); the
*conclusions* -- the regression, the inconclusive ranker test, the significant
freshness win -- are stable.
