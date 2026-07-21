# Phase 12 -- Two-Stage Retrieval + Ranking (the integration)

> **Google's Rule 4 / Rule 14 (again):** *Keep the first model simple; start with an
> interpretable model.* By Phase 12 we have all the parts to build the classic
> two-stage recommender -- so we build it, and then we do the thing everyone skips:
> we *measure whether it was worth it.*

Phases 1 and 2 each ended with the same to-do: *feed the two-tower's candidates into
the LR ranker for the full two-stage architecture.* Every system-design diagram
draws this pipeline -- retrieve a few hundred candidates, then rerank them into a
top-k. Phase 12 finally wires it and grades it against the single-stage baselines on
the exact same users, split, and metric.

## The architecture

```
                 Stage 1: RETRIEVAL                Stage 2: RANKING
  entire        two-tower ANN over user vector    LR over feature-store
  catalog  ->   -> ~200 candidates            ->  features -> top-20
  (7.5k)        (personalized, long-tail)          (reorders the pool)
```

`TwoStageRanker` (in `pipeline.py`) is pure **composition**, not a new model. It
takes any stage-1 generator with `.recommend()`, any stage-2 reranker with
`.rank()`, and the feature store -- and exposes `.recommend()` itself, so
`phase0/evaluate.py`'s `recall_at_k` grades it identically to every other ranker.
That is the whole design: swappable stages, one measurement harness.

## Code tour

| File | Job |
|---|---|
| `phase12/pipeline.py` | `TwoStageRanker` -- composes retrieval + rerank behind the standard `.recommend()` interface, with a backfill safety net. 5 wiring unit tests (fakes, no training). |
| `phase12/run.py` | Assembles the two-tower (P1) + feature store & LR (P2), then grades three rankers head-to-head. |

---

## Results

`cd phase12 && python run.py` (two-tower retrieval, 200-candidate pool, LR rerank,
3000 eval users, same temporal split):

```
  Ranker                     Recall@20   NDCG@20    Coverage
  Two-tower alone (P1)          0.1215    0.0843      0.1564
  Popularity -> LR (P2)         0.0590    0.0372      0.0499
  Two-tower -> LR (P12)         0.0988    0.0697      0.0789
```

### Reading the numbers like an engineer

- **The two-stage system LOSES to two-tower retrieval alone**: -18.7% recall,
  -17% NDCG, and roughly *half* the catalog coverage.
- **But it beats popularity->LR by +67% recall** -- so stage 1 is doing real work;
  swapping a popularity pool for two-tower candidates is a big win.
- **So the problem is stage 2.** The LR's only features are `item_pop`, `user_pop`,
  and `user_cat_affinity` -- all popularity-flavored. Reranking by those *pushes
  popular items back up*, actively undoing the two-tower's personalization and its
  long-tail coverage. The ranker is fighting the retriever.

### Why this is the whole point

A two-stage architecture is not free lift. **It only helps if stage 2 adds signal
that stage 1 didn't already have.** Here the retriever already encodes
personalization; a popularity-shaped ranker on top can only dilute it. This is
exactly Phase 2's lesson (a weak feature hurts) escalated to the *architecture*
level: drawing the right boxes and arrows buys nothing on its own.

### What would actually make stage 2 earn its place

- Feed the **two-tower similarity score itself** in as a ranking feature so the LR
  can *preserve* good retrieval order instead of overwriting it.
- Add features orthogonal to popularity: **recency, brand/price affinity, session
  context**, cross features that vary per (user, item).
- Only then consider upgrading LR -> gradient-boosted / listwise ranker.

---

## What Phase 12 taught us

1. **Integration is a hypothesis, not a guarantee.** "Retrieve then rank" is the
   textbook design; on this data, naively assembled, it regressed. Measure it.
2. **A pipeline is only as strong as its weakest, or most redundant, stage.** Stage
   2 must add *new* signal or it subtracts value.
3. **Composition kept the experiment honest.** Because `TwoStageRanker` reuses the
   real Phase 1 and Phase 2 components behind one eval harness, this is a true
   apples-to-apples verdict -- not a re-implementation that could flatter itself.

Next: put the two-tower score into the ranker's feature set and re-run; wire the
winning configuration into the Phase 3 serving path in place of the popularity
candidate generator.
