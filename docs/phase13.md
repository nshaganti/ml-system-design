# Phase 13 -- Stage 2 That Earns Its Place (the two-tower score as a feature)

> **Google's Rule 16:** *Plan to launch and iterate.* Phase 12 launched the
> two-stage pipeline and it regressed. Rule 16 says that's fine -- as long as you
> diagnose why and iterate. This is the iteration.

Phase 12 wired two-tower retrieval into the LR ranker and measured an honest loss:
the two-stage system scored *below* two-tower retrieval alone, because the LR's only
features were popularity counts, so reranking shoved popular items up and undid the
retriever's personalization. The diagnosis pointed at one missing ingredient: **the
ranker never saw the retrieval signal.** Phase 13 hands it over.

## The idea: turn the retriever into a feature

The two-tower model already produced, for every candidate, a similarity score
`user_vec . item_vec`. Phase 12 threw that number away after retrieval. Phase 13
feeds it into the LR as a feature (`tt_score`) so the ranker can:

- **preserve** the retriever's ordering when nothing else disagrees, and
- **reorder** only when a popularity/affinity signal genuinely helps.

Same score at training and serving time (Rule 32) -- that is the entire contract of
`TwoTowerScorer`.

## Two small, reusable changes made this clean

- `phase2/lr_ranker.to_matrix` grew a `log_columns` argument. Popularity counts are
  heavy-tailed and want `log1p`; a similarity score can be negative, where `log1p`
  is nonsense. So the LR now logs the columns you name and passes the rest raw --
  fully backward compatible (default = log everything, the Phase 2 behaviour).
- `phase12/pipeline.TwoStageRanker` grew an optional `feature_augmenter` hook. The
  generic pipeline still knows nothing about two-towers; Phase 13 plugs the scorer
  in from outside. **Open for extension, closed for modification.**

## Code tour

| File | Job |
|---|---|
| `phase13/scorer.py` | `TwoTowerScorer`: user vector = mean of history embeddings; score = dot product; cold user / OOV item -> 0.0 (neutral). Provides a training-set scorer and a serving `augment` hook. 5 hand-checkable unit tests. |
| `phase13/run.py` | Retrains the two-tower, adds `tt_score` to the LR features, and grades three rankers on the same harness. |

---

## Results

`cd phase13 && python run.py` (two-tower retrieval, 200-candidate pool, 3000 eval
users, same split):

```
  Ranker                   Recall@20   NDCG@20    Coverage
  Two-tower alone             0.1236    0.0871      0.1544
  TT -> LR (pop feats)        0.1002    0.0688      0.0797   (-18.9% recall)
  TT -> LR + tt_score         0.1277    0.0897      0.1346   (+3.3% recall)
```

### Reading the numbers like an engineer

- **The fix works.** One feature -- the retrieval score -- flips Phase 12's -19%
  regression into a **+3.3% recall / +3% NDCG win over two-tower alone.**
- **The popularity-only ranker still regresses** (-19%), confirming the Phase 12
  diagnosis: the problem was never "two stages," it was "a blind stage 2."
- **Coverage is the honest tax.** The +tt_score ranker (0.135) still sits below
  two-tower alone (0.154): the popularity features tilt it slightly toward the head
  of the catalog. A small accuracy gain bought with a little coverage -- know the
  trade, don't pretend it isn't there.

### Why the win is small (and that's fine)

On a small, dense catalog the two-tower retrieval is already very strong, so there
isn't much headroom for a *linear* ranker to add. +3% is a real, measured win, not a
transformation. The lesson is directional: **the two-stage architecture only pays
once stage 2 can see what stage 1 knows** -- and even then, on this data, the payoff
is modest. Bigger wins would need features orthogonal to retrieval (recency, session
intent, price/brand affinity) and a non-linear ranker.

---

## What Phase 13 taught us

1. **Diagnose, then iterate.** Phase 12's regression wasn't a dead end; it named the
   missing feature, and adding it turned the result around.
2. **Let stages talk.** A ranker that can't see the retriever's opinion will happily
   overwrite it. The similarity score is the cheapest possible bridge.
3. **Small, backward-compatible seams beat rewrites.** A `log_columns` argument and a
   `feature_augmenter` hook were enough -- no existing phase changed behaviour, every
   old test still passes.

The two-stage arc (Phases 12 -> 13) is the whole "earn your complexity" thesis in
miniature: the architecture bought nothing until we gave it the right signal, and
even the fixed version pays honestly and modestly. Next: features orthogonal to
retrieval, a gradient-boosted / listwise ranker, and wiring the winner into the
Phase 3 service.
