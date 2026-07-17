# Lessons Learned -- From Notebook to Production

A distilled list of the things this project *taught us by doing*, not by reciting.
Each one cost us a debugging session or a surprising result. If you're an ML
engineer moving from prototypes to production, these are the reflexes to build.

The per-phase walkthroughs ([phase0](phase0.md), [phase1](phase1.md),
[phase2](phase2.md)) tell the full stories; this is the cheat sheet.

---

## 1. A neural net does not automatically beat a heuristic

Our two-tower model **lost** to a popularity+category heuristic (Recall@20
0.0228 vs 0.0310), even after popularity-weighted negatives and a train/serve
fix. That's normal. A strong heuristic is a genuinely hard baseline, and beating
it takes side features and tuning, not just "we used embeddings."

**Reflex:** report lift over a real baseline, measured temporally. "We used a
DNN" is not a result.

## 2. Training-serving skew is real, invisible, and measurable

Building features the naive way (join *today's* stats onto old events) inflated
our feature values by **2.0-2.8x**. No crash, no error -- just a model trained on
a distribution it will never see at serving time.

**Reflex:** never join "current" feature tables to historical events. Use
point-in-time correct joins (`join_asof` / a feature store). If you can, *measure*
the skew like `phase2/run.py` does -- a number makes it undeniable.

## 3. Point-in-time correctness has a concrete implementation

It's not magic. Build a per-entity timeline of *prior* cumulative counts, then do
a **backward as-of join** at each event's timestamp. That yields "the value as of
strictly before this moment" -- leakage-free. (`phase2/feature_store.py`)

## 4. Features constant along one axis cannot personalize

An LR ranker with only `item_pop` and `user_pop` **tied popularity exactly** --
because per user, those features don't vary in a way that reorders items. Adding a
**cross feature** (`user_cat_affinity`, which varies per user x item) is what
produced actual personalized lift.

**Reflex:** if your ranker won't beat popularity, check whether your features can
even *express* personalization before blaming the model.

## 5. Interpretable models tell you things black boxes hide

Our LR ranker learned a **negative** weight on `user_pop` (-0.35): highly active
users are pickier per-item. That's a real behavioral insight, free, from reading
a coefficient. You'd never spot it in a 500-tree ensemble.

**Reflex:** start interpretable (Rules 4, 14). Earn complexity only when a simple
model demonstrably plateaus.

## 6. Evaluate temporally, always

A random train/test split lets the model see the future and produces optimistic,
fictional metrics. Sort by time; train on the past; test on the future. We use
one shared temporal split across all three phases.

## 7. A metric is only comparable against a shared denominator

We had a coverage bug: each ranker divided by its *own* catalog size, making
Phase 0 vs Phase 1 coverage meaningless. Fix: one shared `catalog_size` for
everyone.

**Reflex:** when comparing models, eliminate every difference except the model --
same users, same seed, same denominators, same eval function.

## 8. Make comparison trivial with a shared interface

`TwoTowerRanker` implements the *exact* `.recommend()` signature as
`HeuristicRanker`, so one `recall_at_k` function evaluates both unchanged. Cheap
design discipline, huge payoff in trustworthy comparisons.

## 9. Train == serve, down to the arithmetic

The model trained with a raw dot product; the index served L2-normalized cosine.
Same model, different objective at serving time -- silent skew (Rule 32). Aligning
them was a correctness fix regardless of the metric outcome.

## 10. The pipeline is the deliverable, not the model

Rule 4's real point: the model changes weekly; the infra is forever. What Phase 1
actually shipped was MLflow tracking, an index, and a fair eval harness. The model
inside is now easy to swap and improve -- which is the whole game.

## 11. Serving degrades, it doesn't crash

When the ranker throws, the service returns candidate order with
`fallback_used=True` -- never a 500 (Rule 10). Just as important: it *records*
the fallback so monitoring can count it. A silent fallback is its own outage.

**Reflex:** every model call in the request path needs a fallback, and every
fallback needs to be countable.

## 12. Logging served features is the improvement flywheel (Rule 29)

The single habit that separates a system that improves from one that rots: log
the *exact* features passed to the model at inference time. Train the next model
by joining new labels to that log -- not to the current feature store. This makes
skew-free training data a byproduct of serving.

**Reflex:** if you can't answer "what features did the model see for this exact
request?", you can't train v2 safely.

---

## The meta-lesson

In a notebook, success = a good number on a held-out set. In production, success =
**a number you can trust and explain**, produced by a pipeline that won't silently
lie to you. Most of the work -- temporal splits, point-in-time features, shared
denominators, interpretable models, train/serve parity -- exists to protect that
trust. The model is the easy part.
