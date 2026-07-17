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

## 13. The scary failures are silent -- and drift ties the system together

ML systems rot without raising exceptions: stale features, out-of-stock items,
null columns, shifting priors. Our drift monitor caught a **PSI of 0.47** between
train and serve windows -- the *same* popularity shift that caused the 2-2.8x
skew in Phase 2, now surfaced as a deploy-blocking signal. Measuring one
phenomenon two independent ways and having them agree is how you earn trust in a
system.

**Reflex:** monitor inputs (drift), behavior (fallback, calibration, diversity),
and outcomes (business metrics) -- and make at least one of them a *gate*, not
just a dashboard. A red gate is information, not an insult.

## 14. "Works on my machine" is not "works in CI" (a real pitfall we hit)

Our CI went red with `OSError: [Errno 28] No space left on device` -- on the
Python 3.9 job only, during *dependency install*, with **zero test failures**.
The culprit: the default Linux `torch` wheel drags in ~5GB of NVIDIA CUDA
libraries we never use for CPU tests, and it filled the runner disk. It passed
locally because the dev machine already had a slim torch; it passed on 3.11 by a
hair. Phase 4 didn't "break" anything -- it was a latent time bomb in the CI
setup that finally tipped over the disk margin.

The fix: install **CPU-only torch** from the PyTorch CPU wheel index first, add
`pip --no-cache-dir`, and drop the pip cache restore.

**Reflex:** pin lean, environment-appropriate dependencies for CI (CPU wheels,
no GPU libs). And when CI fails, *read the logs before touching code* -- half the
time the failure is the environment, not your change. We diagnosed this by pulling
the Actions logs via the API, not by guessing.

## 15. Offline lift is a hypothesis; the A/B test is the verdict

Our LR ranker beat popularity by +2% NDCG offline (Phase 2). Under a proper
two-proportion z-test it came back **p=0.84, not significant** -- the confidence
interval straddled zero. Offline wins routinely shrink or vanish when tested,
because offline eval can't see how users react to what they were never shown.

Just as important: the experiment was **12x underpowered** (needed ~14.7k users
per arm, had ~1.2k) -- and the sample-size math said so *before* we ran it.

**Reflex:** compute required sample size BEFORE the experiment. Run until you hit
it. Then let the p-value decide -- not the fact that one number is slightly
higher. "Not significant" means *do not ship on this*, not "ship the bigger one."
And assignment must be sticky and salted, or the whole comparison is silently
contaminated.

## 16. Fresh data can beat a better model

The single largest, most clearly-significant win in this entire project was not
an algorithm. Streaming one in-session event into the online feature store --
with the model **byte-for-byte unchanged** -- lifted hit@20 by **+57%**
(p=0.004). The Phase 5 ranker upgrade, by contrast, was inconclusive. Same
statistics, opposite verdict.

**Reflex:** before tuning the model, ask whether your features are *fresh* and
*correct*. Separate what changes fast (features -> stream them) from what changes
slow (models -> retrain them). Reach for online learning only when streaming
features genuinely can't meet the SLA -- it adds real failure modes
(catastrophic forgetting, time-skew) for a usually-small marginal gain.

## 17. Benchmark against the right TASK, and measure it yourself

We spent six phases on next-*purchase* over the full catalog. The community task
for this dataset is next-*item in a session*. When we finally implemented the
community-standard method (co-visitation) and eval (leave-one-out), a simple,
untuned, pure-Python model hit **Recall@20=0.344** -- beating every learned model
we'd built. It was never a model-complexity problem; it was a **task-framing and
candidate-generation** problem.

Also note *how* we answered "how did others do it?": we had no web access, so
instead of citing (i.e. fabricating) leaderboard numbers, we **built the standard
method and measured it on our own data**. A measured +4236% beats a cited number
you can't reproduce.

**Reflex:** confirm you're solving the same task others benchmark before comparing
numbers. And when you can't verify an external claim, reproduce it -- don't quote
it. Never fabricate a comparison.

---

## The meta-lesson

In a notebook, success = a good number on a held-out set. In production, success =
**a number you can trust and explain**, produced by a pipeline that won't silently
lie to you. Most of the work -- temporal splits, point-in-time features, shared
denominators, interpretable models, train/serve parity -- exists to protect that
trust. The model is the easy part.
