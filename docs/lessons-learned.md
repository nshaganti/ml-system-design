# Lessons Learned -- From Notebook to Production

A distilled list of the things this project *taught us by doing*, not by reciting.
Each one cost us a debugging session or a surprising result. If you're an ML
engineer moving from prototypes to production, these are the reflexes to build.

The per-phase walkthroughs ([phase0](phase0.md), [phase1](phase1.md),
[phase2](phase2.md)) tell the full stories; this is the cheat sheet.

---

## 1. A model's value depends on the data regime -- measure, don't assume

Our two-tower model **beat** the popularity+category heuristic on KuaiRand
(Recall@20 0.1246 vs 0.0714, +75%) -- because the feedback is dense and the
catalog is small. The *same* model on a sparse e-commerce log typically **loses**
to that heuristic. Neither outcome is guaranteed by the architecture; both are
properties of the data.

**Reflex:** report lift over a real baseline, measured temporally, and never
assume a model helps until you've measured it in *your* data regime. "We used a
DNN" is not a result.

## 2. Training-serving skew is real, invisible, and measurable

Building features the naive way (join *today's* stats onto old events) inflated
our feature values by **2.0-2.4x**. No crash, no error -- just a model trained on
a distribution it will never see at serving time.

**Reflex:** never join "current" feature tables to historical events. Use
point-in-time correct joins (`join_asof` / a feature store). If you can, *measure*
the skew like `phase2/run.py` does -- a number makes it undeniable.

## 3. Point-in-time correctness has a concrete implementation

It's not magic. Build a per-entity timeline of *prior* cumulative counts, then do
a **backward as-of join** at each event's timestamp. That yields "the value as of
strictly before this moment" -- leakage-free. (`phase2/feature_store.py`)

## 4. Expressing personalization is necessary but not sufficient

An LR ranker with only `item_pop` and `user_pop` can't personalize at all --
per user, those features don't vary in a way that reorders items. You need a
**cross feature** (`user_cat_affinity`, varying per user x item) even to *express*
personalization. But on KuaiRand that cross feature was too weak -- category is
coarse and engagement is popularity-driven -- so it actually **hurt** (-12% NDCG),
despite the model assigning it a large positive in-sample weight.

**Reflex:** a cross feature is required to express personalization, but judge it
by *out-of-sample* lift, not by whether the model likes it. A feature can be
expressible and still carry no signal (Rules 17 & 20).

## 5. Interpretable models tell you things black boxes hide

Our LR ranker learned a **negative** weight on `user_pop` (-0.61): highly active
users are pickier per-item. That's a real behavioral insight, free, from reading
a coefficient. Interpretability also let us *diagnose* why a positively-weighted
feature still hurt (lesson 4) -- you'd never see either in a 500-tree ensemble.

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

## 13. The scary failures are silent -- run every monitoring layer

ML systems rot without raising exceptions: stale features, ineligible items, null
columns, shifting priors, halved row counts. On KuaiRand feature drift was
*stable* (PSI 0.024), so the drift alarm correctly stayed quiet -- and two *other*
checks caught the real problems: the serving window had ~half the reference rows,
and the ranker was miscalibrated (ECE 0.115). A monitor with a favorite failure
mode is half-blind; the value is in covering inputs, behavior, and outcomes at
once.

**Reflex:** monitor inputs (drift, volume, nulls), behavior (fallback,
calibration, diversity), and outcomes (business metrics) -- and make at least one
of them a *gate*, not just a dashboard. A red gate is information, not an insult.

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

Phase 2's LR ranker *lost* to popularity offline (-12% NDCG). A proper
two-proportion z-test on live-style traffic confirmed it: **-10%, p=0.0006,
significant**, with the 95% CI entirely below zero. This time offline and online
*agreed*, and the test gave us the statistical confidence to kill the change
decisively. (Offline and online don't always agree -- which is exactly why you run
the test.)

Just as important: this experiment was **adequately powered** (~7,500 users per
arm at a ~23% baseline), and you know that because you compute the sample size
*before* running.

**Reflex:** compute required sample size BEFORE the experiment. Run until you hit
it. Then let the p-value decide. A *significant negative* is the tool working --
it just saved you from shipping a -10% regression. And assignment must be sticky
and salted, or the whole comparison is silently contaminated.

## 16. Fresh data helps only on the right workload -- measure before you assume

A popular maxim is "fresh data beats a better model." On KuaiRand it *didn't*:
streaming one in-session event into the online store -- model byte-for-byte
unchanged -- moved hit@20 by **+0.2% (p=0.91, not significant)**. Short-video
engagement here just isn't bursty-intent enough for a 30-second delta layer to
matter. On an e-commerce cart it can be a huge win. Same machinery, opposite
workload, opposite verdict.

**Reflex:** build the freshness lever (it's cheap and correct), but *measure* its
payoff on your workload rather than assuming it. Separate what changes fast
(features -> stream them) from what changes slow (models -> retrain them), and
reach for online learning only when streaming genuinely can't meet the SLA.

## 17. Benchmark against the right TASK, and measure it yourself

The community's classic task for interaction logs is next-*item in a session*, not
next-action over the full catalog. When we implemented the community-standard
method (co-visitation) and eval (leave-one-out), it beat popularity by **+61%**
(Recall@20 0.0797 vs 0.0496) -- a real but *modest* win, because KuaiRand's small
catalog makes popularity a strong session baseline. On a sparse e-commerce log the
same method often wins by orders of magnitude. Win size is a property of the data.

Also note *how* we answered "how did others do it?": with no web access, instead
of citing (i.e. fabricating) leaderboard numbers, we **built the standard method
and measured it on our own data**.

**Reflex:** confirm you're solving the same task others benchmark before comparing
numbers. When you can't verify an external claim, reproduce it -- don't quote it.
Never fabricate a comparison.

## 18. A canonical schema makes the hardest thing to change (the data) easy

We set up a canonical event/property schema in Phase 0 almost as an afterthought.
Its payoff arrived much later: rebasing the **entire project** onto KuaiRand -- a
completely different domain (short video, not e-commerce) -- touched only a loader
module behind the `DATASET` dispatcher. All eight phases spoke the canonical
schema plus the WEAK/MEDIUM/STRONG signal taxonomy, so none of them knew or cared
which dataset was underneath.

**Reflex:** define a canonical internal schema at the boundary and translate every
external source into it *once*. The alternative -- phases reaching into raw,
dataset-specific columns -- turns a data swap into a full rewrite. Cheap discipline
early, huge optionality later.

## 19. Your offline metric can be biased by 2x -- and only a known logging policy fixes it (Part II)

After all of Part I's discipline -- temporal splits, point-in-time features,
shared denominators, honest A/B tests -- the offline metric everyone ships on was
*still* wrong. Grading a target policy on the biased production log overstated its
true value by **+100%** (0.523 vs a ground truth of 0.261). The fix isn't more
data; it's KuaiRand's **uniform-random exposure log**, whose known propensities
let IPS/SNIPS reweight the estimate back to the truth (SNIPS: 0.6% error).

**Reflex:** confounded logs make offline metrics biased, not just noisy. Serve a
small slice of traffic randomly (or at least *log your propensities*), and use
off-policy estimators (SNIPS, doubly robust) to screen policies before spending
live traffic on them. Watch effective sample size when the new policy drifts from
the logging one.

---

## The meta-lesson

In a notebook, success = a good number on a held-out set. In production, success =
**a number you can trust and explain**, produced by a pipeline that won't silently
lie to you. Part I's work -- temporal splits, point-in-time features, shared
denominators, interpretable models, train/serve parity, honest experimentation --
exists to protect that trust. Part II's off-policy evaluation is the humbling
coda: even a disciplined offline number can be *causally* biased by 2x, and fixing
that takes a known logging policy, not a fancier model. The model is the easy
part; trustworthy, causally-sound evaluation is the whole game.
