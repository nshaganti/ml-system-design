# Phase 7 -- Session Co-visitation (Benchmarking Against the Community)

> **Google's Rule 3:** *Choose machine learning over a complex heuristic* -- but
> Rule 4 says start simple. Phase 7 is the reminder that sometimes the *simple*
> thing you skipped is the strong baseline you should have measured first.

Phases 0-6 evaluated one task: next *positive action* over the *full catalog*,
using a user's whole history. The recsys community more often evaluates a
different, easier task: **session-based next-item prediction** -- given the events
so far in a session, predict the next one, leave-one-out. Phase 7 builds the
community-standard method for that task so we can benchmark honestly.

---

## The problem: are we even measuring the right thing?

Quoting someone else's leaderboard number is meaningless if they measured a
different task. Rather than cite unverifiable scores, Phase 7 *implements* the
community protocol on our own data and lets the numbers speak.

## The design: co-visitation (session item-kNN)

Co-visitation is the workhorse baseline of session recommendation, and it's
almost embarrassingly simple:

```
for each session:
    for each pair of items (a, b) within a sliding window:
        covis[a][b] += 1        # they were seen together

recommend(last_item) = top-M items by covis[last_item][*]
```

No training loop, no embeddings, no gradients -- just counting co-occurrences.
`phase7/covisitation.py` sessionizes the event stream (a gap threshold splits
sessions), builds the co-visitation counts on the training split, and exposes a
`.recommend()` matching the same interface every other phase uses.

## Code tour

| File | Job |
|---|---|
| `covisitation.py` | Sessionize, build co-visitation counts (windowed), recommend by neighbor score; popularity backfill for cold items. |
| `run.py` | Leave-one-out next-item eval (Recall@20 / MRR@20 / NDCG@20) vs a popularity baseline. |

---

## Results

`cd phase7 && python run.py` (leave-one-out sessions, 80/20 temporal split,
window=5, top-M=100):

| Metric (session next-item) | Popularity | **Co-visitation** | Lift |
|---|---|---|---|
| Recall@20 | 0.0496 | **0.0797** | +61% |
| MRR@20 | 0.0127 | **0.0203** | +60% |
| NDCG@20 | 0.0206 | **0.0331** | +61% |

### Reading the numbers like an engineer

- **Co-visitation beats popularity by ~60% on the session task.** The in-session
  co-occurrence signal is real, and a pure-Python counter captures it -- no model
  required. This is the "start simple" baseline you measure *before* reaching for
  a sequence model.
- **The lift is modest, not dramatic (+61%, not 40x).** On sparse e-commerce
  sessions co-visitation often dominates everything; on KuaiRand the catalog is
  small and popularity is already strong, so there's less headroom. **The size of
  a method's win is a property of the data**, which is exactly why you benchmark on
  *your* data instead of quoting someone else's.
- **It slots into the funnel for free.** `.recommend()` matches the Phase 3
  candidate-generator interface, so co-visitation can join the candidate union as
  a cheap, orthogonal source.

> **Honest framing:** session next-item is an *easier* target than next-action
> over the full catalog, so these numbers are not comparable to Phases 0-1. That's
> the point -- Phase 7 exists to compare like-for-like against the community
> protocol, not to crown a winner across tasks. See
> [`benchmarking-vs-literature.md`](benchmarking-vs-literature.md).

---

## What Phase 7 taught us

1. **Measure the community's task the community's way.** A number is only
   comparable if the task and protocol match; implement it rather than cite it.
2. **A dumb baseline can be a strong baseline.** Co-visitation, ~40 lines of
   counting, beats popularity by 60% and belongs in every candidate union.
3. **Win size depends on the dataset.** Don't assume a method that dominates one
   dataset will dominate yours; the honest way to know is to run it.

This closes **Part I**. Continue to [`phase8.md`](phase8.md) -- **Part II**, where
we discover the offline metric underneath all of this was 2x biased.
