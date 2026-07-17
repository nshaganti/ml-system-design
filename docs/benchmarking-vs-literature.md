# Benchmarking vs. the Community / Literature

You asked: *since this was a Kaggle dataset, how did the winners approach it, and
where are we underperforming?* This document answers that as honestly as the
tooling here allows.

## Two honesty disclaimers up front

1. **No live web research was possible.** This environment has no open-web search
   or Kaggle-login browser. So this document does **not** cite specific
   leaderboard scores or named "winning solutions" scraped from the internet --
   doing so would mean fabricating numbers, which is the exact sin this whole
   project is built to avoid. Everything below is either (a) a *measured result
   from our own code*, or (b) well-established, widely-known recsys knowledge,
   clearly labelled as such.

2. **RetailRocket is a Kaggle *dataset*, not a *competition*.** It was published
   by the RetailRocket company for research/practice. There is **no official
   leaderboard, prize, or canonical "winners."** There is, however, a large body
   of public notebooks and academic papers that use it, and they converge on a
   clear set of strong approaches. That convergence is what we benchmark against.

---

## The core finding: we were solving a harder, different task -- badly

The community's sweet spot for this data is **session-based next-item
prediction**: given the events so far *in the current session*, predict the next
item, evaluated leave-one-out with Recall@20 / MRR@20 / NDCG@20.

**Phases 0-6 solved a different task:** next-*purchase* over the *full catalog*,
mean-pooling a user's *entire* history. That's legitimate, but it's harder and
not what the literature reports -- so our ~0.03 Recall@20 was never comparable to
the ~0.4-0.6 session numbers you see quoted online.

To make the comparison *real* instead of rhetorical, Phase 7 implements the
community-standard method (co-visitation / session item-kNN) and the
community-standard eval, on our own data:

### Measured result (Phase 7, `cd phase7 && python run.py`)

| Metric (session next-item, leave-one-out) | Popularity | **Co-visitation** | Lift |
|---|---|---|---|
| Recall@20 | 0.0079 | **0.3440** | +4236% |
| MRR@20 | 0.0009 | **0.1541** | +16951% |
| NDCG@20 | 0.0023 | **0.1971** | +8316% |

*(30,000 leave-one-out test sessions; 80/20 temporal split; co-vis window=5,
top-M=100.)*

**Interpretation:**
- **0.344 Recall@20** is a legitimate, respectable session-rec number -- in the
  same ballpark public notebooks report, and ~43x better than popularity on this
  task. A simple, untuned, pure-Python co-visitation model **beats every learned
  model we built in Phases 1-2** *when measured on the task those models should
  have been solving.*
- This is the concrete answer to "where are we underperforming": **we ignored the
  single strongest signal in the data -- the session -- and the single strongest
  method -- co-visitation.** Our own earlier hints (Phase 1's regression, the 39%
  candidate-recall ceiling, Phase 6's +57% freshness win) were all pointing here.

---

## Established strong approaches vs. what we have

These are the approaches that consistently do well on RetailRocket-style
implicit-feedback session data, per widely-known recsys practice and evaluations
(e.g. the session-rec benchmarking work of Ludewig & Jannach, and the standard
Kaggle-notebook toolkit). Not scraped -- general domain knowledge.

| Approach | Why it's strong here | Our status |
|---|---|---|
| **Session co-visitation / item-kNN** | Directly exploits "viewed X -> bought Y" in-session; simple, robust, hard to beat. | **NOW BUILT (Phase 7): Recall@20=0.344** |
| **Session-kNN variants** (V-SkNN, S-SkNN) | Weight neighbor sessions by recency/similarity; often *the* top classical method. | Not built (natural next step) |
| **Sequential neural** (GRU4Rec, SASRec, BERT4Rec) | Model item *order* within a session; strong but often only marginally beat good kNN. | Not built (Phase 1 was mean-pooled, order-blind) |
| **Matrix factorization / ALS** (`implicit`) | Cheap, strong implicit-feedback baseline. | Not built (we skipped a classic baseline) |
| **Two-tower with side features + attention pooling** | Scales retrieval; needs side features + real negatives to shine. | Partial -- ours was ID-only, mean-pooled (the weak version) |
| **Candidate *union* + reranker** | Blend co-vis + trending + category + MF + two-tower, then rank. | Not built -- we used popularity-only candidates (our 39% ceiling) |
| **Leave-one-out session eval (Recall/MRR@20)** | The comparable protocol. | **NOW BUILT (Phase 7)** |

---

## Where we're lacking, prioritized

1. **Candidate generation was the real weakness.** Popularity/ID-two-tower pools
   can't surface the in-session-relevant items co-visitation nails. **Biggest,
   cheapest win -- already demonstrated.**
2. **No session/sequence modelling in the "ML" phases.** Mean-pooling a whole
   history discards order and session boundaries -- the very structure that
   carries the signal.
3. **Missing classic baselines** (ALS, session-kNN) that we should have measured
   before reaching for a two-tower (Rule 4: simple first).
4. **Weak features/negatives in the learned models** (ID-only, uniform-then-
   popularity negatives; no side features, no hard negatives from impressions).

## What this changes about our conclusions

- Our earlier "the two-tower lost to the heuristic" finding stands -- but the
  deeper reason is now clear: **both were the wrong tool for a session dataset.**
  Co-visitation reframes the problem and wins decisively.
- The production system (Phases 3-6) is still sound *as an architecture* -- and
  co-visitation slots straight into it as a candidate source (`recommend()`
  already matches the generator interface pattern from Phase 3).

## Recommended next steps (in ROI order)

1. **Blend co-visitation into the Phase 3 candidate union** and re-run the funnel
   -- this should lift the *whole* system, not just the offline benchmark.
2. **Add V-SkNN** (recency/similarity-weighted session-kNN) and compare on the
   Phase 7 protocol -- usually a further step up from plain co-vis.
3. **Add an ALS baseline** for completeness.
4. **Only then** try a sequence model (SASRec) -- and judge it on the Phase 7
   protocol, not offline recall over the full catalog.

The meta-lesson (consistent with the rest of this repo): **we didn't need a
fancier model; we needed the right task framing and the right candidates.**
Measuring it honestly -- rather than citing someone else's leaderboard -- is what
turned a vague "we're probably underperforming" into a precise, actionable
+4236%.
