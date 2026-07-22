# Benchmarking vs. the Community / Literature

*How does the recommender-systems community approach KuaiRand, and where are we
underperforming?* This document answers that as honestly as the tooling here
allows.

## Two honesty disclaimers up front

1. **No live web research was possible.** This environment has no open-web search.
   So this document does **not** cite specific leaderboard scores scraped from the
   internet -- doing so would mean fabricating numbers, the exact sin this project
   is built to avoid. Everything below is either (a) a *measured result from our
   own code*, or (b) well-established, widely-known recsys knowledge, clearly
   labelled as such.

2. **KuaiRand is a research dataset, not a competition.** It was released by
   Kuaishou (the short-video platform) for research. There is no prize
   leaderboard. What makes it special -- and what the literature actually uses it
   for -- is that it ships a **uniform-random exposure log** alongside the normal
   biased log. That makes it a standard benchmark for **unbiased / causal
   recommendation and off-policy evaluation**, which is precisely why this repo has
   a Part II.

---

## Finding 1: the session signal is real, but modest here

The community's classic sweet spot for interaction logs is **session-based
next-item prediction**: given the events so far in the current session, predict
the next item, evaluated leave-one-out with Recall@20 / MRR@20 / NDCG@20.

Phase 7 implements the community-standard method (co-visitation / session
item-kNN) and eval on our own data:

### Measured result (Phase 7, `cd phase7 && python run.py`)

| Metric (session next-item, leave-one-out) | Popularity | **Co-visitation** | Lift |
|---|---|---|---|
| Recall@20 | 0.0496 | **0.0797** | +61% |
| MRR@20 | 0.0127 | **0.0203** | +60% |
| NDCG@20 | 0.0206 | **0.0331** | +61% |

*(leave-one-out test sessions; 80/20 temporal split; co-vis window=5, top-M=100.)*

**Interpretation:**
- A simple, untuned, pure-Python co-visitation model beats popularity by ~60% on
  the session task. The session co-occurrence signal is real and cheap to exploit.
- But the lift is **modest (+61%)**, not the 40x you see on sparse e-commerce
  session data. Why? KuaiRand's catalog is small (~7.5k videos) and engagement is
  popularity-heavy, so *popularity is already a strong session baseline* -- there's
  less headroom for co-visitation to recover. Dataset shape decides the size of
  the win.
- This is different from a sparse e-commerce log, where co-visitation typically
  dominates everything. The honest takeaway: the *method* is right, but its value
  is dataset-dependent.

## Finding 2: on KuaiRand, the two-tower already wins the retrieval task

Unlike the classic e-commerce story (where an ID two-tower loses to a good
heuristic), Phase 1's two-tower **beats** the heuristic here by +58% recall on the
full-catalog task -- because KuaiRand's feedback is dense enough to learn ID
embeddings well. So on this dataset we are *not* badly underperforming on
retrieval; the model earns its keep.

## Finding 3: the real frontier for KuaiRand is causal evaluation

Because KuaiRand ships a random-exposure log, the literature that features it is
largely about **debiasing and off-policy evaluation** -- exactly Part II. Our
Phase 8 shows the naive offline metric is **+100% biased** and that SNIPS on the
random log recovers truth to 0.6%. That is the community's actual reason to reach
for this dataset, and it's where a KuaiRand project should invest.

---

## Established strong approaches vs. what we have

Widely-known recsys practice for implicit-feedback / short-video data (general
domain knowledge, not scraped):

| Approach | Why it's relevant | Our status |
|---|---|---|
| **Session co-visitation / item-kNN** | Exploits in-session co-occurrence; simple, robust. | **BUILT (Phase 7): +61% over popularity** |
| **Two-tower retrieval (ID + side features)** | Scales retrieval; strong when feedback is dense. | **BUILT (Phase 1), wins +58%; side features are the next step** |
| **Session-kNN variants** (V-SkNN, S-SkNN) | Recency/similarity-weighted neighbors; often top classical method. | Not built (natural next step) |
| **Sequential neural** (GRU4Rec, SASRec, BERT4Rec) | Models item *order* within a session. | Not built (Phase 1 is order-blind mean-pooling) |
| **IPS / DR debiased training & evaluation** | Corrects exposure bias using known/estimated propensities. | **BUILT (Phase 8 evaluation); debiased *training* is the frontier** |
| **Matrix factorization / ALS** | Cheap, strong implicit-feedback baseline. | Not built (skipped a classic baseline) |

---

## Where we're lacking, prioritized

1. **Debiased training, not just debiased evaluation.** Phase 8 measures the bias;
   the next step is training with IPS/DR-weighted objectives so the *policy*
   itself is less confounded. Highest-leverage, most on-theme for KuaiRand.
2. **No sequence modelling in the learned phases.** Mean-pooling discards order
   and session boundaries -- the structure Phase 7 shows carries signal.
3. **ID-only two-tower.** It wins, but adding side features (video tags, duration,
   creator) and hard negatives from impressions should push it further.
4. **Missing classic baselines** (ALS, V-SkNN) worth measuring for completeness
   (Rule 4: simple first).

## What this changes about our conclusions

- The retrieval story is *healthy* on KuaiRand: the two-tower wins, and
  co-visitation adds a real (if modest) session-level lift. We are not solving the
  wrong task badly.
- The genuinely important gap is **causal**: the offline metrics are 2x biased,
  and Part II is where the community-relevant work lives. Co-visitation still slots
  into the Phase 3 candidate union as a cheap, orthogonal source.

## Recommended next steps (in ROI order)

1. **Debiased training (IPS/DR objectives)** and re-evaluate with Phase 8's OPE.
2. **Blend co-visitation into the Phase 3 candidate union** and re-run the funnel.
3. **Add side features + hard negatives** to the two-tower.
4. **Add V-SkNN / ALS baselines** for completeness, judged on the Phase 7 protocol.

The meta-lesson (consistent with the rest of this repo): **measure honestly on
your own data.** That is what turned vague intuitions into precise numbers -- a
+61% session lift, a +58% retrieval win, and a sobering +100% evaluation bias.
