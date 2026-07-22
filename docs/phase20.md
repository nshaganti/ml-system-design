# Phase 20 -- Joint Position-Bias + Relevance Estimation via EM

> **Part II's position-bias arc, finished.** Phase 11 removed position bias with
> inverse-propensity weighting -- but it needed the examination curve handed to it,
> learned from a costly result-randomization bucket. Phase 20 estimates the
> examination curve **and** relevance *jointly, from ordinary biased production logs*,
> with no randomization at all.

Randomization -- showing items in random slots to measure how attention decays with
position -- degrades the user experience and costs real revenue, so you often can't
run it. The question Phase 11 left open: can we debias **production** logs *in place*,
where the ranker already sorted items by (a stale notion of) relevance, confounding
position with quality?

## The model and the algorithm

The Position-Based Model, again:

```
P(click | item i, position p) = e_p * r_i        (a click needs BOTH examine AND relevant)
```

Given only clicks, `e_p` and `r_i` are confounded -- but they're jointly identifiable
(up to a global scale) as long as items appear across a **range** of positions.
**Regression-EM** (Wang et al., 2018; the estimation heart of the Dual Learning
Algorithm, Ai et al., 2018) recovers them by alternating:

- **E-step** -- for each impression, infer the posterior of the latent
  examine/relevant bits given the click and current estimates. A click forces both
  bits to 1; a non-click splits probability across "examined but irrelevant,"
  "relevant but not examined," and "neither":
  `P(E=1 | c=0) = e_p(1-r_i)/(1-e_p r_i)`, `P(R=1 | c=0) = (1-e_p)r_i/(1-e_p r_i)`.
- **M-step** -- re-estimate `e_p` as the mean posterior examination per position and
  `r_i` as the mean posterior relevance per item. Anchor `e_0 = 1` to fix the scale.

`phase20/joint_em.py` is the pure-NumPy core (reuses Phase 11's PBM helpers, DRY).

## Results

`cd phase20 && python run.py` (60 items, 10 positions, 20k CONFOUNDED production
sessions ordered by a *stale* ranker, 8 worlds; a 4k randomization bucket exists only
to feed the Phase 11 baseline):

```
    Method                        Spearman   Top-10 recall
    Naive CTR (confounded)           0.588             31%
    IPW, TRUE exam (oracle)          0.941             74%
    IPW, randomized (Phase 11)       0.937             76%
    Joint EM (Phase 20)              0.923             74%
```
*EM examination-curve error: 0.007 MAE | log-likelihood monotone: yes.*

### Reading the numbers

- **Naive CTR is badly confounded (0.588).** Because the stale ranker put items at
  positions only partly related to their true relevance, plain click-through rate
  ranks *slots* as much as *items* -- it recovers barely half the ordering.
- **IPW fixes it if you know `e_p` (0.937-0.941).** Phase 11 got `e_p` from a
  randomization bucket; the oracle version is handed the true curve. Both recover the
  ranking well.
- **Joint EM matches them (0.923) with NO randomization.** It backs out the
  examination curve (0.007 MAE) and relevance *together* from the biased log itself,
  landing on top of the randomization-based IPW and just shy of the oracle.
- **EM behaves.** The observed-data log-likelihood is monotone non-decreasing every
  iteration, exactly as EM guarantees -- a useful correctness signal in production.

The one gotcha worth internalizing: **EM converges at a linear rate.** An early run
with only 60 iterations left it undertrained (Spearman ~0.72); it needed ~300
iterations to reach 0.92. Watch the log-likelihood plateau, don't guess the iteration
count.

## What Phase 20 taught us

1. **You can debias production logs in place.** Joint EM removes the dependence on a
   revenue-costing randomization bucket.
2. **Latent-variable EM is the right tool for confounded clicks.** Model the
   examine/relevant split explicitly and alternate; the log-likelihood guarantee keeps
   you honest.
3. **The output is a training label.** The recovered per-item `r_i` are the debiased
   labels you'd feed the Phase 2 ranker -- closing the loop from "clicks lie" (Phase
   11) to "here's the unbiased signal to train on."

This is the final phase of the project. Together with Phases 8, 9, 11, and 19 it
completes Part II's thesis: **your logs are biased in knowable ways -- name the bias,
model it, and recover the truth -- rather than training on the convenient signal and
hoping.**
