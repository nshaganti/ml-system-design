# Off-Policy Evaluation (Part II)

> **The question:** we have a new ranking policy. How good is it — *without*
> shipping it to users? Answering that from logged data is **off-policy
> evaluation** (OPE), and it is a causal-inference problem.

Part I (Phases 0–7) builds a classic recommender and evaluates it with offline
metrics computed on logged data. Part II confronts the uncomfortable truth that
**those offline metrics are biased**, and shows how to fix them.

## Why offline metrics lie

Your logs were collected by the policy already in production. That policy only ever
showed items it already favored, to users it already favored. So when you estimate
"how often do users engage with item X?" from that log, you are measuring
`P(engage | shown, and the old policy chose to show it)` — not `P(engage | shown)`.
The act of being logged is *confounded* with the thing you want to measure.

A new policy that would show *different* items in *different* contexts cannot be
graded fairly by replaying the old log. This is not a small correction: on real
KuaiRand data it is a **2× error** (see below).

## What KuaiRand gives us that others don't

KuaiRand ships a **uniform-random exposure log**: a slice where items were shown at
random, independent of context. There, the logging policy `beta` is *known* and
equal for every item: `beta = 1/N`. Known propensities are the key that unlocks
unbiased estimation.

## The estimators (`phase8/ope.py`)

Let `pi` be the target policy we want to evaluate, `beta` the logging policy, `r`
the observed reward (1 if the user engaged), and `rhat` a learned reward model.

| Estimator | Formula | Property |
|-----------|---------|----------|
| **IPS** (inverse propensity scoring) | `mean( (pi/beta) · r )` | unbiased, high variance |
| **SNIPS** (self-normalized IPS) | `Σ(w·r) / Σw`, `w=pi/beta` | tiny bias, much lower variance |
| **DM** (direct method) | `Σ_a pi(a)·rhat(a)` | low variance, only as good as `rhat` |
| **DR** (doubly robust) | `DM + mean( (pi/beta)·(r−rhat) )` | unbiased if **either** `beta` or `rhat` is right |

Two diagnostics also matter:

- **Full support:** `beta > 0` for every action — guaranteed here by the random
  policy. Without it, IPS is undefined (you can't reweight what was never shown).
- **Effective sample size** `ESS = (Σw)² / Σw²` — collapses when `pi` strays far
  from `beta`, warning you that the IPS estimate has become noise.

## The result on real KuaiRand (`phase8/run.py`)

We derive a target policy `pi` from the **biased** log (as any team would), then
evaluate it four ways. Ground truth is computable *only* because the random log
exists: `V_true(pi) = Σ_a pi(a) · reward_rate_random(a)`.

| Est | Value | Error vs truth |
|-----------|-------|----------------|
| **Ground truth** (π × random-log rewards) | 0.261 | — |
| Naive / Direct Method (biased log) | 0.523 | **+100%**  |
| IPS | 0.244 | 6.4% |
| **SNIPS** | 0.259 | **0.6%**  |
| Doubly Robust | 0.277 | 6.1% |

ESS ≈ 149k of 1.19M random-log rows.

**Read it:** the naive offline metric — the one most teams ship on — overstates
the policy's value by **2×**, purely from confounding. Reweighting the *random* log
by the known propensities recovers the truth; SNIPS nails it to 0.6%. Doubly robust
starts from the biased reward model and corrects it with propensities.

## Rules of ML this touches

- **Rule 1 / Rule 23:** measure the thing you'll act on; a metric you can't trust
  is worse than no metric.
- **Rule 36 (avoid feedback loops):** biased logs *are* the feedback loop; OPE is
  how you see past it.
- **Rule 16 (plan to launch and iterate):** OPE lets you screen candidate policies
  offline before spending precious A/B traffic (Phase 5) on the survivors.

## Where this goes next

- Extend the context-free `pi` to a contextual policy (score per user) and reuse
  the same estimators.
- Counterfactual learning: optimize `pi` directly against the SNIPS/DR objective.
- Feed OPE-screened policies into the Phase 5 A/B test — offline OPE narrows the
  field, the online experiment confirms the winner.
