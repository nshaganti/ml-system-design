# Phase 8 -- Off-Policy Evaluation (Part II)

> **Google's Rule 36:** *Avoid feedback loops with hidden features.* Your logs are
> the feedback loop. Off-policy evaluation is how you measure a new policy without
> being fooled by the old one that generated your data.

Phases 0-7 are **Part I**: build a classic recommender and grade it with offline
metrics on logged data. Phase 8 is **Part II**, and it starts with an
uncomfortable admission: *those offline metrics are biased.* This phase shows how
biased, and how to fix it -- using the one thing KuaiRand has that almost no other
public dataset does.

---

## The problem: your logs were written by the incumbent

Every interaction you logged was chosen by the recommender already in production.
That policy only showed items it liked, to users it liked. So a reward rate
estimated from that log answers "how good is item X *given the old policy chose to
show it*" -- not "how good is item X." A new policy that shows different things in
different contexts cannot be graded fairly by replaying that log. The bias is not
academic: on real KuaiRand it is a **factor of two**.

## The lever KuaiRand hands us

KuaiRand ships a **uniform-random exposure log**: items shown at random,
independent of context. There the logging policy is *known* -- `beta = 1/N` for
every item -- so every impression has a known propensity. Known propensities are
exactly what unbiased estimators need.

## Code tour

| File | Job |
|---|---|
| `phase8/ope.py` | Pure, unit-tested estimators: IPS, SNIPS, Direct Method, Doubly Robust, importance weights, effective sample size. Arrays in, numbers out (Rule 5). |
| `phase8/run.py` | Derives a target policy from the *biased* log, then grades it four ways against ground truth computed from the *random* log. |

## The estimators

Target policy `pi`, logging policy `beta`, reward `r`, reward model `rhat`:

| Estimator | Formula | Property |
|-----------|---------|----------|
| **IPS** | `mean( (pi/beta)·r )` | unbiased, high variance |
| **SNIPS** | `Σ(w·r)/Σw`, `w=pi/beta` | tiny bias, much lower variance |
| **DM** | `Σ_a pi(a)·rhat(a)` | low variance, only as good as `rhat` |
| **DR** | `DM + mean((pi/beta)·(r−rhat))` | unbiased if **either** piece is right |

Plus **ESS** `= (Σw)²/Σw²`, which warns when `pi` drifts too far from `beta` for
IPS to be trusted.

---

## Results

`cd phase8 && python run.py`. Ground truth is computable only because the random
log exists: `V_true(pi) = Σ_a pi(a)·reward_rate_random(a)`.

```
V_true  (gold: pi x random-log reward)  = 0.26107
V_naive (DM: pi x BIASED-log reward)    = 0.52266   <- confounded

  Estimator                      Value      Rel.err vs truth
  V_true (ground truth)        0.26107                    --
  Naive / Direct Method        0.52266               100.2%   <- biased
  IPS                          0.24430                 6.4%
  SNIPS                        0.25947                 0.6%
  Doubly Robust                0.27698                 6.1%
  Effective sample size: 149,092 of 1,186,059 random-log rows

  95% bootstrap CIs (1000 resamples of the random-log rows):
    IPS   0.24430 [0.24146, 0.24678]
    SNIPS 0.25947 [0.25708, 0.26180]
    DR    0.27698 [0.27438, 0.27953]
    (V_true = 0.26107)
```

### Reading the numbers like an engineer

- **The naive offline metric overstates the policy's value by +100%.** This is the
  metric most teams actually ship on. It is not a little wrong; it is off by 2x,
  purely from confounding. If you picked a policy because "offline recall went up,"
  this is the failure mode you never saw.
- **IPS/SNIPS recover the truth from the random log** because `beta` is known.
  SNIPS lands at 0.6% error -- self-normalization tames IPS's variance almost for
  free.
- **Doubly robust** starts from the biased reward model (which alone is 100% off)
  and corrects it with the known propensities back to 6% -- unbiased if *either*
  the model or the propensities are right.
- **ESS (149k of 1.19M)** says the estimate rests on ~12% effective weight;
  healthy here, but the number to watch when `pi` diverges from `beta`.
- **The CIs teach the subtlest lesson: a tight interval is not a correct one.**
  With 1.19M rows the bootstrap CIs are razor-thin -- and yet IPS's `[0.2415,
  0.2468]` sits *below* the truth and DR's `[0.2744, 0.2795]` sits *above* it;
  only SNIPS's interval covers 0.26107. A bootstrap CI measures **variance** (how
  much the estimate wobbles under resampling), not **bias** (systematic offset from
  truth). IPS is a touch low from weight noise; DR inherits a pull from its biased
  reward model. Report the interval *and* remember it can't rescue a biased
  estimator -- exactly why the random-log ground truth here is so valuable.

---

## What Phase 8 taught us

1. **Offline metrics on biased logs are not just noisy -- they are biased.** No
   amount of data fixes bias; only a known logging policy does.
2. **A random-exposure slice is worth its weight in gold.** Even 1% of traffic
   served randomly buys you honest evaluation of every future policy.
3. **Screen offline, confirm online.** OPE (Phase 8) narrows the field of
   candidate policies cheaply; the A/B test (Phase 5) spends scarce live traffic
   only on the survivors.

See [`off-policy-evaluation.md`](off-policy-evaluation.md) for the deeper
treatment.
