# Phase 9 -- Off-Policy Learning (Part II)

> **Google's Rule 23:** *You may not be a typical end user.* And your logs may not
> reflect a typical policy. Phase 8 proved you can't *evaluate* fairly on biased
> logs. Phase 9 proves you can't *learn* fairly on them either -- and shows the fix.

Phase 8 was the diagnosis: replaying a biased log overstates a policy's value by
**2x**. Phase 9 is the treatment. Because the policy in Part I was itself *learned*
from that same biased log, it isn't merely mis-measured -- it is genuinely
suboptimal. Learning from confounded data bakes the confounding into the model.

---

## The problem: bias is inherited by the model, not just the metric

If item reward rates estimated from the production log are confounded (Phase 8),
then any policy that maximizes those rates chases the incumbent's preferences, not
the users'. The model faithfully learns the wrong objective. Crucially, **this
doesn't get better with more data** -- a bigger biased log just estimates the
biased quantity more precisely.

## The lever, again: KuaiRand's random log

The uniform-random exposure log gives per-item reward rates that are
*unconfounded* -- every item had the same chance of being shown, to anyone. Learn
from those and you optimize the real objective. The catch is variance: rarely-shown
items have noisy rates, so we **Bayesian-smooth** each rate toward the global mean
with a handful of pseudo-counts. Without smoothing, an item shown once that happened
to get a like scores a perfect 1.0 and hijacks a greedy policy -- a real,
instructive failure we guard against.

## Code tour

| File | Job |
|---|---|
| `phase9/learning.py` | Pure, unit-tested core: smoothed per-item reward rates, softmax / greedy policy construction, and the honest `policy_value` grader. Arrays in, numbers out (Rule 5). |
| `phase9/run.py` | Learns three policies (naive from the biased log, learned from the random log, greedy skyline), grades each by TRUE value, and runs a data-efficiency sweep. |

## How a policy is scored

A policy `pi(a)` is a distribution over items. Its **true value** is

```
V(pi) = Σ_a pi(a) · r_true(a)
```

where `r_true(a)` is the item's reward rate on the random log -- the only place
truth is defined. Every learned policy is graded on this one number.

The entropy-regularized optimal policy given value estimates `v(a)` is
`softmax(v / temperature)`: mass flows to high-value items, temperature sets how
greedily. The *only* thing that changes between the naive and learned policies is
**which log the value estimate came from**.

---

## Results

`cd phase9 && python run.py` (reward = engaged, i.e. MEDIUM or STRONG; temperature
0.05; smoothing 20 pseudo-counts):

```
  Policy (how it was learned)             TRUE value    vs naive
  uniform random (no learning)                0.1792          --
  pi_naive  = softmax(biased-log rates)       0.2173          --
  pi_learned= softmax(random-log rates)       0.4069        +87%
  pi_greedy = argmax(random-log rates)        0.5874       +170%

Data efficiency (learn from N random rows, grade on true value; mean +/- std
over 5 seeds so the sampling noise is visible, not hidden):
  random-log rows=    1,000  ->  0.1804 +/-0.0001   (0/5 seeds beat naive)
  random-log rows=   10,000  ->  0.1915 +/-0.0005   (0/5 seeds beat naive)
  random-log rows=  100,000  ->  0.2682 +/-0.0045   (5/5 seeds beat naive)
  random-log rows=1,186,059  ->  0.4069             (full log, no sampling)
```

### Reading the numbers like an engineer

- **Learning from unbiased data nearly doubles true value** (0.217 -> 0.407, +87%)
  using the *exact same* softmax learner. The only change is an honest input.
- **The greedy skyline (0.587)** is the ceiling if you trusted your estimates
  completely and never explored -- higher value, zero robustness. The softmax
  policy trades some value for exploration, which is what keeps future logs useful.
- **Bias does not average out.** The naive policy was fit on **1.44M** biased rows;
  it takes only ~**100k** *unbiased* rows to overtake it, and unbiased data keeps
  pulling ahead after that. Volume can't rescue a confounded objective.
- **Multi-seed makes the crossover honest.** Repeating each subsample under 5 seeds
  shows the win at 100k is unanimous (5/5) while 1k/10k lose on *every* seed -- so
  the crossover is a real effect, not one lucky draw. (The std at 1k is tiny
  because the Bayesian smoothing dominates a small sample -- itself a fair warning
  that a narrow interval can still be a badly *biased* estimate.)
- **Smoothing matters.** Turn it off and thinly-sampled items fake perfect rates,
  sending greedy value to a meaningless 1.0 -- a textbook small-sample trap.

---

## What Phase 9 taught us

1. **Bias corrupts learning, not just evaluation.** A model trained on confounded
   rewards optimizes the wrong thing, confidently and precisely.
2. **A little unbiased data beats a lot of biased data -- eventually.** There's a
   crossover (here ~100k rows), after which exploration data dominates. Only
   unbiased data fixes confounding; more biased data entrenches it.
3. **Explore on purpose.** In production you won't have a random log handed to you.
   Run a small, principled exploration bucket (epsilon-greedy / Thompson sampling)
   to mint unbiased data, then learn on it -- or IPS-correct the biased log with
   estimated propensities.

This closes Part II: **evaluate honestly (Phase 8), then learn honestly (Phase 9).**
The natural next step is *contextual* off-policy learning (per-user policies) and
online bandits that explore and learn in one loop.
