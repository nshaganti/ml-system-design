# Phase 14 -- The Explore-and-Learn Loop (where unbiased data comes from)

> **Google's Rule 23 & 36:** *Measure the delta between models; watch for silent
> failures.* Part II's quietest assumption was a source of *unbiased* data --
> Phase 8/9 needed a uniform-random log, Phase 11 needed a randomization bucket.
> Phase 14 answers the question those phases dodged: **in a live system, where does
> that data come from?** Exploration. And skipping it is the silent failure.

Every phase so far learned from a *fixed* log. But a real recommender chooses what to
show, and those choices become tomorrow's training data. A system that only ever
shows what it currently believes is best (pure exploitation) manufactures its own
selection bias: it never gathers evidence about the items it dismissed early, so it
can lock onto a lucky-but-mediocre item forever. That is the Phase 8 feedback loop,
created in real time.

## The setup: a bandit grounded in real rates

A multi-armed bandit is the smallest honest model of the loop. Each **arm** is an
item; pulling it yields a click with some hidden true rate; the policy picks an arm
each round and learns from the reward.

> **Honesty note.** Like Part II's causal experiments, the online loop is a
> *simulation* -- you cannot A/B a live policy inside a static log. But the arms are
> not invented: each arm's true reward rate is a **real per-item engagement rate
> measured from KuaiRand-Pure** (positives / impressions for the 50 most-exposed
> items). The world the policies act in is grounded in the data.

Three strategies (`bandit.py`, pure NumPy, unit-tested):

- **Greedy** -- warm up once, then always exploit the empirical best. The trap.
- **Epsilon-greedy** -- exploit, but pull a uniformly random arm 10% of the time,
  forever.
- **Thompson sampling** -- sample each arm's rate from its `Beta(1+wins, 1+losses)`
  posterior and play the sampled-best. Explores exactly as much as its uncertainty
  warrants.

## Results

`cd phase14 && python run.py` (50 arms, 50,000 rounds, averaged over 60 seeded
worlds -- because a *single* run lets greedy get lucky):

```
  Strategy                Mean regret   +/-sd   Best-arm%  Support  FoundBest%
  Greedy (exploit only)          1558    1799        18%      19%        18%
  Epsilon-greedy (e=0.1)         1540     381        38%     100%        53%
  Thompson sampling               814     146        71%     100%        98%
```

### Reading the numbers like an engineer

- **Greedy is a high-variance gamble.** Its mean regret (1558) hides a huge spread
  (+/-1799): sometimes one lucky warmup locks onto a near-best arm and it looks
  brilliant; often it locks onto a mediocre one and bleeds regret forever. Averaged
  over worlds it finds the truly-best arm only **18%** of the time.
- **Greedy's log is useless for causality.** It supports just **19%** of arms -- it
  stopped gathering evidence on everything it dismissed. That is *exactly* the
  confounded log Phase 8 had to correct after the fact, now manufactured live.
- **Naive epsilon-greedy explores wastefully.** Uniform exploration forever keeps
  full support but keeps *paying to pull known-bad arms*, so its regret barely beats
  greedy's.
- **Thompson wins on every axis that matters:** ~48% less regret than greedy, a tiny
  variance (+/-146), finds the best arm **98%** of the time, and keeps **100%**
  support. The regret *curve* tells the story best: Thompson's flattens as it
  converges and stops paying, while greedy's and epsilon's stay stubbornly linear.

### The punchline that ties the course together

That 100% support column is the whole point. **An exploring policy MINTS the
unbiased data that Phases 8, 9, and 11 could only assume it had.** Selection bias
(Part II), position bias (Phase 11), and the greedy feedback loop are the same
disease -- confounded logs -- and exploration is the cure applied at the source
instead of after the fact.

**Exploration is the price of unbiased data, and it is not optional.**

---

## What Phase 14 taught us

1. **Exploitation-only is self-blinding.** It optimizes the metric you can see while
   destroying the data you need to know whether the metric is right.
2. **Not all exploration is equal.** Uniform (epsilon-greedy) is wasteful; uncertainty-
   proportional (Thompson) is cheap. Explore where you're unsure, not everywhere.
3. **Average over worlds before you believe a bandit.** A single run rewards luck;
   the honest comparison is the mean (and variance) across many seeds.

This is the closing argument of the whole project: Part I built a recommender with
production discipline; Part II showed its offline numbers were biased, fixed the
learning, debiased positions -- and Phase 14 shows the *engine* that keeps all of it
honest in production is exploration. Next: a **contextual** bandit (LinUCB) that
conditions on user features, and wiring the exploration log straight into Phase 9's
off-policy learning to close the loop end to end.
