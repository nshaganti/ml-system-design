# Phase 15 -- The Contextual Bandit (LinUCB)

> **Google's Rule 4 / Rule 16:** *Start simple, then iterate.* Phase 14's bandit was
> the simplest honest online learner -- one reward rate per arm. Phase 15 takes the
> obvious next step every recommender needs: **the best item depends on the user.**

Phase 14 treated every user identically: it learned a single reward rate per arm and
picked the global best. But "best video for everyone" is not a recommender. A
*contextual* bandit conditions its choice -- and its exploration -- on a context
vector `x` (the user's features), so it can personalize while still exploring enough
to keep the log unbiased (Phase 14's lesson, now per-user).

## LinUCB in one screen

LinUCB (Li et al., 2010) keeps an independent ridge-regression reward model per arm
and picks the arm maximizing an optimistic score:

```
UCB_a(x) = theta_a . x   +   alpha * sqrt( x^T A_a^-1 x )
           predicted reward     uncertainty bonus (explore where unsure)
```

with `A_a = l2*I + sum(x xT)` and `theta_a = A_a^-1 b_a`, `b_a = sum(reward * x)`.
The bonus shrinks as an arm is seen in similar contexts, so LinUCB explores exactly
where its per-arm model is still uncertain -- the contextual analogue of Thompson's
uncertainty-proportional exploration. A constant `1.0` first feature is the per-arm
intercept, so `theta_a[0]` is Phase 14's context-free base rate.

## Code tour

| File | Job |
|---|---|
| `phase15/linucb.py` | Pure NumPy: `LinUCB` (select/update), `ContextFreeThompson` (Phase 14's sampler behind the same interface = the context-blind baseline), `reward_prob`, and `simulate_contextual`. 6 unit tests incl. a contextual-world proof. |
| `phase15/run.py` | Builds a contextual world (real KuaiRand base rates + synthetic per-arm context weights) and grades three policies over 25 worlds. |

> **Honesty note.** Like every online/causal phase, this is a *simulation* -- you
> cannot A/B a live policy inside a static log. Each arm's BASE rate (its intercept)
> is a **real KuaiRand-Pure per-item engagement rate**; only the context-dependent
> part is synthetic, so the world stays grounded in the data.

---

## Results

`cd phase15 && python run.py` (12 arms, 4 user features + bias, 8,000 rounds,
averaged over 25 worlds):

```
  Policy                        Regret   +/-sd   Best-arm%
  Context-free Thompson           2526     206         9%
  LinUCB (alpha=0, greedy)        1580     474        25%
  LinUCB (alpha=1.0)               203      22        62%
```

### Reading the numbers like an engineer

- **Context helps.** LinUCB with *no* exploration bonus (greedy, 1580) already beats
  context-free Thompson (2526). A policy that ignores the user can only learn each
  arm's *average* rate, so on a world where the best arm flips per user it is
  structurally stuck -- it picks the per-user-best only 9% of the time.
- **Exploration still helps on top of context.** Adding the uncertainty bonus
  (alpha=1) crushes regret from 1580 to **203** and triples per-user-best picks to
  62%. Pure exploitation commits to arms on noisy early estimates *within each
  context region*; the bonus keeps refining where data is thin. This is Phase 14's
  greedy-vs-Thompson gap, reproduced inside the contextual setting.
- **Net:** LinUCB gets **92% lower regret** than the Phase 14 champion, and it
  personalizes -- the whole reason to add context.

### Why the win is so large here

The world is *designed* so context matters (the best arm genuinely flips with a
feature's sign). That is the honest framing: **contextual methods pay in proportion
to how much the right answer actually varies by user.** On a world where one item is
best for everyone, LinUCB would collapse toward Phase 14 and the extra machinery
would buy little -- exactly the "earn your complexity" test from Part I, applied to
personalization.

---

## What Phase 15 taught us

1. **Personalization is a modelling choice with a measurable payoff.** Condition on
   context and you can beat any context-blind policy on a world where users differ --
   but only by as much as they actually differ.
2. **Exploration composes with context.** The uncertainty bonus matters just as much
   per-context as it did globally; greedy gets stuck in every context region at once.
3. **Same interface, honest comparison.** Wrapping Phase 14's Thompson sampler in the
   contextual policy interface let the context-blind baseline run on the exact same
   world -- no re-implementation, a true apples-to-apples verdict.

This is the last rung of the arc: Part I built a disciplined recommender, Part II
proved its metrics were biased and fixed the learning, Phase 14 showed exploration
mints the unbiased data, and Phase 15 makes that exploration *personalized*. The
natural close: feed LinUCB's exploration log into Phase 9's off-policy learning
(explore online -> learn off-policy -> redeploy), and use the two-tower user vector
from Phase 1 as the real context `x`.
