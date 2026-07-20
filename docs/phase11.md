# Phase 11 -- Position-Bias Debiasing (controlled simulation)

> **Google's Rule 36 (again):** *Watch for silent failures.* A click log's silent
> failure is that clicks measure attention, not relevance. Position bias is the
> quietest, most expensive lie in learning-to-rank -- and it has a clean fix.

> **Honesty note.** Every other phase runs on the real KuaiRand-Pure logs.
> This one can't: KuaiRand-Pure does not record the on-screen POSITION of each
> impression (its `tab` field is a feed id, not a rank). So Phase 11 is a
> *controlled simulation* -- known examination curve, known relevance -- which is
> exactly how you validate a debiasing estimator before trusting it on logs that
> DO carry position. Treat it as the physics-lab demo of the technique.

---

## The problem: clicks rank positions, not items

An item at rank 1 is seen by almost everyone; an item at rank 40 is seen by almost
no one. So its click-through rate reflects *where it was shown* at least as much as
*how good it is*. Train a ranker on raw clicks and you get a vicious feedback loop:
whatever the old ranker put on top gets clicks, looks "relevant," and gets put on
top again -- while a genuinely great item buried at rank 30 never gets the chance to
prove itself.

## The model: examination x relevance

The Position-Based Model factorizes a click:

```
P(click | item, position) = P(examine | position) * P(relevant | item)
                          =        e_p            *        r_i
```

- **Naive CTR** estimates `e_p * r_i` -- confounded by the slot.
- **IPW** divides each click by `e_p` (the propensity of the slot it was shown in),
  which cancels the examination term and recovers `r_i` in expectation.
- You may not know `e_p`. A **result-randomization bucket** -- show items in random
  positions for a slice of traffic -- lets you estimate it: with relevance averaged
  out across random items, mean CTR at each position is proportional to `e_p`. Same
  idea as Part II's uniform-random log, applied to position.

## Code tour

| File | Job |
|---|---|
| `phase11/position_bias.py` | Pure core (Rule 5): examination curve, PBM click simulator, naive CTR, IPW, randomization-based propensity estimation, plus Spearman and top-k recovery for grading. |
| `phase11/run.py` | Builds the scenario (imperfect production ranking pins items to slots), logs clicks, and recovers relevance three ways, grading each against truth. |

---

## Results

`cd phase11 && python run.py` (40 items, examination decay 0.6, an imperfect
production prior, IPW graded by how well the recovered ranking matches truth):

```
  Estimator                         Spearman     Top-10 recovery
  Naive CTR                            0.862                60%
  IPW (true propensity)                0.967                80%
  IPW (estimated propensity)           0.948                80%
  Examination-curve recovery (est vs true): Spearman 0.896
```

### Reading the numbers like an engineer

- **Naive CTR mis-ranks** (Spearman 0.86, only 60% of the true top-10 recovered):
  the slot leaks into the score.
- **IPW with the true curve recovers the ranking** (0.97, 80%) by dividing the slot
  effect out.
- **You don't need the true curve.** IPW with a curve *estimated* from the
  randomization bucket (0.95) nearly matches the oracle -- and the estimated curve
  itself tracks the truth at Spearman 0.90.
- **The catch: IPW is unbiased but high variance.** Divide a click by a tiny
  deep-slot propensity (e.g. 0.004) and that one observation swings the estimate
  wildly. With a steep decay over many positions, IPW can actually *lose* to naive
  CTR from variance alone -- which is why production systems clip propensities or use
  self-normalized IPW (the SNIPS trick from Phase 8). Bounded propensities here keep
  it stable; that choice is the whole ballgame.

---

## What Phase 11 taught us

1. **Clicks are examination x relevance.** Modelling that split is what separates a
   ranker that learns relevance from one that launders its predecessor's ranking.
2. **Randomization buys honesty -- for position, too.** A small random-placement
   bucket estimates the examination curve, exactly as the random log estimates
   selection propensities in Part II. Same disease, same cure.
3. **Unbiased is not free.** IPW trades bias for variance; know when to clip or
   self-normalize. "Unbiased but unusable" is a real failure mode.

This ties Parts I and II together: **selection bias (Part II) and position bias
(here) are the same confounding, and both are fixed by propensities you either know
or can go measure.** On logs that carry position, these exact estimators produce
IPW-weighted labels you'd feed straight into the Phase 2 ranker.
