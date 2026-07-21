# Phase 16 -- Closing the Loop (the capstone)

> **Google's Rule 16:** *Plan to launch and iterate.* This phase IS the iterate.
> Every earlier phase was one arc of a cycle a real recommender runs forever. Phase
> 16 wires them into the loop.

```
     deploy a policy  -->  it EXPLORES and logs (context, action, propensity, reward)
         ^                                                        |
         |                                                        v
     redeploy the   <--  LEARN a better policy OFF-POLICY from that log,
     greedy policy       correcting the logging bias with propensities
```

This is the whole course in one diagram:

- **Phase 15** gives the contextual world and an epsilon-soft, LinUCB-style logging
  policy (exploration with *known propensities*).
- **Part II** (Phases 8/11) gives the propensity correction: IPS-weight each arm's
  reward model so the contexts an arm happened to be shown in don't bias it.
- **Phase 9** gives the off-policy learning step and the honest yardstick: grade the
  deployed policy by its **true value**, not its logged reward.

## Code tour

| File | Job |
|---|---|
| `phase16/loop.py` | Pure core: `epsilon_greedy_propensities`, `fit_reward_models` (per-arm ridge, optional IPS weights), `greedy_value` / `skyline_value` / `uniform_value`, and `simulate_loop` (the full explore->learn->redeploy cycle returning true value per iteration). 7 unit tests incl. a closed-loop-beats-trap proof. |
| `phase16/run.py` | Builds the contextual world (real KuaiRand base rates) and ablates three loop variants against the skyline. |

> **Honesty note.** This is a *simulation* -- you cannot run a live loop inside a
> static log. But each arm's base rate is a **real KuaiRand-Pure per-item engagement
> rate** (Phase 15's world), so the ceiling the loop climbs toward is grounded in data.

---

## Results

`cd phase16 && python run.py` (12 arms, 15 redeploy iterations of 1,000 logged rows,
averaged over 15 worlds):

```
  Variant                          True value    % skyline gap
  Uniform floor                        0.5660              0%
  No exploration (trap)                0.7151             41%
  Explore, no IPS                      0.9253             98%
  Closed loop (explore+IPS)            0.9149             96%
  Skyline (oracle)                     0.9311            100%
```

### Reading the numbers like an engineer

- **Exploration is the hero.** A loop that learns once from greedy logs stalls at
  **41%** of the floor->skyline gap: it only ever logs the arm it already likes, so
  every other arm's model stays frozen at the prior and the deployed value plateaus.
  This is Phase 8/14's feedback trap, now shown to cap a *live* system's ceiling.
  Turn exploration on and the deployed value climbs to **96-98%** of the skyline,
  redeploy after redeploy.
- **The honest wrinkle: IPS barely moved the needle** (96% with, 98% without). This
  is not a bug -- it's the deepest lesson in the project. With a **well-specified
  linear** per-arm reward model, ordinary regression is already unbiased even on a
  skewed context distribution (OLS recovers the true weights regardless of which
  contexts an arm was shown in). So IPS here only *added variance* -- Phase 11's
  bias-variance tradeoff, one final time.
- **When do propensities earn their keep, then?** When the reward model is
  *misspecified*, or when you estimate a policy's **value directly** (Phase 8's OPE)
  instead of fitting a reward model. Coverage (exploration) is what a
  correctly-specified model needs; reweighting is situational. Knowing which tool the
  situation calls for is the difference between citing techniques and engineering.

### The one-sentence capstone

**Explore for coverage, learn off-policy, redeploy -- and measure the true value of
what you ship, not the reward your logs happened to record.**

---

## What Phase 16 taught us -- and how it ties the course together

1. **A recommender is a loop, not a model.** The model is one step; the value is
   created (or destroyed) by how the loop gathers and corrects its own data.
2. **Exploration sets the ceiling.** No amount of clever offline learning beats the
   trap of never gathering evidence on your alternatives.
3. **Use the right correction for the right failure.** Propensities fix confounding
   and direct value estimation; a well-specified model just needs coverage. The
   engineering skill is diagnosis, not ritual.

Across the whole project: **Part I** built a recommender with production discipline
(and honestly measured that most added complexity bought robustness, not accuracy).
**Part II** proved the offline metrics were biased and fixed the learning, debiased
positions, and -- Phases 14->16 -- showed that the engine keeping all of it honest in
production is *exploration inside a closed off-policy loop.* Everything the course
assumed about "unbiased data" is, in the end, something a well-run loop manufactures
for itself.

Further iterations (left as the standing backlog): use the two-tower user vector as
the real context `x`; add doubly-robust learning to cut IPS variance where it *does*
matter; and gate every redeploy behind the Phase 4 monitoring + a min-propensity
floor so a bad iteration can never ship.
