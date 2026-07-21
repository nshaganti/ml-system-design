# Phase 17 -- Real Context + Safety-Gated Redeploys

> **Google's Rule 16 (with a parachute):** *Plan to launch and iterate* -- but never
> ship an iteration you haven't checked. Phase 16 closed the loop; Phase 17 makes it
> safe to run in production.

Phase 16 proved the closed loop works, but it took two shortcuts a real system can't:
its context was synthetic uniform noise, and every learned candidate was redeployed
*blindly*. Phase 17 removes both.

## What's new

1. **Real context.** The loop acts on **standardized per-user features from
   KuaiRand-Pure** -- log-activity, strong-signal ratio, medium-signal ratio, and
   distinct-item ratio for ~23.5k real users. That's a correlated, non-uniform
   context distribution, like real traffic, instead of tidy uniform noise.
2. **A safety gate.** Before a candidate replaces the deployed policy, we estimate
   its value **off-policy (Phase 8 IPS)** on a fresh **uniform-random validation
   bucket** with clean rewards, and only redeploy if it beats the incumbent *there*.
   Uniform logging gives every arm propensity `1/n`, so the estimate is unbiased for
   any candidate and no incumbent gets a home-field advantage.
3. **A min-propensity floor.** Logging propensities are clipped from below before
   inversion, bounding importance weights so one rare action can't blow up the gate's
   estimate -- Phase 11's bias-variance tradeoff, made an explicit safety knob.
4. **Realistic online retraining.** Candidates are fit on a **recency window** of
   recent batches (not all history), which is both how real systems retrain and what
   makes a corrupt *recent* batch genuinely dangerous.

## Code tour

| File | Job |
|---|---|
| `phase17/safe_loop.py` | Pure core: `ips_policy_value` (OPE gate), `_uniform_validation` (clean bucket), `simulate_safe_loop` (explore -> learn-on-window -> gate -> redeploy, with a `poison_iter` hook). Reuses Phase 15/16 primitives. 6 unit tests. |
| `phase17/run.py` | Builds real user contexts + arms and ablates four scenarios (gated/ungated x clean/poisoned) over 15 worlds. |

> **Honesty note.** The reward is still a *simulation* (you can't A/B inside a static
> log), but the **contexts are real** KuaiRand user features, so the loop acts on a
> realistic distribution.

---

## Results

`cd phase17 && python run.py` (12 arms, real contexts, 15 iterations, 15 worlds; a
logging bug flips one training batch's rewards at iteration 8):

```
  Scenario                    Final value    % gap   Worst deploy
  Uniform floor                    0.5564       0%
  Ungated, clean                   0.7997      59%         0.6572
  Gated, clean                     0.8322      67%         0.6473
  Ungated, POISONED                0.7922      57%         0.5698
  Gated, POISONED                  0.8272      66%         0.6090
  Skyline (oracle)                 0.9666     100%
```

### Reading the numbers like an engineer

- **The gate helps even in the clean case (67% vs 59%).** With recency-window
  retraining, candidates are noisy; validating each on a fresh uniform-random bucket
  (Phase 8's unbiased log, used *live*) and shipping only true winners beats blind
  redeploys. Off-policy evaluation isn't just an offline audit -- it's a live control.
- **The gate is insurance, and the poison run collects on it.** When a logging bug
  flips a training batch, the ungated loop ships the corrupt candidate and its worst
  deployed value **craters to 0.5698**. The gate rejects it on the clean bucket, so
  the gated loop's worst stays **0.6090**. A safety gate is *priced in the good case
  and cashed in the bad*.
- **Honest caveat on the gate's design.** An earlier version scored candidates on the
  data the incumbent itself generated -- that gives the incumbent a home-field
  advantage and over-blocks (it rejected ~10/15 good candidates). Switching the gate
  to a uniform-random validation bucket fixed it. *How* you estimate the gate matters
  as much as having one -- the same Part II lesson (unbiased data beats convenient
  data) applies to the gate that guards the loop.

---

## What Phase 17 taught us

1. **Off-policy evaluation is a production control, not just an audit.** The same IPS
   estimator from Phase 8 becomes the gate that decides what ships.
2. **Guard the guard.** A safety gate scored on biased data is itself biased; validate
   on an unbiased bucket or you'll over-block or rubber-stamp.
3. **Real contexts change the verdict.** Correlated, non-uniform traffic makes
   candidates noisier and the gate more valuable than the tidy Phase 16 world implied.

This is the production-real finale of the loop arc: explore on real contexts, learn
off-policy, **gate every redeploy on unbiased data**, and never let a bad iteration
ship. The remaining backlog (two-tower user vector as context, a canary slice,
doubly-robust gating) are refinements on this skeleton, not new ideas.
