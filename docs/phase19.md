# Phase 19 -- Contextual Off-Policy Evaluation & Learning

> **Part II, generalized.** Phases 8-9 evaluated and learned a single *context-free*
> policy -- one distribution over items for everyone. Real policies are contextual:
> `pi(a | x)` depends on the user. Phase 19 takes the whole off-policy toolkit
> per-user and shows why that matters.

Phase 8 proved the metric can lie and showed IPS recovers the truth from a random log.
Phase 9 learned a policy off-policy. Both used one global policy. This phase asks: what
changes when the target policy is **per-user**, and does ignoring context break the
estimators we trusted?

## The setup

A controlled contextual world (like Phases 15-16): each arm has a REAL KuaiRand base
rate plus synthetic per-arm context weights, so there's a **known true reward model**
and "truth" is computable. The twist that makes this realistic: the **logging policy
is context-blind** -- it has the right base rates but zero context weights, i.e. the
old system that ranked by popularity and ignored personalization. From its log we try
to (A) evaluate and (B) learn a *contextual* policy.

`phase19/contextual_ope.py` (pure NumPy, reuses Phase 15's reward model and Phase 16's
per-arm ridge fitter -- DRY):

| Estimator | One-liner |
|---|---|
| `contextual_ips` | `mean( pi(a_i\|x_i)/p_i * r_i )` -- unbiased, high variance |
| `contextual_snips` | self-normalized IPS -- lower variance, tiny bias |
| `direct_method` | `mean_x sum_a pi(a\|x) rhat(x,a)` -- zero log-variance, biased if `rhat` is wrong |
| `doubly_robust` | DM + IPS correction on the model residual -- unbiased if EITHER model or propensities are right |

## Results

`cd phase19 && python run.py` (12 arms, 5-dim context, 8k-row logs, 20 worlds):

```
  A. OPE -- estimating the contextual target's TRUE value (0.7568),
     95% CI over 20 worlds:
    Estimator                    Estimate            95% CI     |error|
    Context-free IPS (P8)          0.5766   [0.5727, 0.5806]     23.8%  <- CI misses truth
    Contextual IPS                 0.7589   [0.7465, 0.7714]      0.3%
    Contextual SNIPS               0.7567   [0.7453, 0.7682]      0.0%
    Direct Method                  0.7438   [0.7341, 0.7535]      1.7%  <- CI misses truth
    Doubly Robust                  0.7561   [0.7448, 0.7674]      0.1%

  B. OPL -- TRUE value of the learned policy (95% CI over worlds):
    Logging (context-blind)            0.5725   [0.5710, 0.5741]
    Learned, context-free              0.5753   [0.5736, 0.5770]
    Learned, CONTEXTUAL                0.7362   [0.7276, 0.7449]
    Target (softmax of truth)          0.7568   [0.7452, 0.7684]
    Skyline (oracle)                   0.9332   [0.9251, 0.9413]
```

### Reading the numbers

- **A context-*blind* estimator is structurally biased for a contextual target (24%
  off).** To make the point concrete we construct the strawman deliberately: take the
  Phase 8 estimator and feed it the target policy's action frequencies *marginalized*
  over users (one global distribution). It literally cannot represent "different users
  get different items," so it mis-estimates the value badly.
  - *Honest framing:* this is a **constructed** worst case, not the estimator a
    practitioner would actually reach for -- nobody knowingly marginalizes away the
    context they're trying to evaluate. The value of the demonstration is the
    *principle* it isolates, stated next, not a claim that real teams make this exact
    mistake. (They make subtler versions of it: evaluating a personalized policy with
    a segment-averaged metric.)
  - The principle: the *estimator* has to match the *policy class* -- the contextual
    generalization of Phase 8's "measure it" lesson.
- **Contextual IPS / SNIPS / DR recover the truth (<=0.3%).** Once the estimator uses
  `pi(a|x)` per row, the importance weights are correct and the value comes back.
- **Doubly Robust is the safest.** Direct Method is close but leans entirely on the
  reward model (1.7% bias here); DR adds an IPS correction on the model's residual and
  is unbiased if *either* the model or the propensities are right -- the belt-and-braces
  estimator you want gating a real system.
- **The 95% CIs (across 20 worlds) make the bias visible.** The contextual IPS/SNIPS/DR
  intervals all *cover* the truth (0.7568); the context-free IPS `[0.573, 0.581]` and
  Direct Method `[0.734, 0.754]` intervals sit entirely *off* it. A CI that misses the
  target isn't noise you can average away -- it's a flag that the estimator is
  structurally wrong for this policy. (As in Phase 8: an interval bounds variance, not
  bias.)
- **OPL: context is the whole point.** Learning off the context-blind log, a
  context-free policy barely improves on the logger (0.5753 vs 0.5725) -- it just
  re-derives the old blind ranking. The **contextual** learned policy hits 0.7362,
  **+28%**, nearly matching the target. Personalization was recoverable from the blind
  log all along; you just had to learn *with* context.

## What Phase 19 taught us

1. **Match the estimator to the policy class.** A context-free estimator is biased for
   a per-user policy no matter how much data you have.
2. **Doubly Robust earns its keep contextually.** It's the low-variance, hard-to-fool
   estimator -- ideal as the redeploy gate from Phase 17.
3. **Contextual OPL recovers discarded personalization.** A context-blind log still
   contains the signal; a contextual learner extracts it.

Together with Phases 8, 9, and 11 this completes Part II's off-policy story: measure
value honestly, per user, with the estimator that matches your policy -- then learn the
policy that log implies.
