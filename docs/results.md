# Results & Performance Progression

This is the consolidated scoreboard for Phases 0-8 on **real KuaiRand-Pure data**
(seed 42, 80/20 temporal split). Read the **big caveat** first, because the single
most important thing about these numbers is that **most of them are not directly
comparable to each other** -- and pretending otherwise is exactly the kind of
self-deception this whole project is about avoiding.

> **How this file stays honest.** Every table between `<!-- AUTOGEN -->` markers is
> regenerated from each phase's `results.json` by `scripts/build_results.py` -- so
> the numbers in the tables are exactly what the code last produced, never
> hand-copied. The surrounding prose is human-written and quotes numbers
> *approximately* (they wobble a little run-to-run from negative sampling and user
> subsampling); trust the tables for the precise values.

---

## The big caveat: read groups, not one trend line

It is tempting to draw one line -- "we added complexity, the metric went up." For
this system that line would be a **lie**, for three reasons:

1. **Different metrics.** Phases 0-2 report `Recall@20`; Phases 5-6 report `hit@20`
   (a per-user binary). These measure different things.
2. **Different denominators / candidate sets.** Phase 0-1 recall is over the
   *full catalog*. Phase 2 reranks a fixed candidate pool. Phase 7 is next-item
   *within a session* -- an easier task entirely.
3. **Different populations.** Phase 5/6 evaluate subsets (test positives,
   multi-event users), not the same user set as Phase 0/1.

So the tables below are **grouped by what is actually comparable.** Cross-group
comparisons are explicitly flagged as invalid.

> **The honest through-line (KuaiRand):** on dense feedback and a small catalog,
> the ML retrieval model **does** beat the heuristic (Phase 1, ~+83% recall). But a
> weak single cross-feature **hurts** (Phase 2), the A/B test honestly rules that
> weak ranker out (Phase 5, significant ~-6%), and in-session freshness barely
> moves (Phase 6). Then **Part II (Phase 8) drops the real bomb: the offline metrics
> everyone trusts were biased by 2x.** Complexity bought robustness and honesty;
> causality-aware evaluation bought *truth*.

---

## Group A -- Full-catalog retrieval quality (Phase 0 vs Phase 1)

*Comparable.* Same metric, same full-catalog candidate universe, same test users,
same temporal split.

<!-- AUTOGEN:groupA -->
| Metric | Phase 0 (heuristic) | Phase 1 (two-tower) | Change |
|---|---|---|---|
| Recall@20 | 0.0682 | **0.1246** | +83% |
| Warm-user recall | 0.0665 | **0.1252** | +88% |
| Cold-start recall | **0.1179** | 0.1058 | -10% |
| Catalog coverage | 0.0744 | **0.1603** | +116% |
<!-- /AUTOGEN:groupA -->

**Verdict:** the learned two-tower **beats** the heuristic on the metrics that
measure learning -- roughly 1.8x recall and 2.2x coverage. (Cold-start recall is a
wash: cold users are served by the *same* heuristic fallback in both systems, so
any gap there is sampling noise, not signal.) This is the *opposite* of what a
sparse e-commerce log tends to show: KuaiRand's feedback is dense (a third of
events are strong) and the catalog is small (~7.5k videos), so ID-embedding
retrieval has enough signal to learn from. See [`phase1.md`](phase1.md).

---

## Group B -- Reranking a fixed candidate pool (Phase 2)

*Comparable within the group only.* Both rankers reorder the **same popularity
pool**; the only difference is the ranker.

<!-- AUTOGEN:groupB -->
| Metric | Popularity order | LR ranker | Change |
|---|---|---|---|
| Recall@20 | **0.0689** | 0.0635 | -8% |
| NDCG@20 | **0.0446** | 0.0386 | -13% |
<!-- /AUTOGEN:groupB -->

**Verdict:** the LR ranker with a single `user_cat_affinity` cross feature
**loses** to raw popularity order. On KuaiRand the category tag is coarse and
engagement is popularity-driven, so that one feature carries little signal -- and a
feature with no signal is dead weight (Rules 17 & 20). This is a genuine,
instructive negative result, not a bug. The point-in-time feature store and the
train/serve skew audit still matter regardless of the lift's sign.

---

## Group C -- Per-user hit@20 (Phases 5 & 6)

*Comparable within the group.* Metric = "did any of the user's actual later
positive actions land in the top-20?" (binary per user). **Not comparable to
Groups A/B** (different metric and populations).

<!-- AUTOGEN:groupC -->
| Experiment | Arm A | Arm B | Lift | Significance |
|---|---|---|---|---|
| **Phase 5:** popularity vs LR ranker | **0.2278 (popularity)** | 0.2134 (LR) | -6.3% | p=0.0327 -- **significant** |
| **Phase 6:** frozen vs fresh features | 0.2343 (frozen batch) | 0.2349 (fresh stream) | +0.2% | p=0.903 -- **not significant** |
<!-- /AUTOGEN:groupC -->

**Verdict:** the A/B test (Phase 5) confirms Group B's finding with proper
statistics -- the weak LR ranker is *significantly worse*, so **do not ship**.
Feature freshness (Phase 6) barely moves and is not significant: short-video
engagement here is less bursty-intent than an e-commerce cart, so a 30-second delta
layer adds little. Same statistical machinery, honest verdicts in both directions.

> Sanity check: Phase 5's popularity arm (0.2276) and Phase 6's frozen arm (0.2343)
> are ~equal, as they should be -- both are the batch/popularity setup measured the
> same way. That consistency is a small confidence signal that the harness behaves.

---

## Group D -- Session-based next-item (Phase 7, the community protocol)

*Comparable within the group.* Leave-one-out next-item within a session. **Not
comparable to Groups A-C** (different, easier task: next *item in session*).

<!-- AUTOGEN:groupD -->
| Metric | Popularity | Co-visitation | Lift |
|---|---|---|---|
| Recall@20 | 0.0500 | **0.0798** | +60% |
| MRR@20 | 0.0128 | **0.0204** | +60% |
| NDCG@20 | 0.0207 | **0.0332** | +60% |
<!-- /AUTOGEN:groupD -->

**Verdict:** a simple, untuned, pure-Python co-visitation model beats popularity by
~60% on the session task -- the session signal is real and cheap to exploit. See
[`benchmarking-vs-literature.md`](benchmarking-vs-literature.md).

---

## Group E -- Off-policy evaluation (Phase 8, Part II)

*Not comparable to anything above -- it grades a **policy's value**, and it grades
the honesty of offline evaluation itself.* Ground truth is computable only because
KuaiRand ships a uniform-random log.

<!-- AUTOGEN:groupE -->
| Estimator | Value | Error vs truth |
|---|---|---|
| **Ground truth** (π × random-log rewards) | 0.261 | -- |
| Naive / Direct Method (biased log) | 0.523 | **100.2%** |
| IPS | 0.244 | 6.4% |
| **SNIPS** | 0.259 | **0.6%** |
| Doubly Robust | 0.277 | 6.1% |
<!-- /AUTOGEN:groupE -->

**Verdict:** the naive offline metric -- the one most teams ship on -- overstates
the target policy's true value by **2x**, purely from confounding. Reweighting the
random log by known propensities recovers the truth (SNIPS to 0.6%). This is the
capstone lesson of the repo: *even after all of Part I's discipline, your offline
number can still be a factor of two wrong.* See [`phase8.md`](phase8.md) and
[`off-policy-evaluation.md`](off-policy-evaluation.md).

---

## Group F -- Off-policy LEARNING (Phase 9, Part II)

*Not comparable to Groups A-E -- it grades a **learned policy's true value**
(V(pi) = sum_a pi(a) r_true(a)), where r_true comes from the random log.* Phase 8
showed evaluating on biased logs lies; Phase 9 shows *learning* on them yields a
genuinely worse policy -- and that learning on unbiased data fixes it.

<!-- AUTOGEN:groupF -->
| Policy (how it was learned) | True value | vs naive |
|---|---|---|
| uniform random (no learning) | 0.179 | -18% |
| pi_naive = softmax(biased-log rates) | 0.217 | -- |
| **pi_learned = softmax(random-log rates)** | **0.407** | **+87%** |
| pi_greedy = argmax(random-log rates) | 0.587 | +170% |
<!-- /AUTOGEN:groupF -->

**Verdict:** a policy *learned* from the unbiased random log has ~**1.9x** the true
value of one learned from the (much larger) biased production log. The data-
efficiency sweep drives the point home: it takes ~100k unbiased rows to overtake a
policy fit on 1.44M biased rows, and unbiased data keeps pulling ahead after that.
**Bias doesn't average out with volume -- only unbiased data fixes it.** (Reward
rates are Bayesian-smoothed so thinly-sampled items can't fake a perfect score.)
See [`phase9.md`](phase9.md).

---

## Group G -- Sequence model vs co-visitation (Phase 10)

*Comparable within the group only -- same leave-one-out session protocol and the
SAME test cases as Group D (Phase 7).* The only thing that changes is the model:
popularity, co-visitation (co-occurrence), and GRU4Rec (a sequence model that reads
session order).

<!-- AUTOGEN:groupG -->
| Metric | Popularity | Co-visitation | GRU4Rec |
|---|---|---|---|
| Recall@20 | 0.0498 | **0.0801** | 0.0691 |
| MRR@20 | 0.0128 | **0.0203** | 0.0172 |
| NDCG@20 | 0.0207 | **0.0332** | 0.0283 |
<!-- /AUTOGEN:groupG -->

**Verdict:** an honest surprise. A basic, CPU-budget GRU4Rec **beats popularity**
(order carries real signal) but **loses to plain co-visitation** by ~15%. On a
small catalog with strong pairwise co-occurrence, the cheap item-kNN is a
remarkably hard baseline; closing the gap would need heavier tuning (negative
sampling, a bigger model, many more epochs) -- and might still not be worth it. This
is Phase 2's lesson again: **the fancier model is not automatically better -- you
measure it.** See [`phase10.md`](phase10.md).

---

## Group H -- Position-bias debiasing (Phase 11, controlled simulation)

*Not comparable to any other group -- it grades how well an estimator recovers the
true relevance RANKING from position-confounded clicks.* **Honesty note:**
KuaiRand-Pure logs no on-screen position, so unlike every other phase this one runs
on a controlled simulation (known examination curve + known relevance), the way you
prove an estimator works before trusting it on real position-carrying logs.

<!-- AUTOGEN:groupH -->
| Estimator | Spearman vs truth | Top-10 recovery |
|---|---|---|
| Naive CTR | 0.862 | 60% |
| IPW (true propensity) | **0.967** | 80% |
| IPW (estimated propensity) | 0.948 | 80% |
<!-- /AUTOGEN:groupH -->

**Verdict:** naive click-through rate ranks *positions* as much as items, so it
mis-orders relevance. Inverse-propensity weighting divides out each slot's
examination probability and recovers the true ranking (Spearman 0.86 -> 0.97). And
you don't need to know the examination curve: a small **result-randomization
bucket** estimates it (curve recovered at Spearman 0.90), and IPW with the
estimated curve is nearly as good as the oracle. This is the same disease and cure
as Part II -- confounded logs, fixed by known (or known-able) propensities. One
catch worth remembering: IPW is unbiased but *high variance* -- divide by a tiny
deep-slot propensity and the estimate explodes, so bounded propensities / clipping
matter in practice. See [`phase11.md`](phase11.md).

---

## Group I -- Two-stage integration: does retrieval + ranking beat one stage? (Phase 12)

*Comparable within the group only -- three rankers, same users, same temporal split,
same `recall_at_k` harness.* This is the first phase that WIRES earlier pieces
together: two-tower retrieval (P1) feeding the LR ranker (P2), versus each stage on
its own.

<!-- AUTOGEN:groupI -->
| Ranker | Recall@20 | NDCG@20 | Coverage |
|---|---|---|---|
| Two-tower alone (P1) | **0.1215** | **0.0843** | **0.1564** |
| Popularity -> LR (P2) | 0.0590 | 0.0372 | 0.0499 |
| Two-tower -> LR (P12) | 0.0988 | 0.0697 | 0.0789 |
<!-- /AUTOGEN:groupI -->

**Verdict:** the architecture textbooks draw -- retrieve then rerank -- **loses to
two-tower retrieval alone** here (-18.7% recall, -17% NDCG, half the coverage). Yet
the *same* two-stage system beats popularity->LR by +67% recall, so stage 1
absolutely matters. The lesson is about stage 2: the LR's features
(`item_pop`, `user_pop`, `user_cat_affinity`) are popularity-flavored, so reranking
pushes popular items up and *undoes* the two-tower's personalization and long-tail
coverage. **A two-stage system is only as good as the signal its ranker adds** --
architecture alone buys nothing (Rules 4, 14). Fix: give stage 2 features it can
actually rank with (the two-tower similarity score itself, recency, affinities).
See [`phase12.md`](phase12.md).

---

## Group J -- Stage 2 that earns its place: the two-tower score as a feature (Phase 13)

*Comparable within the group only -- three rankers, same users/split/harness.* The
direct fix for Group I's regression: give the LR the two-tower similarity score as a
feature so the rerank stops fighting the retriever.

<!-- AUTOGEN:groupJ -->
| Ranker | Recall@20 | NDCG@20 | Coverage |
|---|---|---|---|
| Two-tower alone | 0.1236 | 0.0871 | **0.1544** |
| TT -> LR (pop feats) | 0.1002 | 0.0688 | 0.0797 |
| TT -> LR + tt_score | **0.1277** | **0.0897** | 0.1346 |
<!-- /AUTOGEN:groupJ -->

**Verdict:** the fix works. Adding one feature -- the retrieval score itself --
turns Group I's -19% regression into a **+3.3% recall / +3% NDCG win over two-tower
alone**, while popularity-only reranking still regresses (-19%). With the score in
hand, the LR *preserves* good retrieval order and only reorders when a popularity
signal genuinely helps; without it, the ranker was flying blind and shoving popular
items up. Coverage stays below two-tower-alone (the ranker still tilts a little
toward popular items), which is the honest cost of the small accuracy gain. **Two
stages beat one only once stage 2 can see what stage 1 knows.** See
[`phase13.md`](phase13.md).

---

## Group K -- The explore-and-learn loop: where unbiased data comes from (Phase 14)

*Comparable within the group only -- three bandit strategies, same 50 arms (whose
true reward rates are REAL KuaiRand per-item engagement rates), averaged over 60
seeded worlds x 50k rounds.* This closes Part II's loop: exploration is the ONLINE
source of the unbiased data Phases 8/9/11 assumed.

<!-- AUTOGEN:groupK -->
| Strategy | Mean regret | Best-arm % | Log support | Found best % |
|---|---|---|---|---|
| Greedy (exploit only) | 1558 | 18% | 19% | 18% |
| Epsilon-greedy | 1540 | 38% | **100%** | 53% |
| Thompson sampling | **814** | **71%** | 100% | **98%** |
<!-- /AUTOGEN:groupK -->

**Verdict:** pure exploitation is a **high-variance gamble** (regret 1558 +/-1799):
averaged over worlds it rarely finds the true best arm (18%) and its ongoing log has
support on almost no arm (19%) -- literally the confounded log Phase 8 had to correct
after the fact, manufactured live. Naive epsilon-greedy keeps full support but
explores *wastefully* (uniform forever), so its regret barely improves. **Thompson
sampling** explores in proportion to uncertainty: **~48% less regret than greedy**,
tiny variance (+/-146), finds the best arm 98% of the time, and keeps full support.
That support is the punchline -- an exploring policy MINTS the unbiased data that
Phases 8-9-11 could only assume. **Exploration is the price of unbiased data, and it
is not optional.** See [`phase14.md`](phase14.md).

---

## Group L -- The contextual bandit: personalized exploration (Phase 15)

*Comparable within the group only -- three policies, same contextual world (each
arm's base rate is a REAL KuaiRand per-item rate; the context-dependent part is
synthetic), averaged over 25 worlds x 8k rounds.* Extends Phase 14 from "one best
item for everyone" to "the best item depends on the user."

<!-- AUTOGEN:groupL -->
| Policy | Mean regret | Per-user-best % |
|---|---|---|
| Context-free Thompson | 2526 | 9% |
| LinUCB (alpha=0, greedy) | 1580 | 25% |
| LinUCB (alpha=1) | **203** | **62%** |
<!-- /AUTOGEN:groupL -->

**Verdict:** two lessons in one table. **Context helps** -- LinUCB even without an
exploration bonus (greedy, 1580) beats context-free Thompson (2526), because a
policy that ignores the user can only ever learn each arm's *average* rate and is
structurally stuck on a world where the best arm flips per user. **And exploration
still helps on top of context** -- adding the uncertainty bonus (alpha=1) crushes
regret from 1580 to **203** and lifts per-user-best picks to 62%: the bonus finds
good arms in under-seen contexts fast. Net, LinUCB gets **92% lower regret** than
the Phase 14 champion and personalizes where it couldn't. This is Phase 14's lesson,
now per-user, and the natural home for the two-tower user vector as the context x.
See [`phase15.md`](phase15.md).

---

## Group M -- Closing the loop: explore -> learn off-policy -> redeploy (Phase 16)

*The capstone. Comparable within the group -- three loop variants + skyline/floor on
the same contextual world (real KuaiRand base rates), averaged over 15 worlds x 15
redeploy iterations.* Grades each variant by the TRUE value of the policy it
actually deploys.

<!-- AUTOGEN:groupM -->
| Policy | True deployed value | % of skyline gap closed |
|---|---|---|
| Uniform floor | 0.5660 | 0% |
| No exploration (trap) | 0.7151 | 41% |
| Explore, no IPS | 0.9253 | 98% |
| Closed loop (explore+IPS) | 0.9149 | 96% |
| Skyline (oracle) | 0.9311 | 100% |
<!-- /AUTOGEN:groupM -->

**Verdict:** the loop works, and it names the one ingredient that matters most.
**Exploration is the hero**: a loop that learns once from greedy logs stalls at 41%
of the floor->skyline gap -- it never gathers evidence on the arms it dismissed, so
its models (and its ceiling) freeze. Turning on exploration lifts the deployed value
to **96-98%** of the skyline, iteration after iteration. The honest wrinkle: **IPS
barely moved the needle here** (96% with vs 98% without). With a well-specified
*linear* per-arm reward model, plain regression is already unbiased on a skewed
context distribution, so IPS only added variance -- Phase 11's bias-variance tradeoff,
one last time. Propensities earn their keep when the model is misspecified or you
estimate policy *value* directly (Phase 8), not when a correctly-specified model just
needs coverage. **Explore for coverage, learn off-policy, redeploy -- that is the
engine the whole course was building toward.** See [`phase16.md`](phase16.md).

---

## Group N -- Real context + safety-gated redeploys (Phase 17)

*The production-real finale. Comparable within the group -- four loop variants +
skyline/floor, averaged over 15 worlds x 15 redeploy iterations. Contexts are REAL
standardized per-user features (activity + signal-mix) from ~23.5k KuaiRand users; a
simulated logging bug flips one training batch's rewards at iteration 8.*

<!-- AUTOGEN:groupN -->
| Scenario | Final value | % skyline gap | Worst deploy |
|---|---|---|---|
| Uniform floor | 0.5564 | 0% | -- |
| Ungated, clean | 0.7997 | 59% | 0.6572 |
| Gated, clean | 0.8322 | 67% | 0.6473 |
| Ungated, POISONED | 0.7922 | 57% | 0.5698 |
| Gated, POISONED | 0.8272 | 66% | 0.6090 |
| Skyline (oracle) | 0.9666 | 100% | -- |
<!-- /AUTOGEN:groupN -->

**Verdict:** the loop now runs on real, correlated user contexts and each redeploy is
gated by off-policy evaluation. Two wins. (1) With realistic recency-window
retraining, candidates are noisy -- and **the gate helps even in the clean case**
(67% vs 59% of skyline) by validating each candidate on a fresh **uniform-random
bucket** (Phase 8's unbiased log, used live) and only shipping true winners. (2) When
a logging bug ships a corrupt candidate, the insurance pays off: the ungated loop's
worst deployed value **craters to 0.5698**, while the gate rejects the poison and the
gated loop's worst stays **0.6090**. A safety gate is priced in the good case and
cashed in the bad. The min-propensity floor bounds importance weights so one rare
action can't blow up the estimate (Phase 11's variance guard). See
[`phase17.md`](phase17.md).

---

## What each phase actually bought

| Phase | Primary currency | Headline result | Accuracy delta |
|---|---|---|---|
| 0 Heuristic | a **baseline** | Recall@20 = 0.068 | (defines zero) |
| 1 Two-tower | a **pipeline** + real retrieval win | +83% recall, +116% coverage | **up** |
| 2 Feature store + LR | **correctness** (skew audit) + interpretability | weak cross feature hurt (-13% NDCG) | down |
| 3 Serving | **latency & robustness** (p50 4.2ms, fallback) | no accuracy change | flat |
| 4 Monitoring | **trust** (drift gate FAIL, as designed) | no accuracy change | flat |
| 5 A/B testing | **honesty** (don't ship the worse ranker) | significant -6% | down (correctly) |
| 6 Freshness | **fresh data** (streamed, no retrain) | +0.2%, not significant | flat |
| 7 Co-visitation | **task framing** (session signal) | +60% on session next-item | up (diff task) |
| 8 OPE | **causal truth** | naive metric was +100% biased | -- |
| 9 Off-policy learning | **causal learning** | unbiased-learned policy +87% true value | up (true value) |
| 10 Sequence model | **order modelling** (GRU4Rec) | beats popularity, LOSES to co-vis by ~15% | up vs pop, down vs covis |
| 11 Position debiasing | **causal labels** (IPW) | recovered ranking Spearman 0.86 -> 0.97 | up (ranking quality) |
| 12 Two-stage integration | **architecture** (retrieve->rank) | two-stage LOSES to two-tower alone (-19%) | down (weak stage-2 signal) |
| 13 Score-as-feature | **the right stage-2 signal** | +tt_score two-stage BEATS two-tower alone (+3.3%) | **up** |
| 14 Explore-and-learn | **unbiased data at the source** | Thompson -48% regret, 98% finds best, full support | up (regret + data quality) |
| 15 Contextual bandit | **personalized exploration** (LinUCB) | -92% regret vs context-free, 62% per-user-best | **up** |
| 16 Closing the loop | **the full cycle** (explore->learn->redeploy) | exploration 41% -> 96% of skyline; IPS a wash | **up** |
| 17 Real ctx + safety gate | **safe, real-context redeploys** | gate helps clean (67 vs 59%) & blocks a bug (worst 0.61 vs 0.57) | **up** |

**The lesson in one line:** Part I's complexity bought robustness, correctness, and
honest experimentation; Part II's causal evaluation revealed that the offline
numbers underneath all of it were still 2x biased. In production ML, trustworthy
evaluation is the whole game (Rules 8, 23, 36).

---

## Does complexity pay? (the honest curve)

The tempting story is "each phase adds sophistication, so accuracy climbs." **The
data says otherwise.** Sorting the *comparable* additions by whether they moved the
needle -- and remembering that only within-group numbers are comparable:

| Added complexity | Comparable delta | Paid off? |
|---|---|---|
| Two-tower retrieval (P1 vs P0) | Recall@20 +83%, coverage +116% | **Yes -- big** |
| Co-visitation for sessions (P7) | +60% vs popularity (session task) | **Yes** (different, easier task) |
| IPW debiasing of labels (P11) | ranking Spearman 0.86 -> 0.97 | **Yes** (simulation) |
| LR ranker w/ one cross feature (P2) | Recall@20 -8%, NDCG -13% | **No -- hurt** |
| Streaming freshness (P6) | +0.2%, not significant | **No -- flat** |
| GRU4Rec sequence model (P10) | -15% vs co-visitation | **No -- lost to a simpler model** |
| Two-stage retrieve->rank (P12) | -19% vs two-tower alone | **No -- weak stage-2 undid stage-1** |
| ...+ two-tower score as a feature (P13) | +3.3% vs two-tower alone | **Yes -- once stage-2 could see stage-1** |

**The shape of the curve is not monotonic.** Three of six sophistication upgrades
did nothing or actively hurt on this dataset. That is not a failure of the repo --
it is the finding:

- **Retrieval is where modelling capacity paid off** (P1). Once you have a decent
  candidate set, extra ranking/serving/freshness machinery bought *robustness,
  correctness, and honesty* -- not accuracy (P2-P6).
- **A strong baseline is hard to beat.** Popularity and co-visitation are brutal
  baselines on a small, dense catalog; the LR ranker and GRU4Rec both lost to them
  (P2, P10). "Simple" kept winning.
- **The regime decides.** These are KuaiRand results: dense feedback, ~7.5k items.
  On a sparse e-commerce log the same complexity often pays much more -- which is
  exactly why you *measure on your data* instead of trusting the ladder.
- **The real gains in Part II came from causality, not model size** (P8, P9, P11):
  fixing *what you optimize and how you evaluate* beat making the model fancier.

One-line takeaway: **complexity is a cost you pay up front for a benefit you must
measure -- often that benefit is robustness or honesty, not a higher accuracy
number, and sometimes it is negative.** Earn every layer (Rules 4, 14).

---

## Reproducing these numbers

Every number is produced by a `run.py` on real KuaiRand-Pure (seed 42, 80/20
temporal split). To regenerate:

```bash
cd phase0 && python run.py     # Group A: heuristic baseline (writes results.json)
cd phase1 && python run.py     # Group A: two-tower vs Phase 0
cd phase2 && python run.py     # Group B: LR ranker vs popularity + skew audit
cd phase5 && python run.py     # Group C: A/B replay (significant -9%)
cd phase6 && python run.py     # Group C: frozen vs fresh (not significant)
cd phase7 && python run.py     # Group D: session co-visitation benchmark
cd phase8 && python run.py     # Group E: off-policy evaluation (Part II)
```

Small run-to-run variation is expected (negative sampling, user subsampling); the
*conclusions* -- the Phase 1 win, the weak-feature regression, the significant A/B
loss, the flat freshness result, and the 2x OPE bias -- are stable.
