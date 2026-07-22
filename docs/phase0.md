# Phase 0 -- Launch Without ML

> **Google's Rule 1:** *The first model provides the biggest lift, so make it easy
> to get out the door.* Rule 1 goes further: **your first "model" shouldn't be a
> model at all.**

This walkthrough assumes you can write ML in a notebook but haven't shipped a
recommender to production. By the end you'll understand *why* we start with no ML,
what "temporal evaluation" means and why a random split lies to you, and how a
dead-simple popularity ranker sets the bar every future model must clear.

---

## The problem

A user opens the app. We have ~1.44M historical interactions (weak exposures,
medium clicks, strong actions like long-view / like / follow) from the
[KuaiRand-Pure](https://kuairand.com) short-video dataset. Show them 20 items that
maximize the chance they engage.

We have **no trained model, no features, no serving stack.** What do we ship on
day one?

## The design decision: ship a heuristic, not a model

Two reasons a heuristic comes first, and both are about *unblocking the future*:

1. **It collects training data.** You cannot train a personalized model until
   users have interacted with recommendations. The heuristic is what generates
   the clicks and actions your Phase 1 model will learn from.
2. **It establishes a baseline.** Every future model's worth is measured as
   *lift over what you already had*. Without a baseline, "Recall@20 = 0.07"
   is a number with no meaning. With one, it's "we beat / lost to the heuristic."

Our heuristic (`phase0/heuristic_ranker.py`) is two ideas stacked:

```
score(item) = weighted sum of recent interactions
              (strong=3, medium=2, weak=1, last 7 days)

if the user has history:
    keep only items in their top-3 categories   <- crude personalization
    drop items they already acted on (TARGET_SIGNAL)
```

> **Rule 7 in action:** *Turn heuristics into features.* The signal weights
> (strong > medium > weak) encode genuine domain knowledge -- a long-view/like is
> a far stronger signal than a fleeting exposure. We don't throw this away when ML
> arrives; the same taxonomy reappears in Phase 1's training labels.

## Code tour

| File | Job |
|---|---|
| `load_data.py` | Load the CSVs, rename to a **canonical schema** (`user_id`, `item_id`, `event_type`, `timestamp_ms`), validate loudly. |
| `heuristic_ranker.py` | `fit()` computes popularity scores; `recommend()` applies category filtering + already-acted exclusion. |
| `evaluate.py` | Temporal split + Recall@K + coverage + cold/warm breakdown. |
| `metrics.py` | Pure ranking metrics (NDCG, MAP, Precision) reused by every phase. |
| `run.py` | Wires it together, prints results, writes `results.json`. |

Two things worth internalizing here.

### 1. A canonical schema from day one

Raw KuaiRand calls the user `user_id` and the video `video_id`; our adapter maps
these to `user_id` / `item_id`. This normalization in `data_sources/kuairand.py`
looks trivial but it's a discipline: **every downstream stage speaks one
vocabulary.** When Phase 2's feature store joins events to features, it doesn't
care which dataset produced them.

### 2. Validate loudly (Rule 10 preview)

The loader doesn't silently accept whatever it's handed. The KuaiRand adapter maps
raw signals into the canonical `WEAK/MEDIUM/STRONG` taxonomy, drops rows with null
critical fields, and **prints the resulting signal mix on every load** so a skewed or
broken import is visible immediately:

```python
# phase0/data_sources/kuairand.py (load_events)
events = _to_signals(pl.concat(frames))          # canonical taxonomy mapping
counts = events["event_type"].value_counts()
print(f"[load_data] (kuairand) {len(events):,} standard-log events "
      f"(WEAK/MEDIUM/STRONG). Signal mix: {dict(...)}")
```

Unknown dataset names and empty split windows raise loudly too (`load_data.py`,
`heuristic_ranker.py`). In a notebook you'd notice bad data by eyeballing a `head()`.
In production nobody's watching, so the code has to shout. This is the seed of the
monitoring mindset that dominates later phases.

---

## The concept that trips up everyone: temporal evaluation

In a notebook you'd do `train_test_split(shuffle=True)`. **In production that is
a bug**, and it's the single most common way prototype metrics turn out to be
fiction.

Why? A random split lets a test event from *January* sit next to a training
event from *March*. Your model effectively "sees the future" -- it's trained on
data that, in reality, wouldn't exist yet when that January prediction was made.
Your offline metric looks great; production is worse and nobody knows why.

The fix (`temporal_split`) is boring and correct: **sort by time, train on the
first 80%, test on the last 20%.** The model only ever learns from the past, just
like it will at serving time.

```
Train: 1,149,287 events  (before cutoff)
Test:    287,322 events  (after cutoff)
```

> **Rule 33:** *Measure training/serving skew.* Temporal evaluation is your first
> defense. We reuse this exact split -- same cutoff, same seed -- in Phases 1 and
> 2 so every comparison is apples-to-apples.

---

## Results

Run `cd phase0 && python run.py`:

| Metric | Value | What it means |
|---|---|---|
| **Recall@20** | **0.0714** | Of every 100 items users engaged with, ~7 were in our top-20. |
| Warm-user recall | 0.0703 | Users *with* history -- see the twist below. |
| Cold-start recall | 0.1175 | New users get pure global popularity. |
| Catalog coverage | 0.0714 | We only ever recommend ~7% of the catalog. |

Three honest observations:

- **7% recall is a real baseline** -- recommendation is a needle-in-a-haystack
  problem. The number is only meaningful as a *baseline to beat*.
- **The twist: cold-start (0.118) BEATS warm (0.066).** New users get raw
  popularity; warm users get popularity *plus a top-3-category filter*. On
  KuaiRand that category filter **hurts** -- an early, honest hint that the
  category signal is weak here, which Phase 2 will confirm with a proper ranker.
  A "personalization" step is only worth it if it carries signal (Rule 17).
- **Coverage of 7% is the "popularity trap."** A popularity ranker shows the same
  hits to everyone; the long tail never surfaces. Improving coverage is a big part
  of what a real model should add -- and Phase 1's two-tower more than doubles it.

> **Gotcha we hit:** an earlier version divided coverage by each ranker's *own*
> catalog size, which made Phase 0 and Phase 1 use different denominators and
> produced a meaningless comparison. The fix: pass one shared `catalog_size`
> (the full recommendable universe) into `recall_at_k`. Lesson -- **a metric is
> only comparable if every model computes it against the same denominator.**

---

## What to carry into Phase 1

- You now have a **baseline (0.0714)** and an **evaluation harness** that every
  future model plugs into unchanged.
- You've internalized **temporal evaluation** -- the non-negotiable discipline.
- You've seen **coverage vs recall** as two axes, not one -- and an early sign
  that the category filter is a weak signal on this data.

Phase 1 asks: can a learned two-tower model beat this? (On KuaiRand: yes,
handily -- see [`phase1.md`](phase1.md).)
