# Switching Datasets: Retail Rocket -> H&M

This project was built on Retail Rocket, then made **dataset-agnostic** so it can
run on H&M's *Personalized Fashion Recommendations* data with **zero changes to
any phase**. This doc explains how, what's different about H&M, and what to expect
when you run the analysis on it. It's also a small lesson in itself: *good data
abstractions make swapping the hardest-to-change thing (the data) easy.*

---

## Why H&M (recap)

Retail Rocket has anonymized categories, no prices/text, and no impression logs.
H&M is a real e-commerce dataset with **rich item metadata** (product type,
group, colour, department, garment group), **prices**, **customer attributes**
(age, club status), and real timestamps -- and the competition's public winning
solutions used exactly our **two-stage (candidate generation -> reranker)**
architecture. It's the closest public match to our original problem.

See [`dataset-selection.md`](dataset-selection.md) (if present) for the full
comparison of candidate datasets.

---

## How the swap works: a dataset dispatcher

The trick is that **every phase already speaks one canonical schema**:

```
events:          [timestamp_ms, user_id, event_type, item_id]
item_properties: [timestamp_ms, item_id, property, value]
```

`phase0/load_data.py` is now a thin **dispatcher** that picks a loader from
`phase0/data_sources/` based on an environment variable:

```
data_sources/
  retailrocket.py   # events.csv + item_properties_*.csv
  hm.py             # transactions_train.csv + articles.csv (+ synthetic generator)
```

```bash
# Retail Rocket (default) -- unchanged
cd phase0 && python run.py

# H&M -- same command, one env var
DATASET=hm python run.py

# H&M on a smaller machine: cap to the most recent N transactions
DATASET=hm HM_MAX_ROWS=3000000 python run.py
```

**No phase imports a dataset directly.** Adding a third dataset later = one new
module in `data_sources/` + one line in `_SOURCES`. That's the payoff of the
canonical-schema discipline we set up back in Phase 0.

---

## Getting the data

1. Accept the competition rules and download from Kaggle:
   <https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations>
2. Put these in `data/`:
   - `transactions_train.csv` (~3.5 GB)
   - `articles.csv`
   - `customers.csv`
3. Run any phase with `DATASET=hm`.

> `transactions_train.csv` is large. If memory is tight, use `HM_MAX_ROWS` to keep
> the most recent N rows (temporal structure preserved). Or point `DATA_DIR` at a
> sample directory.

---

## What's genuinely different about H&M (and how the port handles it)

| Aspect | Retail Rocket | H&M | How the port handles it |
|---|---|---|---|
| Event types | view / cart / purchase | **purchase only** | `event_type` is always `"purchase"`. Strong-event filters still work (a purchase is strong). |
| Timestamp granularity | milliseconds | **daily** (`t_dat`) | Same-day purchases share a timestamp; Phase 7 treats a customer's day as one **basket/session** -- a natural definition here. |
| Item metadata | versioned per timestamp | **static** (`articles.csv`) | Emitted as `item_properties` stamped at time 0 (known before every event). `product_type_name` -> `categoryid` so the feature store's cross feature works unchanged. |
| Availability / OOS | `available` property | none | Phase 3's out-of-stock filter finds nothing and simply filters nothing (no crash). |
| Scale | ~2.7M events | ~31M transactions | Fine for polars; two-tower/eval subsample as before; `HM_MAX_ROWS` if needed. |

### The honest implications for the analysis

Swapping data doesn't just change numbers -- it changes what some phases *mean*:

- **Phase 2's skew demo still works** (it's built on event-derived popularity, not
  on versioned item metadata), but H&M's static metadata means the *item-property*
  flavor of skew is less pronounced than Retail Rocket's. The popularity-count
  skew story is unchanged.
- **Purchase-only data** removes the view->purchase funnel. Phase 0's event
  weighting collapses to purchase popularity; that's expected.
- **Daily granularity** makes Phase 6 freshness coarser (a day, not 30 seconds) --
  still a valid demonstration, just at day resolution.
- **Phase 7 (co-visitation)** should shine: H&M baskets are strong co-purchase
  signal. This is likely your best-performing approach, consistent with the
  competition's winning co-visitation-heavy solutions.

---

## A note on the synthetic generator (important)

`data_sources/hm.py` ships a `generate_synthetic()` function used by the tests and
for smoke-running the pipeline **before** the real 3.5 GB download. It writes CSVs
with H&M's *schema* and some baked-in structure (baskets, category-correlated
co-purchases, skewed popularity).

**Do not treat synthetic numbers as findings.** They exist only to prove the
plumbing runs end-to-end and to keep CI honest without a giant download. On
synthetic data the phase *results* are artifacts of the generator (e.g. the LR
ranker may lose to popularity purely because of how the synthetic signal was
constructed). The real analysis -- the shareable learning experience -- comes from
running the phases on the **real** H&M CSVs you download.

Generate a synthetic sample yourself:

```python
from pathlib import Path
import sys; sys.path.insert(0, "phase0")
from data_sources import hm
hm.generate_synthetic(Path("/tmp/hm_synth"), n_customers=3000, n_articles=1000, n_days=90)
# then: cd phase0 && DATASET=hm DATA_DIR=/tmp/hm_synth python run.py
```

---

## Recommended path once your data lands

1. `DATASET=hm python run.py` in `phase0` -> establish the real baseline; sanity-
   check the summary (event counts, unique users/items, span).
2. Work up through the phases exactly as documented in `docs/phase*.md` -- the
   walkthroughs' *reasoning* transfers; only the numbers change.
3. **Regenerate the scoreboard** (`docs/results.md`) with the real H&M numbers,
   and update the phase docs' result tables. That regenerated, real-data scoreboard
   is the rich, shareable artifact you're after.
4. Expect Phase 7 (co-visitation) and richer ranker features (price/brand/age) to
   be where H&M pays off most -- exactly the levers Retail Rocket couldn't offer.
