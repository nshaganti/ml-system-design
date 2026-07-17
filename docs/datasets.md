# Datasets

This repo is **dataset-agnostic**. Every phase calls `load_events()` /
`load_item_properties()` and never knows which dataset is underneath. You choose
the dataset with the `DATASET` environment variable; the default needs no download.

```bash
cd phase0 && python run.py                 # synthetic (default, zero download)
DATASET=retailrocket python run.py         # a real e-commerce example
DATASET=hm         python run.py           # another real e-commerce example
```

## The canonical schema

Every adapter normalizes its raw data into two frames:

**events** — one row per interaction:

| column         | type | meaning                                             |
|----------------|------|-----------------------------------------------------|
| `timestamp_ms` | i64  | event time (epoch ms) — drives the temporal split   |
| `user_id`      | str  | who                                                 |
| `item_id`      | str  | what                                                |
| `event_type`   | str  | a **signal level**: `weak` / `medium` / `strong`    |

**item_properties** — a slowly-changing timeline of item attributes:

| column         | type | meaning                                             |
|----------------|------|-----------------------------------------------------|
| `timestamp_ms` | i64  | when this property value became true                |
| `item_id`      | str  | which item                                          |
| `property`     | str  | e.g. `categoryid`, `eligible`                        |
| `value`        | str  | the value at that time                              |

## The signal taxonomy (why not "purchase"?)

Different domains speak different verbs. Rather than weld the pipeline to
e-commerce, every adapter maps its native events into **three ordered signal
levels** (see `phase0/signals.py`, and Google's Rule 7 — encode domain knowledge
in one place):

| level    | intent          | e-commerce  | streaming        | social      | news        |
|----------|-----------------|-------------|------------------|-------------|-------------|
| `weak`   | exposure        | view        | thumbnail shown  | impression  | shown       |
| `medium` | engagement      | add-to-cart | click / dwell    | click       | click       |
| `strong` | target action   | purchase    | long-watch       | like/share  | read-through|

`POSITIVE_SIGNALS = (medium, strong)` are what count as positive labels;
`TARGET_SIGNAL = strong` is "the user definitively acted."

## Available datasets

### `synthetic` (default)

A seeded, in-memory generator (`phase0/data_sources/synthetic.py`). No files, no
downloads — the whole repo runs anywhere. It models the generic recommender
funnel (`weak → medium → strong`) with the structure the phases need: skewed item
popularity, per-user category preference, sessions (co-visitation), a realistic
funnel, a timeline, and an `eligible` property so the Phase 3 policy layer has
something to filter. Tune it with env vars:

```bash
SYNTH_USERS=5000 SYNTH_ITEMS=2000 SYNTH_DAYS=120 SYNTH_SEED=7 python run.py
```

>  Synthetic numbers are for **plumbing, not benchmarking**. They prove the
> pipeline runs and the invariants hold; they are not a measure of model quality.

### `retailrocket` (optional, real data)

A real e-commerce clickstream (view / addtocart / transaction → weak / medium /
strong). Put `events.csv` and `item_properties_*.csv` in `data/` and run with
`DATASET=retailrocket`. Kept as a worked example of mapping a real dataset into
the canonical taxonomy.

### `hm` (optional, real data)

The H&M dataset — purchases only, so every event maps to `strong`. Put
`transactions_train.csv` and `articles.csv` in `data/` and run with `DATASET=hm`.

## Adding your own dataset

1. Create `phase0/data_sources/<name>.py` with two functions:
   ```python
   def load_events(data_dir) -> pl.DataFrame:          # canonical events schema
   def load_item_properties(data_dir) -> pl.DataFrame: # canonical properties schema
   ```
   Map your native event types into `weak` / `medium` / `strong` inside this file
   — that is the *only* place that should know your dataset's vocabulary.
2. Register it in `_SOURCES` in `phase0/load_data.py`.
3. That's it — no phase code changes. (Rule 7 + programming to an interface.)
