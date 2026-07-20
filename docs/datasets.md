# Datasets

This repo runs on **one real dataset: KuaiRand-Pure**. The code is still written
dataset-agnostically — every phase calls `load_events()` / `load_item_properties()`
and never knows which dataset is underneath — but there is exactly one adapter
today, selected by default. The `DATASET` env var exists so a new dataset can be
dropped in later without touching a single phase.

```bash
cd phase0 && python run.py                 # kuairand (default)
KUAIRAND_MAX_ROWS=200000 python run.py     # cap rows on small machines
```

See [`data/README.md`](../data/README.md) for the download + folder layout.

## Why KuaiRand?

Most public recsys datasets give you a **biased** log: interactions collected by
whatever recommender was already in production. You can train on that, but you
cannot honestly *evaluate a new policy* from it, because the log only ever shows
you items the old policy chose to show (confounding).

KuaiRand is special: alongside the biased production log it ships a
**uniform-random exposure log** — a slice where items were shown at random,
independent of the user. Because the logging policy there is known (uniform), every
impression has a known propensity, and that is exactly what unbiased **off-policy
evaluation** (Part II / Phase 8) requires. That single property is why the whole
repo is built on it.

| File | Rows | Role |
|------|------|------|
| `log_standard_4_08_to_4_21_pure.csv` | ~1.14M | biased production log — **Part I** |
| `log_standard_4_22_to_5_08_pure.csv` | ~0.30M | more biased log |
| `log_random_4_22_to_5_08_pure.csv`   | ~1.19M | **uniform-random** log — **Part II (OPE)** |
| `video_features_basic_pure.csv`      | 7,584  | item metadata (`tag` → `categoryid`) |
| `user_features_pure.csv`             | —      | user metadata (optional) |

## The canonical schema

The adapter (`phase0/data_sources/kuairand.py`) normalizes the raw logs into two
frames:

**events** — one row per interaction:

| column         | type | meaning                                             |
|----------------|------|-----------------------------------------------------|
| `timestamp_ms` | i64  | event time (epoch ms) — drives the temporal split   |
| `user_id`      | str  | who                                                 |
| `item_id`      | str  | what (a video)                                      |
| `event_type`   | str  | a **signal level**: `weak` / `medium` / `strong`    |

**item_properties** — a slowly-changing timeline of item attributes:

| column         | type | meaning                                             |
|----------------|------|-----------------------------------------------------|
| `timestamp_ms` | i64  | when this property value became true                |
| `item_id`      | str  | which item                                          |
| `property`     | str  | today: `categoryid` (first video tag)               |
| `value`        | str  | the value at that time                              |

The random log adds a `propensity` column (uniform `1/N`) — see
[`off-policy-evaluation.md`](off-policy-evaluation.md).

## The signal taxonomy (why not "purchase" or "watch"?)

Different domains speak different verbs. Rather than weld the pipeline to one
domain, the adapter maps KuaiRand's native flags into **three ordered signal
levels** (see `phase0/signals.py`, and Google's Rule 7 — encode domain knowledge
in one place):

| level    | intent          | KuaiRand mapping                              | e-commerce  | news        |
|----------|-----------------|-----------------------------------------------|-------------|-------------|
| `weak`   | exposure        | shown, `is_click=0`                            | view        | shown       |
| `medium` | engagement      | `is_click=1`                                  | add-to-cart | click       |
| `strong` | target action   | `long_view` / `is_like` / `is_follow` / `is_forward` / `is_comment` | purchase | read-through |

`POSITIVE_SIGNALS = (medium, strong)` are what count as positive labels;
`TARGET_SIGNAL = strong` is "the user definitively acted."

On the standard log this yields roughly a **54% weak / 13% medium / 34% strong**
signal mix (short-video engagement is dense compared to e-commerce).

## Adding your own dataset

1. Create `phase0/data_sources/<name>.py` with:
   ```python
   def load_events(data_dir) -> pl.DataFrame:          # canonical events schema
   def load_item_properties(data_dir) -> pl.DataFrame: # canonical properties schema
   def load_random_log(data_dir) -> pl.DataFrame:      # optional: for Part II OPE
   ```
   Map your native event types into `weak` / `medium` / `strong` inside this file
   — that is the *only* place that should know your dataset's vocabulary.
2. Register it in `_SOURCES` in `phase0/load_data.py`.
3. That's it — no phase code changes. (Rule 7 + programming to an interface.)
