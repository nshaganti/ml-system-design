# Data

This folder holds the dataset. **Nothing in here is committed** (see `.gitignore`) —
the CSVs are large and contain user interaction logs.

## KuaiRand-Pure (the dataset this repo runs on)

Download **KuaiRand-Pure** from the official source
([kuairand.com](https://kuairand.com) or the
[KuaiRand GitHub repo](https://github.com/chongminggao/KuaiRand)) and place it here
so the layout is:

```
data/KuaiRand-Pure/data/
  log_standard_4_08_to_4_21_pure.csv   # biased production-policy logs   -> Part I
  log_standard_4_22_to_5_08_pure.csv
  log_random_4_22_to_5_08_pure.csv     # UNIFORM-RANDOM policy logs       -> Part II (OPE)
  video_features_basic_pure.csv        # item metadata (tag -> categoryid)
  user_features_pure.csv               # user metadata (optional)
  video_features_statistic_pure.csv    # item stats (optional)
```

The loader (`phase0/data_sources/kuairand.py`) finds the CSVs automatically. Cap
rows on small machines with `KUAIRAND_MAX_ROWS=200000`.

> Why KuaiRand: it ships both a **biased** production log and a **uniform-random**
> exposure log. The random log's known logging propensities are what make honest
> off-policy evaluation (Part II) possible — the property most public datasets lack.

The test suite does **not** need this data: `tests/` runs on tiny in-memory frames.
