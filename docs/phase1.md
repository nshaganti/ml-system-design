# Phase 1 -- The First ML Model (Two-Tower Retrieval)

> **Google's Rule 4:** *Keep the first model simple and get the infrastructure
> right.* The model inside the pipeline will change every week. The pipeline is
> forever.

This is where we replace hand-written rules with *learned* item embeddings. It's
also where you'll learn a humbling production truth: **a real ML model does not
automatically beat a good heuristic**, and figuring out *why* is the actual job.

---

## The problem: you can't score 50M items in 100ms

In the real system there are 50M items and a 100ms latency budget. You physically
cannot run a model on every item for every request. Every production recommender
splits into two stages:

```
50M items ──► [Stage 1: Candidate Generation] ──► ~500 ──► [Stage 2: Ranking] ──► 20
              cheap, recall-oriented                        expensive, precision-oriented
```

Phase 1 builds **Stage 1**: turn the whole catalog into a few hundred plausible
candidates, fast. (Stage 2, the ranker, is [Phase 2](phase2.md).)

## The design decision: a two-tower model with no user table

The classic retrieval architecture is "two towers": a user tower and an item
tower, each producing a vector, scored by dot product. Our twist (and the
standard production choice) is in `phase1/two_tower.py`:

- **One embedding table -- items only.** There is *no* `user_id -> vector` table.
- **A user is the mean of their history's item embeddings.**

Why no user table? A user-ID embedding can only represent users seen during
training. 30% of real sessions are new/anonymous. Mean-pooling item embeddings
gives you a vector for **any** user with history -- including ones the model has
never seen. This is how YouTube's and many production systems work.

```python
# user vector = average of the items they interacted with
user_vec = (item_embeddings * mask).sum(dim=1) / n_valid_items
score(user, item) = user_vec · item_vec        # dot product
```

### The training signal: BPR on strong events only

We train with **Bayesian Personalized Ranking** (`bpr_loss`): for each
`(user, item_they_liked, random_item)` triple, push the liked item's score above
the random one.

```
loss = -mean( log( sigmoid( score(pos) - score(neg) ) ) )
```

Crucial data decision (`dataset.py`): **positives are cart-adds and purchases
only -- not views.** Views are 96.7% of events and carry weak intent. If you
train on views as positives, the model just learns to reproduce a popularity
ranker. We also drop items with fewer than 3 strong interactions -- their
embeddings can't be learned from 1-2 examples, and a near-random embedding in
your search index is worse than useless.

## Code tour

| File | Job |
|---|---|
| `dataset.py` | Build the item vocab, per-user histories, and BPR triples. |
| `two_tower.py` | The model, `bpr_loss`, embedding extraction. |
| `train.py` | Training loop with **MLflow** experiment tracking. |
| `index.py` | Brute-force nearest-neighbor search (stands in for Qdrant/Milvus). |
| `ranker.py` | Wraps model+index in the **same interface as Phase 0**, with heuristic fallback for cold-start. |
| `run.py` | Trains, indexes, evaluates, compares to Phase 0. |

> **The interface trick worth stealing:** `TwoTowerRanker.recommend()` has the
> exact same signature as `HeuristicRanker.recommend()`. That means Phase 0's
> `recall_at_k` evaluates both **without modification** -- same users, same seed,
> same denominator. When you compare models, eliminate every difference except
> the model itself.

---

## The honest result: our ML model *lost* to the heuristic

Here's what actually happened when we ran it:

| Metric | Phase 0 (heuristic) | Phase 1 (two-tower) | Verdict |
|---|---|---|---|
| Recall@20 | **0.0310** | 0.0228 | worse (regression) |
| Warm-user recall | **0.0474** | 0.0190 | much worse |
| Catalog coverage | 0.0140 | 0.0118 | worse |
| Cold-start recall | 0.0244 | 0.0244 | tie (both fall back to heuristic) |

**Our shiny neural model was worse than counting purchases.** This is not a bug
in the sense of a crash -- it's the normal, sobering reality of Phase 1. The doc
even predicts it: *"If Phase 1 doesn't beat this, the model isn't learning --
debug features first."* So we debugged.

### Debugging step 1: is it a coverage ceiling?

First hypothesis: maybe the two-tower's 8,219-item vocab is too small to even
contain the items people bought. We measured:

```
test-purchased unique items: 3,292
  of which in two-tower vocab: 1,292 (39.2%)
  -> recall CEILING for two-tower alone: 39.2%
```

39% is a ceiling, but it's *way* above our 2% recall. So coverage isn't the
bottleneck -- the model has plenty of reachable items, it's just **ranking them
badly.** On to the ranking quality itself.

### Debugging step 2: harder negatives

Our first version sampled negative items **uniformly at random**. The problem:
a random tail item is a trivially easy negative -- the model learns nothing from
"is this popular sneaker better than this obscure widget nobody wants?" The
signal that sharpens ranking is *"why did you pick this popular item and not that
equally-popular one?"*

Fix (`dataset.py`, the word2vec trick): sample negatives **proportional to
popularity^0.75**. Harder negatives, better gradients.

Result: warm-user recall **0.0150 → 0.0190 (+27%)**. Real improvement -- still
below the heuristic, but the model is learning more.

### Debugging step 3: a train/serve consistency bug

This one is subtle and important. The model **trains** with a raw dot product,
where an item embedding's *magnitude* naturally grows for popular items (a useful
signal). But the search index was **L2-normalizing** everything to cosine
similarity at serving time -- **throwing that magnitude away.**

That's training-serving skew (Rule 32: *use the same code/logic in training and
serving*). The model optimized one objective; we served a different one. We
changed `index.py` to default to raw dot product, matching training.

---

## Why didn't we "win"? (The real lesson)

After all three fixes, the two-tower still trails the heuristic on this dataset.
That is a **legitimate finding, not a failure**, and here's the intuition:

- Retail Rocket has **sparse, short user histories.** Mean-pooling 2-3 item
  embeddings is a weak user representation.
- The heuristic's **popularity + category filter is genuinely strong** for
  e-commerce -- most purchases *are* popular in-category items.
- A pure ID-embedding model with no side features (price, brand, recency,
  category) has little to work with.

Beating a strong heuristic takes side features, richer architectures, and
tuning -- real work, not a config flip. **The takeaway for a production engineer:
"we used a neural net" is not a result. "We beat the baseline by X%, measured
temporally, and here's why" is.**

What Phase 1 *did* deliver is the thing Rule 4 actually cares about: a working,
tracked, reproducible **pipeline** (MLflow runs, an index, a fair eval harness).
The model inside it is now easy to improve.

> **Where the two-tower shines anyway:** it feeds Phase 2. Even a mediocre
> candidate generator gives the ranker a small, relevant set to work on -- and
> the ranker is where personalization lift actually shows up.

Continue to [`phase2.md`](phase2.md) -- the feature store and the ranker that
finally beats popularity.
