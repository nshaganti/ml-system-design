# Phase 1 -- The First ML Model (Two-Tower Retrieval)

> **Google's Rule 4:** *Keep the first model simple and get the infrastructure
> right.* The model inside the pipeline will change every week. The pipeline is
> forever.

This is where we replace hand-written rules with *learned* item embeddings. On
KuaiRand it delivers a clean win over the heuristic -- but the more durable lesson
is *why* it wins here when the same model can lose on a sparser dataset. "We used
a neural net" is never the result; "we beat the baseline by X%, measured
temporally, and here's why" is.

---

## The problem: you can't score 50M items in 100ms

In the real system there are millions of items and a 100ms latency budget. You
physically cannot run a model on every item for every request. Every production
recommender splits into two stages:

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
training, and a large fraction of real sessions are new/anonymous. Mean-pooling
item embeddings gives you a vector for **any** user with history -- including ones
the model has never seen. This is how YouTube's and many production systems work.

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

Crucial data decision (`dataset.py`): **positives are POSITIVE signals**
(MEDIUM engagement OR STRONG target action -- e.g. click/dwell plus
long-view/like/follow) -- not mere WEAK exposures. Weak exposures carry little
intent; if you train on them as positives, the model just learns to reproduce a
popularity ranker. We also drop items with fewer than 3 positive interactions --
their embeddings can't be learned from 1-2 examples, and a near-random embedding
in your search index is worse than useless. (Exact vocab/triple counts print at
run time and are regenerated into the scoreboard.)

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

## The result: the two-tower beats the heuristic

Here's what actually happened when we ran it on KuaiRand:

<!-- AUTOGEN:groupA-phase1 -->
| Metric | Phase 0 (heuristic) | Phase 1 (two-tower) | Verdict |
|---|---|---|---|
| Recall@20 | 0.0695 | **0.1099** | **+58%** |
| Warm-user recall | 0.0681 | **0.1098** | **+61%** |
| Catalog coverage | 0.0679 | **0.1473** | **+117%** |
| Cold-start recall | 0.1157 | 0.1157 | ~wash (both served by the heuristic fallback) |
<!-- /AUTOGEN:groupA-phase1 -->
**The learned model wins where it can act** -- ~1.6x recall and ~2.2x catalog
coverage for warm users. Cold-start is a wash (both systems serve those users with
the *same* heuristic fallback, so the small difference is sampling/classification
noise, not the model). That coverage jump matters as much as the recall: a popularity
ranker recommends the same head items to everyone (~7% of the catalog); the two-tower
surfaces long-tail items (~15%), which is what drives discovery.

### Why it wins *here* (and when it wouldn't)

This is the opposite of what a sparse e-commerce log usually shows, and the reason
is the data, not the code:

- **KuaiRand's feedback is dense.** A third of all events are strong signals, so
  users have long, informative histories -- mean-pooling many item embeddings
  yields a rich user vector.
- **The catalog is small (~7.5k videos).** ID embeddings have enough interactions
  per item to actually learn.

Run the *same model* on a sparse dataset (short histories, millions of items, most
interactions weak) and it can easily **lose** to a strong popularity+category
heuristic -- mean-pooling 2-3 embeddings is a weak user representation, and a pure
ID model with no side features has little to work with. The lesson: model value is
a function of the data regime, and you only learn which regime you're in by
measuring against a real baseline.

### Two design choices that make the model learn

Even with favorable data, two details are doing quiet work (both baked into the
code):

- **Popularity-weighted negatives (`dataset.py`).** Sampling negatives uniformly
  at random gives trivially easy negatives ("is this popular item better than this
  obscure one?") and weak gradients. Sampling proportional to `popularity^0.75`
  (the word2vec trick) produces *hard* negatives -- "why this popular item and not
  that equally-popular one?" -- which is the signal that sharpens ranking.
- **Train/serve dot-product consistency (`index.py`).** The model trains with a
  raw dot product, where an item embedding's *magnitude* encodes popularity -- a
  useful signal. L2-normalizing to cosine at serving time would throw that away:
  training-serving skew (Rule 32). The index defaults to raw dot product to match
  training exactly.

And the thing Rule 4 actually cares about: Phase 1 delivers a working, tracked,
reproducible **pipeline** (MLflow runs, a search index, a fair eval harness). The
model inside it is now easy to improve.

> **Where the two-tower feeds forward:** it is Stage 1 for Phase 2's ranker. A
> good candidate generator hands the ranker a small, relevant set to work on --
> and the ranker is where fine-grained personalization lift shows up.

Continue to [`phase2.md`](phase2.md) -- the feature store and an honest negative
result about feature quality.
