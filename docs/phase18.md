# Phase 18 -- Does Self-Attention Beat Co-Visitation? (SASRec)

> **Google's Rule 4:** *Keep the first model simple* -- and keep proving the fancy one
> earns its place. Phase 10 found GRU4Rec *lost* to co-visitation on KuaiRand. Phase
> 18 asks the natural follow-up: does **self-attention** (SASRec) close the gap?

Phase 10 left an honest loose end. A GRU squeezes a whole session through one
recurrent state; **SASRec** (Kang & McAuley, 2018) can attend to any earlier item
directly, which is why it's the go-to sequential recommender in the literature. If
any sequence model beats co-visitation here, it should be this one. So we measured it
-- on the *same* protocol, split, and test cases as Phases 7 and 10, with both neural
models trained on the identical pairs and epochs. Only the model changes.

## The model

`phase18/sasrec.py` -- item embedding + a learned **recency-based** positional
embedding, a stack of **causal** self-attention blocks (each position attends only to
itself and earlier items), predicting the next item from the most-recent position.
Index 0 is PAD (frozen to zero, masked out of attention). It reuses Phase 10's
`seq_data.py` for vocab/encoding and Phase 7's eval harness, so "a session" means
exactly the same thing across all three phases (DRY, and what makes the comparison
fair).

## Results

`cd phase18 && python run.py` (7,540-item vocab, 300k training pairs, 20 epochs, same
leave-one-out protocol as Phases 7/10):

```
  Metric       Popularity    Co-vis   GRU4Rec    SASRec   SAS vs covis
  Recall@20        0.0495    0.0802    0.0617    0.0229          -71%
  MRR@20           0.0123    0.0200    0.0143    0.0035          -83%
  NDCG@20          0.0203    0.0330    0.0243    0.0076          -77%
```

**SASRec finishes last -- below even popularity.** That's the honest result, and it
is *not* a broken model.

### Why this is a real finding, not a bug

- **The model is sound.** In a smoke test it learns a trivial "next = f(last)" pattern
  to 100% training accuracy in seconds. The architecture, masking, and training loop
  all work.
- **It was not under-resourced relative to the winner.** SASRec trained for *20
  epochs* -- more than GRU4Rec needed -- and its loss was *still descending* at the
  end. It is data/compute-hungry, and we gave it a practical CPU budget, not an
  infinite one.
- **The regime is wrong for attention.** SASRec shines on **large, sparse** catalogs
  with **long** histories, where reading distant items directly pays off.
  KuaiRand-Pure is the opposite: ~7.5k items, dense feedback, short sessions, 300k
  pairs. In that regime a 7,540-way softmax starves a transformer, while
  co-visitation's item-item co-occurrence is a brutally strong, data-efficient
  baseline and GRU4Rec's lighter inductive bias extracts more from the same data.

### The debugging story (an honest process note)

The first SASRec run scored *below popularity* and looked broken. Rather than ship it
or blindly tune, the fix was diagnosis: a smoke test proved the model could learn, so
the problem was training, not code. An `embedding * sqrt(d)` scaling I had added
actually *hurt* and was removed; positions were switched to recency-based; the budget
was raised to 20 epochs. The result improved but the verdict held. **The point of the
exercise was never to make attention win -- it was to measure it honestly.** Forcing a
"SASRec wins" conclusion by endless tuning would have been the exact anti-pattern this
whole project argues against.

## What Phase 18 taught us

1. **Newer and fancier is not better by default.** SASRec is state-of-the-art on the
   right data; on the wrong data it loses to counting co-occurrences.
2. **Match the model to the data regime.** Attention buys long-range, sparse-catalog
   power you don't need when the catalog is small and feedback is dense.
3. **Diagnose before you tune.** A 3-second smoke test separated "broken" from
   "underfit" and stopped a wild-goose chase.

This is the fourth honest negative in the project (after Phases 2, 6, 10, and the
naive Phase 12), and together they are its spine: **complexity has to earn its place
on YOUR data, measured on a fixed protocol -- not assumed from a paper's leaderboard.**
