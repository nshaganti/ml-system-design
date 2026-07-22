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
  Recall@20        0.0500    0.0798    0.0594    0.0652          -18%
  MRR@20           0.0127    0.0203    0.0138    0.0149          -27%
  NDCG@20          0.0206    0.0331    0.0236    0.0256          -23%
```

**SASRec is the best neural model here -- it beats popularity (+30%) and edges out
GRU4Rec (+10%) -- but still loses to plain co-visitation (-18%).** Self-attention
extracts more session signal than the GRU, yet a count-based item-item baseline is
still the one to beat on this small, dense catalog.

> ### An honest correction (read this -- it's the real lesson of Phase 18)
>
> An earlier version of this page reported SASRec finishing **last, below popularity**
> (Recall@20 **0.0229**, "-71% vs co-visitation"), and confidently explained it away as
> "attention is data-hungry, wrong regime." **That was a bug, not a result.**
>
> `SASRec.recommend()` runs `forward` under `eval()` + `torch.no_grad()`, which trips
> PyTorch's *fused TransformerEncoder fast path*. On left-padded input, the leading
> all-PAD query rows are fully masked (causal + key-padding), and the fused kernel
> returns **all-NaN** for the whole row -- so `topk` saw NaN logits and returned items
> in index order. `run.py` backs off to popularity only when a session is *entirely*
> PAD, so every session shorter than `max_len=20` (i.e. essentially all of them) was
> scored on NaN garbage. Confirmed on the pinned `torch==2.8.0`.
>
> Why did it hide for so long? Because `forward()` in **train** mode is finite and the
> model learns fine -- so the shape/PAD/mask *contract* tests all passed. They never
> exercised the `eval()/no_grad` **inference** path that `recommend()` actually uses.
> A belated **learning test** (parity with Phase 10's GRU4Rec overfit test) ran that
> path for the first time, got index-order garbage from a model whose training loss was
> ~0, and exposed the NaN. The fix (`phase18/sasrec.py` disables the fused kernel so
> train and serve take the identical, correct math path -- Rule 32) lifted SASRec from
> 0.0229 to **0.0652**. Every number on this page is now from the *fixed* model.
>
> The meta-lesson is sharper than the original "honest negative": **a green test suite
> proved nothing about the path that ships. Test the inference path, in the mode it
> runs in.**

### Why co-visitation still wins (the finding that survived the fix)

Even with SASRec working correctly, the ranking is `co-vis > SASRec > GRU4Rec >
popularity`. The count-based baseline holds:

- **The regime favours co-occurrence.** SASRec shines on **large, sparse** catalogs
  with **long** histories, where attending to distant items directly pays off.
  KuaiRand-Pure is the opposite: ~7.5k items, dense feedback, short sessions, 300k
  pairs. There, item-item co-occurrence is a brutally strong, data-efficient baseline.
- **But attention did earn its keep over the GRU.** SASRec beating GRU4Rec (+10%) is
  the expected ordering finally showing up once the inference path is correct -- direct
  attention extracts a bit more from the same session than a single recurrent state.
- **Compute-bounded, not broken.** SASRec trained for 20 epochs (more than GRU4Rec
  needed) with its loss still descending; a larger budget would likely narrow the gap
  to co-visitation further -- but "keep tuning until the transformer wins" is exactly
  the anti-pattern this project argues against. We report the fixed-budget result.

## What Phase 18 taught us

1. **Test the path that ships, in the mode it ships in.** The bug survived a full
   contract-test suite because every test ran `forward()` in train mode; none ran the
   `eval()/no_grad` path `recommend()` uses. A model that trains to loss ~0 can still
   serve NaN. (See the correction box above.)
2. **A confident negative result can be a hidden bug.** "SASRec finishes last, below
   popularity" *felt* like a satisfying honest-negative -- which is exactly why it went
   unquestioned. The tidy story was the trap. Reproduce the surprising number through
   the real code path before you explain it.
3. **Newer and fancier is still not better by default.** Corrected, SASRec beats the
   GRU and popularity but *still* loses to counting co-occurrences on this data.
   Complexity has to earn its place on YOUR data, measured on a fixed protocol -- not
   assumed from a paper's leaderboard, and not conceded to a bug either.

Alongside Phases 2, 6, 10, and the naive Phase 12, this remains an honest result --
co-visitation wins -- but Phase 18's lasting contribution is the correction itself:
**trustworthy evaluation means the test harness has to touch the serving path.**
