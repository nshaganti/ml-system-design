# Phase 10 -- Sequence Model: GRU4Rec (Part I extension)

> **Google's Rule 4:** *Keep the first model simple and get the infrastructure
> right.* By Phase 10 the infrastructure is right -- so we can finally ask whether a
> heavier model earns its keep. The answer here is a useful, humbling *no*.

Phase 7 measured session next-item prediction and found co-visitation beats
popularity by ~60%. It ended with a to-do: *add a sequence model and compare on
the same protocol.* Phase 10 does exactly that -- and keeps the comparison honest
by reusing Phase 7's split, sessions, test cases, and metrics verbatim.

---

## Why a sequence model at all

Co-visitation reads *pairwise* co-occurrence: "X and Y show up together." It throws
away order. A sequence model reads the *trajectory* of a session and predicts where
it's going -- so it can, in principle, tell "A then B then C" apart from
"C then B then A." GRU4Rec (Hidasi et al., 2016) is the canonical baseline: embed
each item, run a GRU over the session, predict the next item from the final hidden
state.

## Code tour

| File | Job |
|---|---|
| `phase10/model.py` | Pure GRU4Rec (`nn.Embedding` -> `nn.GRU` -> `nn.Linear`), PAD at index 0, plus a `recommend()` inference helper. Unit-tested, including an overfit test that proves it can learn a next-item pattern. |
| `phase10/seq_data.py` | Vocabulary + next-item training windows, reusing Phase 7's sessionization. `encode_context` mirrors serving: unknown items are dropped, recent items kept, left-padded. |
| `phase10/run.py` | Trains GRU4Rec, then grades popularity, co-visitation (Phase 7), and GRU4Rec on the IDENTICAL leave-one-out protocol via the shared `phase7/session_eval.py`. |

The eval protocol lives in `phase7/session_eval.py` so both phases import the exact
same `build_test_cases` / `reciprocal_rank` -- one obvious home, no copy-paste, no
`import run` name clash.

---

## Results

`cd phase10 && python run.py` (embedding/hidden 64, 12 epochs, 500k training
windows, full-softmax cross-entropy; CPU-friendly, ~90s):

```
  Metric       Popularity    Co-vis   GRU4Rec  GRU vs covis
  Recall@20        0.0498    0.0801    0.0691          -14%
  MRR@20           0.0128    0.0203    0.0172          -15%
  NDCG@20          0.0207    0.0332    0.0283          -15%
```

### Reading the numbers like an engineer

- **GRU4Rec beats popularity** (0.069 vs 0.050 recall): session order genuinely
  carries signal, and the model captures some of it.
- **GRU4Rec loses to plain co-visitation** by ~15%. On this dataset -- small catalog
  (~7.5k videos), strong pairwise co-occurrence -- item-kNN is a brutally strong,
  nearly-free baseline. The sequence model's extra capacity doesn't pay for itself
  in this regime.
- **This is not a bug; it's the point.** The training loss was still descending, so
  more epochs / a bigger model / negative sampling would narrow the gap -- but you'd
  be spending real complexity and compute to *maybe* catch a `defaultdict` of
  counters. Whether that trade is worth it is a business call, not a default.

### Where GRU4Rec *would* pull ahead

Sequence models tend to win when order matters more and co-occurrence matters less:
large catalogs, long and intentful sessions, strong sequential patterns (e.g.
next-episode, tutorials, funnels). KuaiRand-Pure's short-video sessions on a small
catalog are close to the worst case for them and the best case for co-visitation.

---

## What Phase 10 taught us

1. **A fancier model is not automatically better -- measure it on a fixed protocol.**
   This is Phase 2's negative result all over again, one rung up the sophistication
   ladder.
2. **Strong baselines are strong.** Co-visitation is cheap, interpretable, and here
   it beats a neural sequence model. "Simple" is a feature.
3. **The honest comparison is only possible because the harness was shared.** Same
   split, same sessions, same metrics -- reused, not re-implemented. If Phase 7 and
   Phase 10 each rolled their own eval, this table would be meaningless.

Natural next steps: SASRec (self-attention) on the same protocol, negative sampling
to speed and sharpen training, and blending sequence-model candidates into the
Phase 3 service candidate union rather than treating it as winner-take-all.
