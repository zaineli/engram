# engram

A hippocampal–neocortical memory architecture for long-horizon LLM agents,
measured on public conversational-memory benchmarks against the baselines a
reviewer would ask for.

```python
from engram import EngramMemory

mem = EngramMemory()
mem.remember("In week 7 on Monday, Priya reviewed the atlas runbook.", session=0)
mem.remember("In week 9 on Monday, Priya reviewed the ledger runbook.", session=0)
mem.sleep(cycles=20)                     # consolidation: where schemas come from

mem.recall("what did Priya review in week 7?")                    # conjunction-resolved
mem.recall("which project does Priya usually work on?")           # routed to the slow store
mem.recall("what did Priya do right after the atlas runbook?")    # context reinstatement
```

## The result, in one paragraph

On **LongMemEval-S** (official retrieval protocol, 343 held-out questions),
engram ties the strongest baseline, a BM25 + dense z-score hybrid whose mixing
weight was tuned on the same dev split. Its session recall@5 is 0.767 against
0.764 (+0.003, 95% CI [−0.023, +0.029]), and it trails that hybrid on turn
NDCG@10 (−0.019, [−0.034, −0.005]). It beats every single retriever by a wide
margin. On **LoCoMo**, which no decision in this release ever saw, it is the
best system with two of three encoders and ties dense retrieval with the
third. Most of what separates it from a vector store is one mechanism, the
conjunctive binding code, and that mechanism turns out to be a sparse lexical
channel. The memory-specific mechanisms (context reinstatement, schemas,
consolidation) do what they are designed to do on tasks built to isolate them.
On the public benchmarks they contribute nothing measurable, and this README
says so.

## Results

### LongMemEval-S — official protocol, held-out test split

LongMemEval (Wu et al., ICLR 2025), September 2025 cleaned release, the
retrieval-only setting. A haystack of about 48 sessions per question; keys are
user turns. Abstention questions and questions without user-side evidence are
excluded, as in the reference harness. Of the n = 419 that remain, a
hash-defined 20% dev split (n = 76) was used for every tuned choice, and the
rest (n = 343) is reported. `all-MiniLM-L6-v2`:

| system | session R@5 | session NDCG@5 | session R@10 | session NDCG@10 | turn R@10 | turn NDCG@10 |
|---|---|---|---|---|---|---|
| BM25, official tokeniser | 0.513 | 0.528 | 0.630 | 0.560 | 0.577 | 0.552 |
| BM25, normalised tokens | 0.673 | 0.659 | 0.758 | 0.683 | 0.703 | 0.675 |
| dense (flat vector store) | 0.691 | 0.611 | 0.825 | 0.644 | 0.703 | 0.622 |
| hybrid, reciprocal rank fusion | 0.743 | 0.694 | 0.875 | 0.722 | 0.767 | 0.706 |
| hybrid, z-score fusion (α tuned on dev) | 0.764 | **0.721** | **0.878** | **0.747** | **0.802** | **0.735** |
| engram v0.1 | 0.647 | 0.623 | 0.799 | 0.661 | 0.714 | 0.647 |
| **engram v0.2** | **0.767** | 0.699 | 0.869 | 0.728 | **0.802** | 0.716 |

R@k is `recall_all@k`, the paper's "Recall@k": every evidence item must be in
the top k. Paired differences, engram v0.2 minus each row, with a percentile
bootstrap interval over questions and a sign-flip randomisation test:

| vs | session R@5 | turn NDCG@10 |
|---|---|---|
| BM25, normalised | +0.093 [+0.055, +0.131], p < 0.001 | +0.041 [+0.022, +0.061], p < 0.001 |
| dense | +0.076 [+0.035, +0.117], p < 0.001 | +0.094 [+0.067, +0.121], p < 0.001 |
| hybrid, RRF | +0.023 [−0.009, +0.055], p = 0.20 | +0.010 [−0.008, +0.028], p = 0.27 |
| hybrid, z-score | +0.003 [−0.023, +0.029], p = 1.00 | −0.019 [−0.034, −0.005], p = 0.008 |
| engram v0.1 | +0.120 [+0.082, +0.157], p < 0.001 | +0.069 [+0.049, +0.089], p < 0.001 |

The same configuration, frozen on MiniLM's dev split, applied unchanged to two
other encoders:

| encoder | dense | hybrid, z-score | engram | engram − hybrid (session R@5) | engram − hybrid (turn NDCG@10) |
|---|---|---|---|---|---|
| `bge-small-en-v1.5` | 0.790 | 0.802 | **0.813** | +0.012 [−0.015, +0.038] | −0.014 [−0.030, +0.000] |
| `e5-small-v2` | 0.776 | **0.793** | 0.776 | −0.017 [−0.047, +0.012] | −0.020 [−0.034, −0.005] |

(session R@5; full tables in `bench/results/lme_test_*.json`.)

The paper's own retrieval tables are on LongMemEval-M, and they predate the
cleaned release, so none of its numbers is placed beside these. Every row
here is run by the same harness. The scorer and the BM25 baseline are
reimplementations, and `bench/check_reference.py` downloads the originals and
confirms they agree (300 random rankings: identical metrics; BM25 scores
identical to 1e-9).

### LoCoMo — the held-out check

LongMemEval's test split was scored more than once during development (see
[the tuning record](#how-the-configuration-was-chosen)). LoCoMo (Maharana et al.,
ACL 2024) was not used for any decision. It was run once, with the frozen
configuration and the frozen hybrid weight. The data: 10 conversations, 5,882
turns, and 1,535 questions in categories 1–4. Recall@10 is the official
fractional recall: the share of evidence turns in the top 10.

| encoder | BM25 | dense | hybrid, RRF | hybrid, z-score | **engram** | engram − best baseline, conversation-cluster CI |
|---|---|---|---|---|---|---|
| MiniLM | 0.543 | 0.507 | 0.586 | 0.614 | **0.640** | +0.025 [+0.005, +0.051] vs z-score |
| bge-small | 0.543 | 0.635 | 0.644 | 0.638 | **0.680** | +0.036 [+0.020, +0.054] vs RRF |
| e5-small | 0.543 | 0.690 | 0.656 | 0.636 | **0.693** | +0.003 [−0.015, +0.021] vs dense |

Questions in one conversation share a corpus. So each interval resamples whole
conversations, which is wider and more honest than resampling questions with
only ten clusters. With MiniLM, engram is best in all four categories
(`bench/results/locomo_minilm.json`).

### What carries it

Each LongMemEval row changes one thing from the frozen system (MiniLM, test
split, paired intervals):

| change | session R@5 | Δ | turn NDCG@10 | Δ |
|---|---|---|---|---|
| frozen system | 0.767 | | 0.716 | |
| no conjunctive pathway | 0.691 | −0.076 [−0.117, −0.035] | 0.622 | −0.093 [−0.121, −0.066] |
| no dense pathway | 0.644 | −0.122 [−0.160, −0.085] | 0.650 | −0.066 [−0.083, −0.048] |
| terms only, no pairs | 0.770 | +0.003 [−0.009, +0.015] | 0.718 | +0.003 [−0.007, +0.012] |
| all pairs (v0.1's binding) | 0.758 | −0.009 [−0.026, +0.009] | 0.714 | −0.002 [−0.012, +0.008] |
| unweighted units | 0.770 | +0.003 [−0.017, +0.023] | 0.712 | −0.004 [−0.013, +0.005] |
| BM25-weighted units | 0.781 | +0.015 [+0.003, +0.029] | 0.711 | −0.005 [−0.012, +0.002] |
| z-score fusion | 0.767 | +0.000 [−0.017, +0.017] | 0.711 | −0.004 [−0.012, +0.003] |
| rank fusion | 0.743 | −0.023 [−0.052, +0.006] | 0.697 | −0.018 [−0.031, −0.006] |
| no context reinstatement | 0.767 | exactly 0 | 0.716 | exactly 0 |
| after 40 cycles of sleep | 0.767 | exactly 0 | 0.715 | −0.001 [−0.001, +0.000] |

Four readings:

1. **The gain over a vector store is the conjunctive pathway.** Without it,
   engram *is* the dense baseline, to three decimals.
2. **On this benchmark the pathway works as a lexical channel.** Terms alone
   do as well as windowed pairs. Its value is in putting back the lexical
   detail a sentence encoder compresses away, which is also what the incident
   experiment below shows.
3. **BM25 unit weighting scored higher on test than the frozen choice**, and
   the dev split preferred IDF-cosine (0.845 against 0.841). The dev decision
   stands. Reporting the test-side row is the honest alternative to quietly
   adopting it.
4. **Reinstatement and consolidation move nothing here.** LongMemEval has no
   "what happened right after X" questions (0 of 470 match the adjacency
   cues). Schemas are not scored, because a schema is not an evidence turn.

### The synthetic benchmark: mechanisms in isolation

`bench/generate.py` builds per-person timelines with exact ground truth and
four tasks, each built so that one architectural failure causes it:
*episodic* (name two attributes, recover the episode), *interference*
(near-duplicates differing in one token), *abstraction* (a person's usual
project, which is in no single episode) and *temporal* (the event right after
a named one). Five seeds pooled: 2,314 queries, 80 of them abstraction.
MiniLM, the frozen configuration. The top result must be the right episode
(key-level). Abstraction, which has no key, is scored by answer.

| task | recency | BM25 | flat-vector | hybrid RAG | engram v0.1 | **engram v0.2** |
|---|---|---|---|---|---|---|
| **token-literal queries** | | | | | | |
| episodic | 0.000 | **1.000** | 0.044 | 0.281 | 0.651 | 0.995 |
| interference | 0.000 | **1.000** | 0.140 | 0.485 | 0.858 | **1.000** |
| abstraction | 0.150 | 0.637 | 0.562 | 0.613 | **0.875** | 0.863 |
| temporal | 0.001 | 0.003 | 0.015 | 0.011 | 0.428 | **0.981** |
| overall | 0.006 | 0.664 | 0.083 | 0.270 | 0.652 | **0.987** |
| **paraphrased queries** | | | | | | |
| episodic | 0.000 | 0.149 | 0.039 | 0.176 | 0.177 | **0.181** |
| interference | 0.000 | 0.144 | 0.060 | **0.249** | 0.228 | 0.223 |
| abstraction | 0.150 | 0.637 | 0.537 | 0.525 | **0.875** | 0.775 |
| temporal | 0.001 | 0.008 | 0.020 | 0.019 | 0.128 | **0.388** |
| overall | 0.006 | 0.119 | 0.057 | 0.160 | 0.201 | **0.282** |

engram v0.2 against the strongest baseline, paired over the same 2,314 queries:
+0.323 [+0.304, +0.343] against BM25 on literal queries, and +0.122 [+0.104,
+0.140] against hybrid RAG on paraphrased ones. The v0.1 column is the v0.1
source tree scored by this harness. v0.1 consolidation read the wall clock,
so its abstraction cells carry run-to-run noise.

Three things to take from it:

- **The literal setting no longer discriminates.** Once week numbers are coded
  and the anchor of a temporal query is found exactly, v0.2 is at ceiling on
  three tasks. It stays in the table as a regression check, not as evidence.
- **v0.1's headline did not survive correct scoring.** Scored by key, v0.1 on
  literal queries ties BM25 (0.652 against 0.664, −0.012 [−0.034, +0.011]).
  The old README's lead came from answer-level credit for wrong episodes; see
  [corrections](docs/corrections.md).
- **Reinstatement is the only mechanism that answers sequence questions.**
  Every retrieval baseline returns the anchor and scores about zero on
  temporal. Removing reinstatement takes engram from 0.981 to 0.009.

Ablation, Δ overall with paired 95% intervals (`bench/ablate.py`):

| removed | literal | paraphrased | note |
|---|---|---|---|
| conjunctive binding | −0.665 [−0.684, −0.646] | −0.090 [−0.105, −0.075] | carries both settings |
| temporal reinstatement | −0.315 [−0.334, −0.296] | −0.121 [−0.134, −0.107] | the whole temporal task |
| dense semantic | +0.002 [−0.001, +0.005] | −0.082 [−0.102, −0.063] | earns its place only under paraphrase |
| pairs (terms only) | −0.035 [−0.043, −0.028] | +0.016 [+0.006, +0.026] | binding matters when the tokens match |
| schema store | −0.005 [−0.009, −0.001] | −0.003 [−0.007, +0.001] | abstraction 0.863 → 0.713 (literal) |
| sleep | −0.005 [−0.009, −0.001] | −0.003 [−0.008, +0.003] | builds the schemas |
| theta context, neurogenesis | exactly 0 | exactly 0 | never activated / storage-side |

On paraphrased queries, the LongMemEval-frozen binder settings are *not* the
best ones here. Unweighted units (+0.021), all pairs (+0.017) and terms only
(+0.016) all score higher. The configuration was frozen on real data, not on
this benchmark, and it is reported as frozen.

### Why not just use a vector store

`bench/incidents.py`: 125 incident reports, every combination of 5 days × 5
services × 5 faults in one frame (*"On {day} the {service} service suffered
{fault} during the rollout."*), each queried as *"What went wrong with the
{service} service on {day}?"*. Mean pairwise cosine under MiniLM is **0.529**
and the maximum is **0.998**. Each pathway ranks alone:

| pathway | right service and day (ceiling 1.0) | exact report (ceiling 0.2) | MRR (ceiling 0.457) |
|---|---|---|---|
| dense cosine, MiniLM | 0.200 | 0.040 | 0.168 |
| dentate expansion of that vector | 0.240 | 0.048 | 0.147 |
| **conjunctive binding** | **1.000** | **0.200** | **0.457** |

The query names two of three slots, so five reports are genuinely tied and 0.2
is the ceiling on the exact report. Expansion does not help, because it can
only separate what the input vector already distinguishes. The finding that
reframes this is a different run: with the lexical `HashingEncoder`, dense
cosine is already at the ceiling (1.000 / 0.200 / 0.457). The slot confusion
is not a property of pooled vectors. It is what a *semantic* sentence encoder
does to lexical detail, and the conjunctive pathway is how engram gets that
detail back.

## How the configuration was chosen

Everything tunable is in `EngramConfig`. Four settings were chosen on the
LongMemEval-S dev split and frozen: `conj_window=4`, `conj_weighting="idf"`,
`fusion="minmax"` and `w_conjunctive=0.9`. The objective was the mean of the
four session metrics the reference harness prints. The record, all in
`bench/results/lme_tuning_minilm*.json`:

1. **r1**, 72 configurations. The optimum sat on two edges of the grid.
2. **r2**, 294 configurations, with both axes extended. The optimum was
   interior. The config was frozen and scored on test, where engram tied the
   z-score hybrid and its conjunctive pathway alone trailed BM25.
3. **r3**, 441 configurations, adding BM25 weighting of conjunction units in
   response. Dev still preferred r2's config, so nothing changed, but the test
   split had been seen before r3. That is why LoCoMo is reported as the
   untouched check.

The hybrid baseline's α was tuned on the same dev split over 0.0–1.0 and both
BM25 variants; it chose normalised BM25 with α = 0.5.

## Architecture

```
experience
    │
    ▼
┌──────────────────────────┐
│ entorhinal encoding      │  dense semantic vector + theta/position code
└─────────┬────────────────┘
          ├──────────────┬──────────────────┬─────────────────┐
          ▼              ▼                  ▼                 ▼
   ┌────────────┐  ┌───────────┐     ┌────────────┐    ┌────────────┐
   │ dentate    │  │ conjunct. │     │  theta     │    │   CA1      │
   │ k-sparse   │  │ binding   │     │  context   │    │ novelty /  │
   │ expansion  │  │ (what×    │     │            │    │ prediction │
   │            │  │  where×   │     │            │    │  error     │
   └─────┬──────┘  │  when)    │     └────────────┘    └─────┬──────┘
         ▼         └───────────┘                             │
   ┌────────────┐                                            │
   │    CA3     │  modern-Hopfield attractor                 │
   │ completion │  (= attention over stored patterns)        │
   └─────┬──────┘                                            │
         │                                                   ▼
         │            ┌────────────────────────────────────────────┐
         └───────────▶│ sharp-wave ripples: prioritised, sequential │
                      │ interleaved replay                          │
                      └──────────────────┬──────────────────────────┘
                                         ▼
                              ┌────────────────────┐
                              │ neocortex: schemas │  slow, semantic
                              └────────────────────┘
```

Four retrieval pathways, fused after per-pathway normalisation, each separately
ablatable by setting its weight to zero:

- **semantic**: cosine over dense embeddings. What a flat store consists of.
- **conjunctive**: terms, and pairs of terms fewer than four content words
  apart (the unordered-window feature of Metzler & Croft's sequential
  dependence model), hashed to units. Each unit is weighted by its own
  document frequency in the store and scored by weighted cosine through a
  sparse index.
- **temporal**: theta-context similarity. Opt-in; needs an explicit scope.
- **schema**: neocortical gist, routed to on generalisation cues ("usually").

Plus **temporal context reinstatement**. On an explicit adjacency cue ("right
after…", "what came next") it retrieves the anchor on content, reinstates the
anchor's encoding context to cue the neighbouring slots, and suppresses the
anchor itself.

### Storage-side measurements

These don't move retrieval numbers, and they are measured directly instead.

`bench/ablate_dg.py`, 600 confusable episodes, MiniLM:

| neurogenesis | mean code overlap | p95 | population coverage | usage Gini |
|---|---|---|---|---|
| 0.00 | 0.114 | 0.298 | 0.221 | 0.883 |
| 0.35 | 0.047 | 0.130 | 0.593 | 0.652 |
| **0.70** (default) | **0.021** | 0.052 | 0.792 | 0.384 |
| 1.00 | 0.018 | 0.052 | 0.841 | 0.305 |

Without novelty-biased excitability the code concentrates on 22% of the
population at Gini 0.88.

`bench/ca3_capacity.py`, attractor completion from 90% deletion of a code's
active units (6 of 61 kept):

| stored patterns | 400 | 1,000 | 2,000 | 4,000 |
|---|---|---|---|---|
| accuracy | 0.993 | 0.987 | 0.967 | 0.907 |

It degrades slowly with store size, and sharply once fewer than about three
units survive: at 2,000 patterns, 95% deleted gives 0.620, 97% gives 0.220 and
98% gives 0.007. v0.1 described the first row as flat at 1.000; the committed
script never showed that.

### Things that did not work

Kept in the tree, defaulted off or documented where they live:

- **The schema store with a lexical encoder.** With `HashingEncoder`, removing
  the schema store *raises* abstraction from 0.287 to 0.613 (n = 80). Lexical
  centroids merge different projects, and the exemplar a schema returns need
  not name the modal one. With MiniLM the store helps (0.713 → 0.863). A
  schema is an online cluster with no read-out of its own statistics, and that
  is the next thing to fix.
- **Rank fusion inside engram.** RRF discards how far apart two candidates
  were. It costs −0.018 turn NDCG@10 on LongMemEval and −0.449 on literal
  synthetic queries, where the conjunctive pathway's margin is the whole
  signal.
- **Lateral inhibition in the dentate gyrus.** Random sparse projections are
  already near-orthogonal: Gram off-diagonal 0.020 against activation σ 1.00,
  about 196× below the selection margin. It perturbs about 12% of selected
  units (Jaccard 0.88) and buys no separation (mean overlap 0.1277 → 0.1284).
  Relevant only if the projection is ever learned.
- **DG/CA3 as a ranking pathway.** 0.048 on the incident reports, no better
  than cosine. Kept for what it is good at: storage separation and fragment
  completion.
- **Eviction coupled to promotion.** Releasing a trace as soon as a schema
  absorbed it destroyed episodic detail the moment generalisation began.
  Eviction is capacity-driven, and an unconsolidated trace is never dropped.
  v0.1's passive decay broke that rule in the other direction; see the
  corrections.

## Install

```bash
uv sync                      # core: numpy, scipy
uv sync --extra bench        # adds sentence-transformers for real embeddings
```

`HashingEncoder` is the dependency-free default and runs the whole test suite
hermetically.

## Reproduce

```bash
uv run pytest -q                                              # 63 tests

# LongMemEval-S (downloads 277 MB; embeddings are cached in bench/.cache)
uv run --extra bench python bench/longmemeval.py download
uv run --extra bench python bench/longmemeval.py embed --encoder minilm   # also bge-small, e5-small
uv run --extra bench python bench/longmemeval.py tune  --encoder minilm   # dev split only
uv run --extra bench python bench/longmemeval.py run   --encoder minilm --split test --ablations
uv run --extra bench python bench/check_reference.py                      # scorer vs the originals

# LoCoMo (CC BY-NC 4.0: downloaded, never committed)
uv run --extra bench python bench/locomo.py download
uv run --extra bench python bench/locomo.py run --encoder minilm

# synthetic benchmark and mechanism probes
uv run --extra bench python bench/run.py --seeds 5 [--paraphrase]
uv run --extra bench python bench/ablate.py --seeds 5 [--paraphrase]
uv run --extra bench python bench/incidents.py
uv run --extra bench python bench/ablate_dg.py
uv run --extra bench python bench/ca3_capacity.py

# the v0.1 column, from the v0.1 tree
git worktree add ../engram-v0.1 cc2577a
uv run --extra bench python bench/run.py --seeds 5 --engram-src ../engram-v0.1/src --engram-label "v0.1 (cc2577a)"
```

Every number in this README comes from those scripts. The JSON and text
artefacts are in `bench/results/`, with per-question scores so that any
comparison can be re-derived.

## Further reading

- [`docs/corrections.md`](docs/corrections.md): what v0.1 got wrong, and
  whether each error moved a reported number.
- [`docs/reading.md`](docs/reading.md): the benchmarks, code and papers read
  for this release, and what each one changed.
- [`docs/architecture.md`](docs/architecture.md): the mechanisms, why each is
  there, and which parts of the biology turned out not to matter.

## References

- Marr (1971), *Simple memory: a theory for archicortex.*
- McClelland, McNaughton & O'Reilly (1995), *Why there are complementary learning systems.*
- Howard & Kahana (2002), *A distributed representation of temporal context.*
- Metzler & Croft (2005), *A Markov random field model for term dependencies.*
- Smucker, Allan & Carterette (2007), *A comparison of statistical significance tests for information retrieval evaluation.*
- Cormack, Clarke & Büttcher (2009), *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.*
- Michon et al. (2019), *Post-learning hippocampal replay selectively reinforces spatial memory.*
- Ramsauer et al. (2020), *Hopfield networks is all you need.*
- Maharana et al. (2024), *Evaluating very long-term conversational memory of LLM agents* (LoCoMo).
- Wu et al. (2025), *LongMemEval: benchmarking chat assistants on long-term interactive memory.*

MIT licensed.
