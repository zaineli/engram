# engram

A hippocampal–neocortical memory architecture for long-horizon LLM agents.

Flat vector stores fail on agent memory in two specific, diagnosable ways. They
cannot separate confusable near-duplicates, and they cannot answer a question
whose answer was never inside any single stored item. Both failures have a
well-studied biological solution. This is an implementation of it, measured
against the baselines it claims to beat.

```python
from engram import EngramMemory

mem = EngramMemory()
mem.remember("In week 7 on Monday, Priya reviewed the atlas runbook.", session=0)
mem.remember("In week 9 on Monday, Priya reviewed the ledger runbook.", session=0)
mem.sleep(cycles=20)                     # consolidation: where schemas come from

mem.recall("what did Priya review in week 7?")       # episodic, conjunction-resolved
mem.recall("which project does Priya usually work on?")  # routed to the slow store
mem.recall("what did Priya do right after the atlas runbook?")  # context reinstatement
```

## Why not just use a vector store

Take 125 incident reports of the form *"On {day} the {service} service suffered
{fault} during the rollout"*. Mean pairwise cosine across the set is **0.465**,
maximum **0.998** — to a sentence encoder these are nearly the same sentence.
Ask *"What went wrong with the auth service on Monday?"*:

| pathway | top-1 |
|---|---|
| dense cosine (`all-MiniLM-L6-v2`) | 0.040 |
| dentate expansion of the same vector | 0.056 |
| **conjunctive binding** | **0.200** |

0.200 is the ceiling for that query set — it names two of three attributes, so
five episodes are genuinely tied, and the measured MRR of 0.457 is exactly the
expected value of a uniform draw within a tied group of five. The conjunction
pathway narrows to the correct equivalence class and then stops, because the
question does not contain the answer.

The lesson that shaped the architecture is the middle row. Expanding the
embedding does not help, because **expansion is injective**: the dentate gyrus
can only separate what the input already distinguishes. A pooled vector
represents the *frame*, not which value filled which slot.

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

- **semantic** — cosine over dense embeddings. What a flat store consists of.
- **conjunctive** — conjunction-code intersection. Resolves slot confusion.
- **temporal** — theta-context similarity. Opt-in; needs an explicit scope.
- **schema** — neocortical gist. Routed to on generalisation cues.

Plus **temporal context reinstatement**: a directional query ("right after…")
retrieves its anchor on content, reinstates the anchor's encoding context to cue
adjacent slots, and suppresses the anchor itself.

## Results

800 episodes across 16 per-person timelines, 466 queries, `all-MiniLM-L6-v2`.
Top-1 accuracy, `k=5`. Reproduce with `bench/run.py`.

**Token-literal queries** — the query shares surface tokens with the episode:

| task | recency | bm25 | flat-vector | hybrid-rag | **engram** |
|---|---|---|---|---|---|
| episodic | 0.160 | **1.000** | 0.440 | 0.553 | 0.807 |
| interference | 0.187 | **1.000** | 0.200 | 0.513 | 0.867 |
| abstraction | 0.188 | 0.500 | 0.500 | 0.500 | **0.875** |
| temporal | 0.173 | 0.333 | 0.333 | 0.333 | **0.687** |
| **overall** | 0.174 | 0.768 | 0.330 | 0.468 | **0.790** |

BM25 scores 1.000 on the first two tasks and deserves to: `week 14`, `Saturday`
and `Priya` appear verbatim in the target. That is a string index doing exactly
what a string index is good at, and reporting only this setting would be
measuring lexical overlap and calling it memory.

**Paraphrased queries** — same referents, surface forms rewritten:

| task | recency | bm25 | flat-vector | hybrid-rag | **engram** |
|---|---|---|---|---|---|
| episodic | 0.160 | 0.460 | 0.407 | 0.453 | **0.533** |
| interference | 0.187 | 0.253 | 0.133 | 0.313 | **0.320** |
| abstraction | 0.188 | 0.500 | 0.688 | 0.625 | **0.750** |
| temporal | 0.173 | 0.333 | 0.347 | 0.333 | **0.460** |
| **overall** | 0.174 | 0.354 | 0.309 | 0.376 | **0.448** |

BM25 loses more than half its accuracy when literal overlap goes. engram is the
only system that leads in both settings, which is the claim: not dominance in
any one regime, but not falling apart when the regime changes.

### Ablation

Every mechanism, removed one at a time (token-literal / paraphrased Δ overall):

| removed | Δ literal | Δ paraphrased | verdict |
|---|---|---|---|
| conjunctive binding | **−0.358** | **−0.084** | carries the system |
| temporal reinstatement | **−0.114** | −0.017 | the only thing answering sequence |
| schema store | −0.006 | −0.002 | −0.187 on abstraction alone |
| sleep / consolidation | −0.013 | −0.006 | builds the schemas |
| semantic (dense) | **+0.009** | **−0.028** | earns its place only under paraphrase |
| theta context | 0.000 | 0.000 | opt-in; needs an explicit scope |
| neurogenesis | 0.000 | 0.000 | storage-side, measured separately |

Two entries are negative results and are reported as such. The dense semantic
pathway is *harmful* on token-literal queries and load-bearing on paraphrased
ones — it is a robustness mechanism, not an accuracy one. Theta context and
neurogenesis do not move retrieval accuracy on this benchmark at all;
neurogenesis is a storage-side mechanism (below), and theta context only
activates when the caller supplies a temporal scope.

### Storage-side measurements

`bench/ablate_dg.py`, 600 confusable episodes:

| neurogenesis | mean code overlap | p95 | population coverage | usage Gini |
|---|---|---|---|---|
| 0.00 | 0.232 | 0.402 | 0.162 | 0.939 |
| 0.35 | 0.109 | 0.220 | 0.516 | 0.772 |
| **0.70** (default) | **0.019** | 0.070 | 0.744 | 0.444 |
| 1.00 | 0.015 | 0.070 | 0.805 | 0.345 |

Without novelty-biased excitability the code collapses onto 16% of the
population at Gini 0.94 and the expansion is mostly wasted.

`bench/ca3_capacity.py` — attractor completion, fraction of cue units deleted:

| stored patterns | accuracy @ 90% deleted |
|---|---|
| 400 | 1.000 |
| 1,000 | 1.000 |
| 2,000 | 1.000 |
| 4,000 | 1.000 |

Flat in store size, as the exponential-capacity result predicts. It breaks only
when fewer than about three units survive: 95% deletion → 0.933, 97% → 0.507,
98% → 0.013.

### Things that did not work

Kept in the tree, defaulted off, documented where they live:

- **Lateral inhibition in the dentate gyrus.** Random sparse projections are
  already near-orthogonal — measured Gram off-diagonal 0.020 against activation
  σ 1.00, about 200× below the selection margin. It perturbs ~12% of selected
  units (Jaccard 0.88) and buys no separation (0.2229 → 0.2255, marginally
  worse). Relevant only if the projection is ever learned.
- **DG/CA3 as a ranking pathway.** 0.056 top-1, no better than cosine.
  Retained for what it is actually good at: storage separation and fragment
  completion.
- **Eviction coupled to promotion.** An early version released the hippocampal
  trace as soon as a schema absorbed it, which destroyed episodic detail the
  moment generalisation began — the exact catastrophic forgetting the two-store
  design exists to prevent. Eviction is now capacity-driven and never drops an
  unconsolidated trace.

## Install

```bash
uv sync                      # core: numpy, scipy
uv sync --extra bench        # adds sentence-transformers for real embeddings
```

`HashingEncoder` is the dependency-free default and runs the whole test suite
hermetically. `SentenceTransformerEncoder` is what the benchmarks above use.

## Reproduce

```bash
uv run pytest -q                                         # 18 tests, ~0.4s
uv run --extra bench python bench/run.py                 # main table
uv run --extra bench python bench/run.py --paraphrase    # realistic setting
uv run --extra bench python bench/ablate.py              # pathway ablation
uv run --extra bench python bench/ablate_dg.py           # separation / neurogenesis
uv run --extra bench python bench/ca3_capacity.py        # attractor capacity
```

Every number in this README comes from those scripts; JSON artefacts are written
to `bench/results/`.

## Design notes

`docs/architecture.md` is the longer write-up: the mechanisms, why each one is
there, what the measurements said, and which parts of the biology turned out not
to matter.

## References

- Marr (1971), *Simple memory: a theory for archicortex.*
- McClelland, McNaughton & O'Reilly (1995), *Why there are complementary learning systems.*
- Ramsauer et al. (2020), *Hopfield networks is all you need.*
- Howard & Kahana (2002), *A distributed representation of temporal context.*
- Michon et al. (2019), *Post-learning hippocampal replay selectively reinforces spatial memory.*
- Nemhauser, Wolsey & Fisher (1978) — for the submodular selection argument reused elsewhere.

MIT licensed.
