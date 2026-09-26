# What we read, and what we took

v0.2 came out of reading other people's benchmarks, code and papers against
this one. Each entry says what was read, what it changed here, and where the
change lives. Nothing was copied: where an external harness defines a protocol,
it was reimplemented from its description and then checked against the original
at run time (`bench/check_reference.py`), which downloads the original into a
temporary directory and never into this tree.

## Benchmarks and their protocols

**LongMemEval** — Wu et al., *LongMemEval: Benchmarking Chat Assistants on
Long-Term Interactive Memory*, ICLR 2025 (arXiv 2410.10813). Read: §3.3 on
metrics, §5 on retrieval, Tables 3 and 9; the reference harness
(`src/retrieval/run_retrieval.py`, `src/retrieval/eval_utils.py`,
`src/evaluation/print_retrieval_metrics.py`); the dataset card for the
September 2025 cleaned release (MIT).

- Took: the whole retrieval protocol. Keys are user turns. Evidence sessions
  relabel their non-evidence user turns `noans`. Abstention questions and
  questions without user-side evidence are excluded, which leaves n = 419 on S.
  The paper's "Recall@k" is `recall_all@k`. NDCG uses the harness's DCG
  (`rel_1 + Σ_{p≥2} rel_p / log2 p`). Session scores are derived from turn
  rankings by widening the cut until it spans k sessions. →
  `bench/longmemeval.py`.
- Took: its caveat that the published retrieval baselines are on
  LongMemEval-M, not S, and predate the cleaned release. So every baseline
  here is re-run in the same harness, and no number from the paper is put
  in a table beside ours.
- Did not take: its LLM-extracted "fact" keys and time-range query expansion.
  Both need an LLM in the loop, and the question here is what the memory
  architecture does without one.

**`rank_bm25`** — `BM25Okapi` (Apache-2.0), which the LongMemEval harness uses
as its BM25 baseline. Read: `_calc_idf` and `get_scores`.

- Took: the exact semantics of the official BM25 row. IDF is
  `ln(N − df + 0.5) − ln(df + 0.5)`; negative values are floored at 0.25 × the
  mean IDF; a repeated query token counts once per occurrence; tokenisation is
  a case-sensitive `split(" ")`. → `Okapi` and `tok_official` in
  `bench/longmemeval.py`, checked equal to 1e-9 on 60 haystacks.
- Took: the practice of reporting both that row and a normalised-token BM25.
  SelRoute (below) reports a substantial gap between SQLite FTS5 and the
  published BM25 row, which it attributes partly to differences between
  lexical engines. What counts as "BM25" is an implementation choice and has
  to be stated.

**LoCoMo** — Maharana et al., ACL 2024 (arXiv 2402.17753), with its repository
and the `locomo-audit` notes. Read: §4.1, Table 3, and the recall block of
`task_eval/evaluation.py`.

- Did not use it, for three reasons. It has 10 conversations, so any interval
  is a 10-cluster bootstrap. The audit lists 99 wrong gold answers and 57
  evidence-label issues. And the licence is CC BY-NC 4.0, which cannot sit in
  an MIT repository. It remains the natural second benchmark, downloaded at
  run time.

**LMEB** — arXiv 2603.12572 and its data release. Read: the paper summary and
the LongMemEval split's files.

- Did not use it for LongMemEval. Its split is built on M and keeps the
  single-session-assistant queries in the gold, so it is not the official
  protocol.

**SelRoute** — arXiv 2604.02431. Read: §4 to §6.

- Took a caution. Its LongMemEval-M averages run over all 500 questions,
  with the 30 abstentions counted in the denominator, so its numbers cannot be
  compared with ones computed by the official exclusion rule. Its routing
  table was derived from a 51-instance hard subset of the evaluation data, a
  limitation it states itself. Here every tuned choice is made on a
  hash-defined dev split and frozen before the test split is scored.

**Training-free lexical–dense fusion for conversational memory** — arXiv
2606.04194. Read: §3 to §4 on fusion, and §10 on LongMemEval-S.

- Took: z-score fusion, `α·z(s_lex) + (1 − α)·z(s_dense)`, as the strongest
  simple hybrid baseline, with α chosen on held-out data. It is `hybrid_z` in
  `bench/longmemeval.py`, and on LongMemEval-S it is the baseline engram ties
  rather than beats. `fusion="zscore"` is also offered inside engram
  (`engram.memory._normalise`). On the dev split it came within 0.002 of
  min-max, which is noise at n = 76, so the default stayed min-max.
- Took: its finding that LongMemEval-S is a lexical regime. BM25 reaches
  session R@5 0.948 on its 150-question subset, and fusion's margin over BM25
  there was not significant. This is why BM25 is treated as the bar here, and
  why turn-level metrics, where there is headroom, are reported next to the
  session-level ones.

## Retrieval ideas

**Sequential dependence model** — Metzler & Croft, *A Markov Random Field Model
for Term Dependencies*, SIGIR 2005. Read: §4, the window-size experiments and
Table 4.

- Took: unordered-window pair features `#uwN(q_i, q_j)`. engram's pair units
  were already conjunctions of two terms, and SDM says which conjunctions are
  worth coding: nearby ones. That replaced v0.1's
  all-pairs-of-the-first-24-alphabetical-terms, which was a bug. The paper's
  sentence-level window of 8 terms was the starting point. Here the window
  counts content terms, and the dev split chose 4 over 2, 3, 6, 8, 16 and
  unbounded. → `ConjunctiveBinder(window=4)`.
- Found: on LongMemEval-S the pairs add nothing measurable over terms alone
  (+0.003 session R@5 for terms only, interval [−0.009, +0.015]). On the
  synthetic benchmark's literal queries they are worth +0.035. Proximity
  features pay where the question repeats the episode's wording.
- Changed: SDM smooths each feature against collection statistics. Here a
  unit is weighted by its *own* document frequency in the store, so a pair is
  as rare as the pair actually is. → `ConjunctionIndex`.
- Tried: BM25 saturation and length normalisation over conjunction units
  (`conj_weighting="bm25"`), the same form as the `rank_bm25` baseline above.
  Dev preferred IDF-cosine (0.845 against 0.841). On test BM25 weighting
  scored higher (0.781 against 0.767 session R@5), and the dev decision was
  kept and the test number reported.

**Reciprocal rank fusion** — Cormack, Clarke & Büttcher, SIGIR 2009.

- Took: `1 / (60 + rank)` as the parameter-free fusion check. It is
  `hybrid_rrf` and `fusion="rrf"`.

**BM25 IDF** — the non-negative form `ln(1 + (N − df + 0.5)/(df + 0.5))`, as in
Lucene.

- Took: it for conjunction-unit weights. A unit present in most episodes gets a
  small positive weight rather than the negative one the Okapi form gives, and
  no floor is needed.

## Memory theory

**Temporal context model** — Howard & Kahana, *A distributed representation of
temporal context*, J. Math. Psych. 2002.

- Already the basis of context reinstatement. Re-read for what it does and
  does not predict. Its contiguity effect is about *adjacent* items: recalling
  one item cues its neighbours in the list. That is the evidence for narrowing
  v0.1's trigger (any of `after|before|then|next|…`) to explicit adjacency
  (`right after`, `immediately before`, `what came next`). LongMemEval's
  "how many days before X did I Y" relates events days and sessions apart,
  and it needs X as evidence. →
  `_ADJ_AFTER` / `_ADJ_BEFORE` in `engram/memory.py`.

## Statistics

**Smucker, Allan & Carterette**, *A comparison of statistical significance tests
for information retrieval evaluation*, CIKM 2007.

- Took: the paired randomisation (sign-flip) test as the significance test.
  The paper finds that the randomisation test, the bootstrap and the t-test
  largely agree, and it recommends retiring the sign and Wilcoxon tests. →
  `bench/stats.py`. Intervals are paired percentile bootstraps over
  questions.

## What we read and decided against

- **Learned sparse retrieval (SPLADE)** and **late interaction (ColBERT)**. Both
  are stronger lexical-semantic retrievers than anything here, and both are
  learned. engram's claim is about memory organisation on top of a fixed
  encoder, so the comparison that isolates it is engram against the same
  encoder's flat store and hybrids. A learned retriever would slot in as the
  encoder.
- **Cross-encoder reranking.** It is orthogonal to the memory and applies to
  every row equally; left out so that the table compares first-stage
  retrieval.
