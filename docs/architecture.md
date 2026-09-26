# engram: design notes

*A hippocampal–neocortical memory architecture for long-horizon agents — what
each mechanism is for, what the measurements said, and which parts of the
biology turned out not to matter.*

---

## 1. The failure this starts from

An agent that runs for months accumulates episodes. The standard memory layer is
an embedding index: encode each episode, store the vector, retrieve top-k by
cosine. It fails in two ways that are worth separating, because they have
different causes and different fixes.

**Slot confusion.** Take 125 incident reports of the form *"On {day} the
{service} service suffered {fault} during the rollout."* Mean pairwise cosine
across the set is 0.529 under MiniLM and the maximum is 0.998
(`bench/incidents.py`). To the encoder these are
nearly the same sentence — and they should be, because they *are* nearly the
same sentence. The information distinguishing them is which values filled which
slots, and a single pooled vector does not represent that. It represents the
frame. Measured: dense cosine retrieves the right episode 4.0% of the time.

**Absent answers.** Ask *"which project does Priya usually work on?"*. No stored
episode contains the answer. It is a statistic over episodes. Retrieval cannot
return what was never written, however good the index.

The first is a *representation* problem. The second is a *storage architecture*
problem. Conflating them is why "add a better embedding model" never fixes
either.

## 2. Why expansion alone does not work

The obvious neuroscience-flavoured response to slot confusion is pattern
separation: the dentate gyrus expands entorhinal input into a population ~5×
larger while holding activity at 1–2% sparsity, and expansion plus
k-winners-take-all is a decorrelating transform. It works exactly as advertised.
Two inputs at cosine 0.77 produce codes overlapping at 0.220; at cosine 0.64,
0.140; at 0.41, 0.070 (`bench/ablate_dg.py`).

It does not fix retrieval, and the reason is the single most useful thing this
project turned up:

> **Expansion is injective.** The dentate gyrus can only separate what the input
> already distinguishes. If the decisive tokens contributed almost nothing to
> the pooled vector, they contribute almost nothing to its expansion.

Measured on the same 125 reports, the DG pathway finds the exact report 4.8%
of the time against cosine's 4.0%, and the right service-and-day group 24%
against 20%. An architecture that stacked more expansion here would have looked
sophisticated and done nothing.

There is a second half to this that v0.1 missed. With a *lexical* encoder
(`HashingEncoder`, a signed bag of hashed n-grams), dense cosine solves the same
task perfectly. Slot confusion is not a property of pooled vectors as such. It
is what a sentence encoder trained for semantic similarity does to the
lexical detail that distinguishes one slot filler from another.

## 3. Conjunctive coding

CA3's recurrent network implements conjunctive coding: cells fire for
*combinations* — this object, in this place, at this time — not for the elements
separately. A conjunction is a distinct addressable unit, so *auth-on-Monday* is
a different memory address from *auth-on-Tuesday* no matter how similar the
surface forms.

The implementation is a sparse binding code. Each content term is one unit, and
so is each unordered pair of content terms fewer than four content terms
apart. Pairs are hashed into a 2³² space. A query mentioning `auth` and
`Monday` activates the unit for that pair, which is active for exactly the
episodes containing both close together. Retrieval becomes weighted set
intersection instead of angle comparison. Each unit is weighted by its own
document frequency in the store, `ln(1 + (N − df + 0.5)/(df + 0.5))`, so a
rare conjunction of two common words counts as rare. The score is a weighted
cosine, computed for every episode by one sparse matrix-vector product.

Information retrieval knows this feature. It is the unordered-window term of
Metzler & Croft's sequential dependence model (`#uwN`). Their sentence-level
window of 8 terms was the starting point, and the LongMemEval-S dev split
chose 4 content terms.

On the 125 reports it finds the right service-and-day group every time, and
the exact report 20% of the time, with MRR 0.457. Both are the ceiling: the
queries name two of three attributes, so five reports are genuinely tied, and
0.457 is the expected MRR of a uniform draw among five.

It is deliberately not learned. It is a hash and a counter; it costs no
training; two episodes sharing a conjunction share a unit.

**What it is on real data.** On LongMemEval-S, removing this pathway turns
engram into the dense baseline to three decimals, and it accounts for all of
engram's gain over a flat vector store (+0.076 session R@5). Keeping terms and
dropping pairs changes nothing measurable there (+0.003, [−0.009, +0.015]).
On conversational memory, then, the pathway is a sparse lexical channel. It
restores the specificity the sentence encoder removed. The pair units earn
their place where a question repeats an episode's wording, which on the
synthetic literal queries is worth 0.035.

**Two bugs, fixed in v0.2.** v0.1 kept the *alphabetically* first 24 terms,
while documenting them as "the earliest, the topical ones". It also dropped
every one-character token, so `week 7` lost its 7 while `week 14` kept its 14.
See `docs/corrections.md`.

## 4. Complementary learning systems

One store cannot do both jobs. Learning fast enough to capture a single
experience in one shot requires large weight changes, and large weight changes
overwrite what is already there. Learning slowly enough to extract stable
structure means you cannot capture anything in one shot. McClelland,
McNaughton & O'Reilly's resolution is two stores with different learning rates
and a transfer process between them.

Here the fast store is the hippocampal trace set — one-shot writes, sparse codes,
conjunction codes, full episodic detail. The slow store is a set of schemas:
online-clustered centroids over replayed traces, which represent not the
episodes but the regularities across them.

Transfer happens during **sharp-wave ripples**. Three properties of biological
replay are load-bearing and all three are implemented, because dropping any one
changes the slow store's behaviour:

- **Prioritised** by prediction error, salience and recency, with an
  inverse-count term. Uniform replay spends the budget on episodes the model
  already predicts.
- **Sequential** — replay follows trajectories, pulling in temporal neighbours,
  which is what preserves order through the transfer. I.i.d. sampling hands the
  slow store a bag of disconnected facts.
- **Interleaved** with already-consolidated material. This is the one that is
  easy to omit and expensive to omit: a slow store fed only new material
  interferes with itself just as badly as a fast one, which defeats the entire
  purpose of having it.

### The regression worth recording

The first version released a hippocampal trace as soon as a schema absorbed it.
That looked tidy — consolidation frees space — and it destroyed the system. A
schema answers *what usually happens*; it can never answer *what happened on
Tuesday*. Coupling eviction to promotion meant episodic detail was deleted at the
exact moment generalisation began: textbook catastrophic forgetting, produced by
the architecture that exists to prevent it.

Eviction is now capacity-driven and completely decoupled from promotion.
Promotion marks a trace as safely represented elsewhere. Whether it still earns
its space is a separate question, asked only under real pressure, and an
unconsolidated trace is never evicted at all.

## 5. Retrieval: four pathways and a routing decision

Four pathways, fused after per-pathway min–max normalisation:

| pathway | mechanism | good at |
|---|---|---|
| semantic | dense cosine | paraphrase |
| conjunctive | weighted conjunction-set intersection | slot confusion, lexical specificity |
| temporal | theta-context similarity | explicit temporal scope |
| schema | neocortical centroid + mass | "usually", "typically" |

**Normalisation is not cosmetic.** Dense cosine and conjunction overlap live on
unrelated scales, and summing them raw lets whichever has the larger absolute
range dominate regardless of how discriminative it is. An earlier revision
summed raw scores, and the schema pathway could then be deleted without
moving a single number. Min–max, z-score and reciprocal-rank normalisation are
all implemented (`fusion=`). On the LongMemEval-S dev split min–max and z-score
were within 0.002 of each other, and rank fusion was clearly worse: it
discards how far apart two candidates are, which is the conjunctive
pathway's whole signal on the synthetic literal queries.

**Schema ranking uses mass.** Asked *"which project does Priya usually work
on"*, Priya's 12 schemas span centroid similarity 0.311–0.562
(`bench/schemas.py`, seed 0). The *most* similar one is an ingest schema with
mass 4.1, so similarity alone gives the wrong answer. Mass spans 2.4–53.2, and
the mass-weighted ranking returns an atlas schema; atlas is her modal project,
30 of 50 events. Similarity finds the topic; mass answers *usually*.

**Generalisation cues route the query.** A question containing *usually*,
*typically* or *generally* is not a question about an episode, and ranking
schemas against episodes by score is the wrong operation. The two stores
answer different questions, and the query says which one it is asking. This
is the CLS division of labour made explicit at the read path.

**Temporal context reinstatement** handles sequence. *"What did Priya work on
right after she reviewed the ledger diff?"* names one episode and asks for a
different one, so content matching retrieves the anchor. That is exactly what
every retrieval baseline does, and scored by key they all land near zero on
the temporal task. engram retrieves the anchor on content, reinstates its
encoding context to cue the adjacent slots, and suppresses the anchor.
Temporal accuracy by key: 0.009 without reinstatement, 0.981 with it (literal
queries), and 0.016 → 0.388 paraphrased.

The trigger is explicit adjacency: *right after*, *immediately before*, *what
came next*. v0.1 fired on any of *after, before, then, next, following*. On
LongMemEval-S those words occur in 28 of 470 answerable questions, almost all
of the form *"how many days before X did I Y"*, where X is evidence that
suppression would bury. Howard & Kahana's contiguity effect is about adjacent
items, and so is this mechanism.

## 6. What the ablations say

On the synthetic benchmark (MiniLM, 5 seeds, 2,314 queries, key-level scoring
where a query has a key), Δ overall with paired 95% intervals:

| removed | literal | paraphrased |
|---|---|---|
| conjunctive binding | −0.665 [−0.684, −0.646] | −0.090 [−0.105, −0.075] |
| temporal reinstatement | −0.315 [−0.334, −0.296] | −0.121 [−0.134, −0.107] |
| dense semantic | +0.002 [−0.001, +0.005] | −0.082 [−0.102, −0.063] |
| schema store | −0.005 [−0.009, −0.001] | −0.003 [−0.007, +0.001] |
| sleep / consolidation | −0.005 [−0.009, −0.001] | −0.003 [−0.008, +0.003] |
| theta context | exactly 0 | exactly 0 |
| neurogenesis | exactly 0 | exactly 0 |

On LongMemEval-S (MiniLM, 343 test questions), the conjunctive pathway is
worth −0.076 session R@5 when removed and the dense one −0.122. Reinstatement
is worth exactly zero, because no question carries an adjacency cue, and 40
cycles of sleep change one turn-level metric by 0.001.

Four readings are uncomfortable, and all of them are reported rather than
tuned away:

1. **The dense pathway earns its place only under paraphrase.** On literal
   synthetic queries it is neutral. On real conversations it is the larger of
   the two content pathways. It is a robustness mechanism.
2. **The schema store is a small-n mechanism with an encoder dependence.**
   Only 80 of 2,314 queries are abstraction queries. On that task alone the
   store is worth 0.713 → 0.863 with MiniLM, and with the lexical hashing
   encoder it is *harmful* (0.613 → 0.287). Lexical centroids merge different
   projects, and a schema answers with its first exemplar, which need not
   name the modal one.
3. **The memory-specific mechanisms do nothing on the public benchmarks.**
   LongMemEval and LoCoMo ask for evidence turns. Neither asks about what
   usually happens or what came next, which are what schemas and
   reinstatement are for. That is a statement about the benchmarks as much as
   about the mechanisms, and both halves belong in the write-up.
4. **Theta context and neurogenesis move nothing.** Neurogenesis is
   storage-side. Measured directly, it takes mean code overlap from 0.114 to
   0.021 and population coverage from 22% to 79%. Theta context activates only
   when the caller supplies an explicit scope, which no benchmark here does.
   Both are unmeasured by retrieval, not shown to be useless, and
   "unmeasured" is what should be written down.

## 7. Where CA3 actually earns its place

Not in ranking. In completion.

The attractor uses the modern Hopfield update, `ξ ← Xᵀ softmax(β·Xξ)`, which is
scaled dot-product attention with the stored set as both keys and values. That
is worth noticing: CA3 completion and a single attention head are the same
operation, differing only in whether the pattern set is learned or written.

Measured (`bench/ca3_capacity.py`): recovery of the correct episode from **90%
deletion** of its active units, 6 of 61 kept, at 0.993 with 400 stored
patterns, 0.987 with 1,000, 0.967 with 2,000 and 0.907 with 4,000. It degrades
slowly with store size. The fixed six-unit cue is shared with more patterns
as the store grows, and the degradation is steep once fewer than about three
units survive: at 2,000 patterns, 95% deleted gives 0.620, 97% gives 0.220
and 98% gives 0.007. v0.1 described the first curve as flat at 1.000, a claim
its own script does not reproduce.

β is the interesting control. Low β averages over many stored patterns and
returns a metastable blend that behaves like a prototype; high β drives the
update to the single nearest pattern and returns a verbatim episode. One knob,
the gist/verbatim axis.

## 8. Results

The README holds the full tables. In brief:

- **LongMemEval-S**, official retrieval protocol, 343 held-out questions,
  MiniLM. engram 0.767 session R@5 against 0.764 for a z-score BM25 + dense
  hybrid whose weight was tuned on the same dev split, a tie. On turn
  NDCG@10 it trails that hybrid by 0.019. It is +0.093 over BM25, +0.076 over
  dense and +0.120 over v0.1.
- **LoCoMo**, untouched by every decision, run once. engram is best with
  MiniLM (+0.025 recall@10 over the z-score hybrid, conversation-cluster
  interval [+0.005, +0.051]) and with bge-small (+0.036 over RRF), and ties
  dense retrieval with e5-small (+0.003).
- **Synthetic**, key-level. With literal queries it is at ceiling (0.987), so
  that setting is now a regression check. With paraphrased queries engram
  scores 0.282 against hybrid RAG's 0.160.

## 9. Limitations

- **The conjunction binder is lexical.** It matches surface terms, so
  paraphrase costs it: interference drops 1.000 → 0.223 on synthetic
  queries. Lemmatisation, or binding over extracted entities rather than raw
  tokens, is the obvious next step.
- **engram does not beat a well-tuned hybrid on LongMemEval-S.** Everything
  that makes it better than a vector store there, a tuned BM25 + dense hybrid
  also has. Its advantage on LoCoMo is real but smaller than the gap to single
  retrievers.
- **The public benchmarks do not exercise the memory-specific mechanisms.**
  A benchmark with sequence and generalisation questions over real
  conversations is what would test them. The synthetic one tests them, but
  only on templated text.
- **Schema formation is online clustering**, not a learned abstraction. A
  schema returns an exemplar rather than a statistic, and with a lexical
  encoder that fails outright (section 6).
- **No forgetting-curve validation.** Decay and eviction are implemented and
  bounded but not fitted against human retention data, which is the honest way
  to set those constants.

---

*Implementation: `src/engram/`. Every number here is produced by a script in
`bench/`; artefacts in `bench/results/`. What v0.1 got wrong:
`docs/corrections.md`. What was read for v0.2: `docs/reading.md`.*
