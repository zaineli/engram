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
across the set is 0.465 and the maximum is 0.998. To the encoder these are
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
Two inputs at cosine 0.82 produce codes overlapping at 0.232; at cosine 0.63,
0.089.

It does not fix retrieval, and the reason is the single most useful thing this
project turned up:

> **Expansion is injective.** The dentate gyrus can only separate what the input
> already distinguishes. If the decisive tokens contributed almost nothing to
> the pooled vector, they contribute almost nothing to its expansion.

Measured on the same 125 reports, the DG pathway reaches 5.6% top-1 against
cosine's 4.0%. Within noise. An architecture that stacked more expansion here
would have looked sophisticated and done nothing.

## 3. Conjunctive coding

CA3's recurrent network implements conjunctive coding: cells fire for
*combinations* — this object, in this place, at this time — not for the elements
separately. A conjunction is a distinct addressable unit, so *auth-on-Monday* is
a different memory address from *auth-on-Tuesday* no matter how similar the
surface forms.

The implementation is a sparse binding code. Every unordered subset of content
terms up to order *k* hashes to one unit in a 2^16 binary space. A query
mentioning `auth` and `Monday` activates the unit for that pair, which is active
for exactly the episodes containing both. Retrieval becomes set intersection in
conjunction space rather than angle comparison in embedding space.

Top-1 goes 0.040 → **0.200**, MRR 0.165 → **0.457**. Both are the ceiling for
that query set: the queries name two of three attributes, so five episodes are
genuinely tied, and 0.457 is exactly the expected MRR of a uniform draw within a
tied group of five. The pathway narrows to the correct equivalence class and
then stops, because the question does not contain the answer.

It is deliberately not learned. It is a hash; it costs no training; two episodes
sharing a conjunction share a unit, with collisions bounded by the code width.

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
| conjunctive | conjunction-set intersection | slot confusion |
| temporal | theta-context similarity | explicit temporal scope |
| schema | neocortical centroid + mass | "usually", "typically" |

**Normalisation is not cosmetic.** Dense cosine spans ~0.45–0.55 across
candidates; conjunction overlap spans 0.0–0.3. Summing them raw lets whichever
has the larger absolute range dominate regardless of how discriminative it is.
Before normalisation the schema pathway could be deleted entirely without moving
a single number — it was scoring 0.30 against a hippocampal 0.57 and could never
rank first.

**Schema ranking uses mass, and mass is most of the signal.** Asked *"which
project does Priya usually work on"*, every schema about Priya has near-identical
centroid similarity (measured 0.483–0.505, a spread of 0.02). What separates
them is accumulated evidence: 47.7 for the modal project against 9.3 for the
rarest. Similarity finds the topic; mass answers *usually*.

**Generalisation cues route the query.** A question containing *usually*,
*typically*, *generally* is not a question about an episode, and ranking schemas
against episodes by score is the wrong operation. The two stores answer different
questions and the query says which one it is asking. This is the CLS division of
labour made explicit at the read path.

**Temporal context reinstatement** handles sequence. *"What did Priya work on
right after she reviewed the ledger diff?"* names one episode and asks for a
different one, so content matching retrieves the anchor and scores it as a miss —
which is precisely what every baseline does, all four landing on 0.333. The
system retrieves the anchor on content, reinstates its encoding context to cue
adjacent slots, and suppresses the anchor. Temporal accuracy 0.333 → **0.687**.

## 6. What the ablation says

Removing one mechanism at a time, Δ overall top-1:

| removed | literal | paraphrased |
|---|---|---|
| conjunctive binding | **−0.358** | **−0.084** |
| temporal reinstatement | **−0.114** | −0.017 |
| sleep / consolidation | −0.013 | −0.006 |
| schema store | −0.006 | −0.002 |
| semantic (dense) | **+0.009** | **−0.028** |
| theta context | 0.000 | 0.000 |
| neurogenesis | 0.000 | 0.000 |

Three readings are uncomfortable and all three are reported rather than tuned
away:

1. **The dense semantic pathway is harmful on token-literal queries** and
   load-bearing on paraphrased ones. It is a robustness mechanism, not an
   accuracy one. That is a reason to keep it and a reason not to weight it
   highly.
2. **Schema ablation costs almost nothing overall** — because only 16 of 466
   queries are abstraction queries. On that task alone it is −0.187, and sleep
   ablation costs the same, which is consistent: sleep is what builds schemas.
   An aggregate that hides a task-specific mechanism is an aggregate problem.
3. **Theta context and neurogenesis move nothing.** Neurogenesis is storage-side
   and the retrieval benchmark never fills the store enough for code collisions
   to matter; measured directly it takes mean code overlap from 0.232 to 0.019
   and population coverage from 16% to 74%. Theta context only activates when the
   caller supplies an explicit scope, which this benchmark never does. Both are
   unmeasured by this benchmark rather than useless — but "unmeasured" is what
   should be written down, not a plausible story.

## 7. Where CA3 actually earns its place

Not in ranking. In completion, where it is close to perfect.

The attractor uses the modern Hopfield update, `ξ ← Xᵀ softmax(β·Xξ)`, which is
scaled dot-product attention with the stored set as both keys and values — worth
noticing, because it means CA3 completion and a single attention head are the
same operation, differing only in whether the pattern set is learned or written.
The practical consequence is capacity exponential in dimension rather than the
classical ~0.14N.

Measured: recovery of the correct episode from **90% deletion** of its active
units at **100% accuracy**, flat from 400 to 4,000 stored patterns. It degrades
only when fewer than about three units survive — 95% → 0.933, 97% → 0.507,
98% → 0.013. The limit is absolute surviving units, not sparsity fraction or
store size.

β is the interesting control. Low β averages over many stored patterns and
returns a metastable blend that behaves like a prototype; high β drives the
update to the single nearest pattern and returns a verbatim episode. One knob,
the gist/verbatim axis.

## 8. Results

800 episodes, 466 queries, `all-MiniLM-L6-v2`, top-1 at k=5:

| | recency | bm25 | flat-vector | hybrid-rag | **engram** |
|---|---|---|---|---|---|
| token-literal | 0.174 | 0.768 | 0.330 | 0.468 | **0.790** |
| paraphrased | 0.174 | 0.354 | 0.309 | 0.376 | **0.448** |

BM25 scores 1.000 on episodic and interference in the literal setting and
deserves to — the query tokens appear verbatim in the target. It loses more than
half its accuracy when they do not. engram leads in both settings, which is the
actual claim: not dominance in one regime, but not collapsing when the regime
changes.

## 9. Limitations

- **The conjunction binder is lexical.** It matches surface terms, so paraphrase
  costs it (interference 0.867 → 0.320). Lemmatisation, or binding over extracted
  entities rather than raw tokens, is the obvious next step.
- **The benchmark is synthetic.** Templated episodes with exact ground truth,
  which is what makes the ablation interpretable and what makes the absolute
  numbers not transfer.
- **Schema formation is online clustering**, not a learned abstraction. It finds
  modes; it does not find structure the centroid cannot express.
- **No forgetting curve validation.** Decay and eviction are implemented and
  bounded but not fitted against human retention data, which is the honest way to
  set those constants.

---

*Implementation: `src/engram/`. Every number here is produced by a script in
`bench/`; JSON artefacts in `bench/results/`.*
