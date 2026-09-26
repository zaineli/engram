# Corrections in v0.2

What v0.1 got wrong, found while putting engram on a public benchmark. Each
entry says what the code or the README did, whether a reported number moved,
and what replaced it. The v0.1 numbers quoted below were re-measured under the
v0.2 scorer by running the v0.1 source tree itself (`--engram-src`), so every
before-and-after pair is the same harness on the same queries.

## Bugs in the memory

**The binder kept the alphabetically first 24 terms.** `_content_terms`
returned a sorted list, and `encode` kept the first 24 of it. The docstring
said this kept "the earliest, which in practice are the topical ones". On the
templated benchmark no episode reaches 24 content terms, so the truncation
never fired there. On real turns it discarded most of the content. Now terms
keep reading order, pairs are windowed (below) and nothing is truncated.
Pinned by `test_long_episode_keeps_topic_terms_that_sort_late`.

**Single-character tokens were dropped, digits included.** `week 7` was coded
as `week`, while `week 14` kept its number. On v0.1's own benchmark, weeks 1–9
could not be told apart by the conjunctive pathway, and
`What did Priya review in week 7 on Monday?` scored all nine weeks at 0.414.
This did move reported numbers; it is part of the v0.1 → v0.2 gain on the
synthetic benchmark. Pinned by `test_single_digits_are_content_terms`.

**All pairs, in a 2¹⁶ code.** v0.1 bound every pair of (truncated) terms and
hashed them into 65,536 units. v0.2 binds pairs fewer than 4 content terms
apart (SDM's unordered window, with the size chosen on the LongMemEval dev
split) into a 2³² space. Each unit is weighted by its own document frequency.

**Passive decay deleted the traces it promised to keep.** `CA1.decay` dropped
traces below the strength floor only if they were *not* consolidated. That is
the inverse of the rule the README and `ReplayScheduler.utility` state:
nothing else holds an unconsolidated trace. At the default rate a trace needs
about 6,000 unrefreshed writes to cross the floor, and no benchmark here
writes that many, so no reported number moved. Pinned by
`test_passive_decay_never_drops_an_unconsolidated_trace`.

**Evicted traces stayed in CA3.** Capacity eviction released a trace from CA1
but left its attractor in CA3, where it could still capture a completion.
Every structure is now reconciled against CA1 after decay and after each
ripple. No benchmark sets a capacity, so no reported number moved. Pinned by
`test_released_traces_leave_every_structure`.

**A schema hit could be the reinstatement anchor, and it rehearsed a
stranger.** Schema results carry a placeholder key `(0, 0)`. In v0.1 they took
part in reinstatement, where a winning schema made the first episode of
session 0 the "anchor". They were also rehearsed on recall, which strengthened
whatever trace sat at `(0, 0)`. Both paths now exclude them. Pinned by
`test_schema_hits_neither_anchor_nor_rehearse`.

**Reinstatement fired on any temporal word.** The trigger matched `after`,
`before`, `then`, `next`, `following` and others. The anchor was then
multiplied by 0.15. On LongMemEval-S those words occur in 28 of 470 answerable
questions, nearly all of the form "how many days before X did I Y", where X is
evidence. The trigger is now explicit adjacency (`right after`, `immediately
before`, `what came next`, …). None of the 28 match it. Pinned by
`test_loose_temporal_word_keeps_the_anchor`.

**Consolidation depended on the wall clock.** Replay priority has a recency
term, and `sleep` always measured it from `time.time()`. Two identical
benchmark runs could therefore consolidate different schemas and report
different abstraction scores. v0.2's first ablation run showed this directly:
configurations identical to the default scored differently. `sleep(now=...)`
and explicit timestamps make a run a function of its seed. v0.1's reported
abstraction numbers carry this noise, and the v0.1 re-runs below still do,
since that tree cannot take a clock. Pinned by
`test_consolidation_is_reproducible_given_a_clock`.

**A docstring cited a script that was never written.** `replay.py` said
`bench/ablate_replay.py` measures what interleaving is worth. No such script
exists. The docstring now says that it is unmeasured.

## Bugs in the benchmark

**Answer-level scoring credited wrong episodes.** v0.1 scored a query correct
when the top result *mentioned* the right project. Each simulated person works
on a home project about 70% of the time. So a wrong episode by the right
person carries the right project about 40% of the time. On the temporal task,
the anchor, the one episode the question rules out, carries the successor's
project in 33–40% of queries. On seed 0 that rate is 0.333, which is exactly
the temporal score v0.1's README gave BM25, flat-vector and hybrid RAG. They
were being credited for returning the anchor. v0.2 scores the episodic,
interference and temporal tasks by key: the top result must be the episode the
query is about. Answer-level scoring is kept only for abstraction, which has
no key. Pinned by
`test_returning_the_anchor_earns_answer_credit_but_no_key_credit`.

**Evaluation mutated the memory.** Every recall rehearsed what it returned,
so each score depended on the queries before it. `recall(rehearse=False)` is
now used by every benchmark.

**One seed, no intervals.** The abstraction task has one query per person, so
16 per seed, and v0.1 reported it from one seed with no interval. v0.2 pools 5
seeds (80 abstraction queries) and reports paired bootstrap intervals and
randomisation tests.

## Claims that did not reproduce

**"engram leads in both settings."** Under key-level scoring, v0.1 on
token-literal queries scores 0.652 overall against BM25's 0.664. The difference
is −0.012, with interval [−0.034, +0.011] and p = 0.32: a tie, not a lead. The
paraphrased lead held: +0.041 over hybrid RAG, [+0.023, +0.059].

**The incident-report numbers had no script.** The README's 0.040 / 0.056 /
0.200 table and its 0.465 mean cosine could not be regenerated from the
repository. `bench/incidents.py` is that experiment written down. It gives
0.040 / 0.048 / 0.200 with MiniLM, a mean cosine of 0.529 and a maximum of
0.998. Those replace the old table.

**The dentate table did not come from the committed script.** Running v0.1's
own `bench/ablate_dg.py` gives different values from its README. At
neurogenesis 0, for example, it gives coverage 0.221 and Gini 0.883, where the
README said 0.162 and 0.939. The qualitative claim survives: without
neurogenesis the code collapses onto a minority of units. The table now shows
what the script prints.

**"CA3 completion is flat in store size."** The README gave 1.000 accuracy at
90% deletion for 400, 1,000, 2,000 and 4,000 stored patterns. The committed
script measures 0.993, 0.987, 0.967 and 0.907. Completion degrades slowly with
store size, as expected once the six surviving cue units start to be shared
with other patterns. It is not flat. The README and the `complete_partial`
docstring are corrected.
