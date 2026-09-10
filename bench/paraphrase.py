"""Paraphrased queries: the realistic operating point.

On token-literal queries BM25 scores 1.000 on the episodic and interference
tasks, and it deserves to — *"week 14"*, *"Saturday"* and *"Priya"* appear
verbatim in the target episode, so exact matching is not a heuristic there, it
is the answer. Reporting only that setting would be measuring a string index and
calling it a memory system.

Real queries do not arrive tokenised against the store. A user asks *"what was
Priya focused on that Saturday in her fourteenth week?"*. The conjunction is
still fully specified and a person answers it instantly; the literal overlap is
mostly gone.

This module rewrites queries with surface-form changes only — synonyms, spelled
number words, reordered clauses. The referent never changes, so ground truth is
untouched and any drop is a robustness result rather than a harder question.
"""

from __future__ import annotations

import random
import re

from generate import Corpus, Query

__all__ = ["paraphrase_corpus"]

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


ORDINALS = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
            "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth",
            "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth"]
TENS_ORD = ["", "", "twentieth", "thirtieth", "fortieth", "fiftieth",
            "sixtieth", "seventieth", "eightieth", "ninetieth"]


def _spell(n: int) -> str:
    if n < 20:
        return ONES[n]
    t, o = divmod(n, 10)
    return TENS[t] + (f"-{ONES[o]}" if o else "")


def _ordinal(n: int) -> str:
    """Spelled ordinal. Naive suffixing produced 'threeth' and 'oneth'."""
    if n < 20:
        return ORDINALS[n]
    t, o = divmod(n, 10)
    return TENS_ORD[t] if o == 0 else f"{TENS[t]}-{ORDINALS[o]}"


PROJECT_PHRASE = ["workstream", "initiative", "effort", "codebase", "track"]
WORKED = ["was focused on", "spent time on", "was busy with", "put hours into", "was heads-down on"]
ARTIFACT_SYN = {
    "migration plan": "migration write-up",
    "rollout checklist": "release checklist",
    "incident report": "postmortem writeup",
    "schema diff": "schema change",
    "load test": "stress test",
    "runbook": "ops guide",
}
ACTION_SYN = {
    "drafted": "wrote up",
    "reviewed": "went through",
    "rewrote": "reworked",
    "signed off on": "approved",
    "rolled back": "reverted",
    "benchmarked": "profiled",
}
AFTER = ["right after", "immediately following", "straight after", "in the session after"]


def paraphrase_query(q: Query, rng: random.Random) -> Query:
    t = q.text

    if q.task in ("episodic", "interference"):
        t = re.sub(r"Which project did (\w+) work on in week (\d+) on (\w+)\?",
                   lambda m: f"What {rng.choice(PROJECT_PHRASE)} was {m.group(1)} "
                             f"{rng.choice(WORKED)[4:] if rng.random() < 0.5 else 'occupied with'} "
                             f"on the {m.group(3)} of their {_ordinal(int(m.group(2)))} week?", t)
        t = re.sub(r"In week (\d+) on (\w+), which project did (\w+) (.+)\?",
                   lambda m: f"On the {m.group(2)} of their {_ordinal(int(m.group(1)))} week, "
                             f"what did {m.group(3)} "
                             f"{ACTION_SYN.get(m.group(4), m.group(4))}?", t)
    elif q.task == "abstraction":
        t = re.sub(r"Which project does (\w+) usually work on\?",
                   lambda m: f"What does {m.group(1)} spend most of "
                             f"their time on, generally speaking?", t)
    elif q.task == "temporal":
        t = re.sub(r"What did (\w+) work on right after they (.+) the (\w+) (.+) in week (\d+)\?",
                   lambda m: f"{rng.choice(AFTER).capitalize()} {m.group(1)} "
                             f"{ACTION_SYN.get(m.group(2), m.group(2))} that {m.group(3)} "
                             f"{ARTIFACT_SYN.get(m.group(4), m.group(4))} in their "
                             f"{_ordinal(int(m.group(5)))} week, what came next for them?", t)

    return Query(text=t, answer_key=q.answer_key, task=q.task, answer_contains=q.answer_contains)


def paraphrase_corpus(corpus: Corpus, seed: int = 0) -> Corpus:
    """Return a copy of ``corpus`` with surface-rewritten queries."""
    rng = random.Random(seed)
    out = Corpus(
        episodes=corpus.episodes, keys=corpus.keys, facts=corpus.facts,
        queries=[paraphrase_query(q, rng) for q in corpus.queries],
    )
    return out


if __name__ == "__main__":
    from generate import build_corpus

    c = build_corpus(n_people=4, events_per_person=12)
    p = paraphrase_corpus(c)
    for a, b in list(zip(c.queries, p.queries))[:2] + list(zip(c.queries, p.queries))[-6:]:
        print(f"[{a.task}]\n  before: {a.text}\n  after : {b.text}")
