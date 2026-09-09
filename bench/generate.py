"""Synthetic long-horizon agent-memory benchmark with exact ground truth.

Every query has exactly one correct answer. Ambiguity in the query set is
indistinguishable from failure in the system, and two earlier versions of this
generator were wrong in ways worth recording, because both produced numbers that
looked like architecture results and were not:

*Ambiguous queries.* Asking *"what went wrong with the auth service on Monday?"*
against a corpus with five matching episodes made 0.20 the ceiling and made
every system look broken. The primary key is now ``(person, week, day)`` and
queries name enough of it to be unique.

*Meaningless succession.* The temporal task used to ask what happened in the
next global slot, which interleaved sixteen unrelated actors: *"what happened
right after Mei signed off on the ledger runbook"* had the ground-truth answer
*"Tomas reviewed the quarry incident report"*. Well-defined, unanswerable, and
not what anyone means by the question. One session is now one person's
timeline, so the successor of an event is that person's next event.

Four tasks, chosen because they fail for different architectural reasons:

``episodic``
    Name two attributes, recover the third. Tests precise binding. A flat store
    does poorly not because it lacks the episode but because hundreds of
    episodes share the frame and it cannot tell which one filled which slot.

``interference``
    Restricted to queries whose corpus contains deliberately constructed
    near-duplicates differing in exactly one token. The pattern-separation
    stress case.

``abstraction``
    *"Which project does Priya usually work on?"* The answer is the mode across
    her episodes and appears in no single one. Retrieval alone cannot answer
    this; it needs a store that built something over the episodes.

``temporal``
    *"What did Priya do right after she signed off on the migration plan?"*
    Requires order, which neither content pathway encodes.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

__all__ = ["Corpus", "Query", "build_corpus"]

PEOPLE = [
    "Priya", "Marcus", "Yuki", "Dilnoza", "Tomas", "Aisha", "Ravi", "Freya",
    "Idris", "Nadia", "Bjorn", "Mei", "Oscar", "Leila", "Kwame", "Sana",
]
PROJECTS = ["orchestrator", "ledger", "ingest", "atlas", "beacon", "quarry"]
ARTIFACTS = [
    "migration plan", "rollout checklist", "incident report",
    "schema diff", "load test", "runbook",
]
PLACES = ["the Oakland office", "the Berlin hub", "a video call", "the Kyoto lab", "the Austin annex"]
ACTIONS = ["drafted", "reviewed", "rewrote", "signed off on", "rolled back", "benchmarked"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class Query:
    text: str
    answer_key: tuple[int, int]
    task: str
    #: Token that must appear in a correct answer. Scoring is answer-level, not
    #: key-level: a different episode carrying the right project has answered
    #: the question, and on the abstraction task no single episode is "the" key.
    answer_contains: str = ""


@dataclass
class Corpus:
    episodes: list[str] = field(default_factory=list)
    #: Parallel to ``episodes``. ``(session, index)`` where one session is one
    #: person's timeline and index is their event number within it.
    keys: list[tuple[int, int]] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)

    def by_task(self, task: str) -> list[Query]:
        return [q for q in self.queries if q.task == task]


def build_corpus(
    n_people: int = 16,
    events_per_person: int = 50,
    n_queries_per_task: int = 150,
    interference_rate: float = 0.25,
    seed: int = 0,
) -> Corpus:
    """Generate per-person timelines with a unique ``(person, week, day)`` key."""
    rng = random.Random(seed)
    c = Corpus()
    people = PEOPLE[:n_people]

    # Each person has a home project they work on ~70% of the time. The
    # abstraction task asks for exactly this, and it is never stated outright.
    home = {p: rng.choice(PROJECTS) for p in people}
    timelines: dict[int, list[int]] = {}
    person_projects: dict[str, Counter] = {p: Counter() for p in people}

    for pi, person in enumerate(people):
        used_days: set[tuple[int, str]] = set()
        week, idx = 1, 0
        timelines[pi] = []
        while idx < events_per_person:
            day = rng.choice(DAYS)
            if (week, day) in used_days:
                week += 1
                continue
            used_days.add((week, day))

            project = home[person] if rng.random() < 0.70 else rng.choice(PROJECTS)
            action, artifact, place = rng.choice(ACTIONS), rng.choice(ARTIFACTS), rng.choice(PLACES)
            text = f"In week {week} on {day}, {person} {action} the {project} {artifact} in {place}."

            g = len(c.episodes)
            c.episodes.append(text)
            c.keys.append((pi, idx))
            c.facts.append(dict(
                person=person, person_idx=pi, day=day, week=week, project=project,
                action=action, artifact=artifact, place=place, index=idx,
            ))
            timelines[pi].append(g)
            person_projects[person][project] += 1
            idx += 1

            # Interference: a near-duplicate differing in exactly one token
            # (the project), on a different day, later in the same timeline.
            if rng.random() < interference_rate and idx < events_per_person:
                for _ in range(16):
                    d2 = rng.choice(DAYS)
                    if (week, d2) not in used_days:
                        break
                else:
                    continue
                used_days.add((week, d2))
                alt = rng.choice([p for p in PROJECTS if p != project])
                t2 = f"In week {week} on {d2}, {person} {action} the {alt} {artifact} in {place}."
                g2 = len(c.episodes)
                c.episodes.append(t2)
                c.keys.append((pi, idx))
                c.facts.append(dict(
                    person=person, person_idx=pi, day=d2, week=week, project=alt,
                    action=action, artifact=artifact, place=place, index=idx,
                    near_duplicate_of=g,
                ))
                timelines[pi].append(g2)
                person_projects[person][alt] += 1
                idx += 1
            if idx % 7 == 0:
                week += 1

    # ---------------------------------------------------------------- queries
    n = len(c.episodes)
    dupes = [i for i, f in enumerate(c.facts) if "near_duplicate_of" in f]

    for i in rng.sample(range(n), min(n_queries_per_task, n)):
        f = c.facts[i]
        c.queries.append(Query(
            text=f"Which project did {f['person']} work on in week {f['week']} on {f['day']}?",
            answer_key=c.keys[i], task="episodic", answer_contains=f["project"],
        ))

    for i in rng.sample(dupes, min(n_queries_per_task, len(dupes))):
        f = c.facts[i]
        c.queries.append(Query(
            text=f"In week {f['week']} on {f['day']}, which project did {f['person']} {f['action']}?",
            answer_key=c.keys[i], task="interference", answer_contains=f["project"],
        ))

    for pi, person in enumerate(people):
        if not person_projects[person]:
            continue
        modal, _ = person_projects[person].most_common(1)[0]
        tgt = next((g for g in timelines[pi] if c.facts[g]["project"] == modal), timelines[pi][0])
        c.queries.append(Query(
            text=f"Which project does {person} usually work on?",
            answer_key=c.keys[tgt], task="abstraction", answer_contains=modal,
        ))

    # Temporal: the successor is the same person's next event.
    flat = [(pi, j) for pi, tl in timelines.items() for j in range(len(tl) - 1)]
    for pi, j in rng.sample(flat, min(n_queries_per_task, len(flat))):
        cur, nxt = timelines[pi][j], timelines[pi][j + 1]
        f, g = c.facts[cur], c.facts[nxt]
        c.queries.append(Query(
            text=(f"What did {f['person']} work on right after they {f['action']} "
                  f"the {f['project']} {f['artifact']} in week {f['week']}?"),
            answer_key=c.keys[nxt], task="temporal", answer_contains=g["project"],
        ))

    return c


if __name__ == "__main__":
    c = build_corpus()
    print(f"{len(c.episodes)} episodes, {len(c.queries)} queries")
    for t in ("episodic", "interference", "abstraction", "temporal"):
        qs = c.by_task(t)
        print(f"  {t:13s} n={len(qs):3d}  e.g. {qs[0].text!r} -> {qs[0].answer_contains!r}")
    print("\nPriya's first four events:")
    for e, k in list(zip(c.episodes, c.keys))[:60]:
        if k[0] == 0 and k[1] < 4:
            print(f"   {k}  {e}")
