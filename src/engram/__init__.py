"""engram — a hippocampal-neocortical memory architecture for LLM agents.

Flat vector stores fail on long-horizon agent memory in two specific,
diagnosable ways: they cannot separate confusable near-duplicates, and they
cannot answer a question whose answer was never inside a single stored item.
Both failures have a well-studied biological solution, and this is an
implementation of it.

    from engram import EngramMemory

    mem = EngramMemory()
    mem.remember("deploy on tuesday failed on auth", session=0, salience=0.8)
    mem.remember("deploy on thursday failed on billing", session=0)
    mem.sleep(cycles=8)
    for r in mem.recall("which deploy broke auth?"):
        print(r.score, r.episode.text)
"""

from .binding import ConjunctiveBinder
from .ca1 import CA1
from .ca3 import CA3, CompletionResult
from .dentate import DentateGyrus, separation_gain
from .encoding import Encoder, HashingEncoder, SentenceTransformerEncoder, ThetaContext
from .memory import EngramConfig, EngramMemory
from .neocortex import Neocortex
from .replay import ReplayScheduler
from .types import ConsolidationEvent, Episode, Recall, Schema, Trace

__version__ = "0.1.0"

__all__ = [
    "EngramMemory",
    "EngramConfig",
    "DentateGyrus",
    "separation_gain",
    "CA3",
    "CompletionResult",
    "CA1",
    "ConjunctiveBinder",
    "Neocortex",
    "ReplayScheduler",
    "ThetaContext",
    "Encoder",
    "HashingEncoder",
    "SentenceTransformerEncoder",
    "Episode",
    "Trace",
    "Recall",
    "Schema",
    "ConsolidationEvent",
    "__version__",
]
