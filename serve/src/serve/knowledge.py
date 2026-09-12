"""Retrieval store — facts live here, not in the weights.

The guide: "Do not fine-tune a model merely to memorize documents. Store those
documents in a retrieval system so they can be updated, removed and cited."

Scoring, and why it produces abstention
---------------------------------------
Score = (IDF mass of query terms found in the document)
      / (IDF mass of ALL content terms in the query)

A term that appears in no document gets maximum IDF. So for "the capital of
Zzyrgquist", the common word "capital" matches but "zzyrgquist" contributes a
large unmatched mass, dragging the ratio under the threshold and triggering
abstention. For "the capital of Karnataka", both terms are matched and the ratio
is high.

That is the whole mechanism: **a question about something we have no document for
cannot score highly, so the system says so instead of inventing an answer.**

Coverage is deliberately narrow and stated plainly: Indian states and capitals,
basic Indian geography, and a handful of world capitals. It is NOT a general
knowledge base, and it will abstain on most things - which is correct behaviour,
not a gap.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to",
    "for", "and", "or", "what", "which", "who", "where", "when", "how", "does",
    "do", "did", "it", "its", "that", "this", "with", "by", "from", "be", "as",
    "name", "tell", "me", "about", "city", "serves", "lies", "located", "please",
}


@dataclass
class Evidence:
    id: str
    text: str
    score: float = 0.0


# Small, verified corpus. Every entry is an uncontested fact; genuinely disputed
# items (such as Andhra Pradesh's working capital, which has been through
# multiple legal reversals) are deliberately absent rather than asserted.
CORPUS: Dict[str, str] = {
    "geo_0001": "New Delhi is the capital of India and the seat of the national government.",
    "geo_0002": "Hyderabad is the capital of Telangana, a state in southern India on the Deccan Plateau.",
    "geo_0003": "Telangana is a state in southern India, formed on 2 June 2014 when it separated from Andhra Pradesh.",
    "geo_0004": "Andhra Pradesh is a state on the southeastern coast of India, bordered to the east by the Bay of Bengal.",
    "geo_0005": "Visakhapatnam is the largest city in Andhra Pradesh and a major port on the Bay of Bengal.",
    "geo_0006": "Bengaluru, also called Bangalore, is the capital of the Indian state of Karnataka.",
    "geo_0007": "Chennai is the capital of the Indian state of Tamil Nadu, on the Coromandel Coast.",
    "geo_0008": "Thiruvananthapuram is the capital of the Indian state of Kerala.",
    "geo_0009": "Mumbai is the capital of Maharashtra and India's largest city by population.",
    "geo_0010": "Amaravati is the legislative capital of Andhra Pradesh.",
    "geo_0011": "The Bay of Bengal lies to the east of the Indian peninsula.",
    "geo_0012": "The Arabian Sea lies to the west of the Indian peninsula.",
    "wld_0001": "Tokyo is the capital of Japan.",
    "wld_0002": "Berlin is the capital of Germany.",
    "wld_0003": "Paris is the capital of France.",
    "wld_0004": "London is the capital of the United Kingdom.",
    "lit_0001": "William Shakespeare wrote the play Hamlet, first performed around 1600.",
}


# ---------------------------------------------------------------- live store
# If india_kb.json exists (built by `python -m serve.build_india_kb`) it is
# merged over the hand-written seed above. This is the whole point of keeping
# facts in retrieval: refreshing them is a file write, not a training run.
def _load_kb() -> int:
    import json
    # Resolved relative to THIS FILE, not the cwd. A cwd-relative path loads the
    # KB when run from serve/ and silently loads nothing under pytest or CI -
    # which would look like a retrieval regression rather than a missing file.
    here = Path(__file__).resolve()
    # Walk every ancestor rather than counting .parent hops by hand - counting
    # them wrong finds no file, loads nothing, and raises no error, so the only
    # symptom is retrieval quietly answering from the seed corpus.
    roots = [here.parent, *here.parents, Path.cwd()]
    for root in roots:
        p = root / "india_kb.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for k, v in data.get("docs", {}).items():
            CORPUS[k] = v["text"] if isinstance(v, dict) else v
            if isinstance(v, dict) and v.get("source"):
                SOURCES[k] = v["source"]
        return len(data.get("docs", {}))
    return 0


SOURCES: Dict[str, str] = {}
KB_LOADED = _load_kb()


def tokenize(text: str) -> List[str]:
    # Single characters are dropped: "India's" tokenizes to ["india", "s"], and
    # that orphan "s" is rare enough to earn a high IDF, so a long document that
    # happens to contain a stray letter scores as though it matched the topic.
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in STOPWORDS and len(t) > 1]


_DF: Dict[str, int] = {}
_N = 0
_AVG_LEN = 1.0


def _reindex() -> None:
    """Recompute document frequencies. Must run AFTER the KB merge, or IDF is
    computed over the seed corpus only and every loaded fact scores wrongly."""
    global _DF, _N, _AVG_LEN
    _DF = {}
    for _doc in CORPUS.values():
        for _t in set(tokenize(_doc)):
            _DF[_t] = _DF.get(_t, 0) + 1
    _N = len(CORPUS)
    lens = [len(tokenize(d)) for d in CORPUS.values()]
    _AVG_LEN = (sum(lens) / len(lens)) if lens else 1.0


_reindex()


def idf(term: str) -> float:
    # Unknown terms get the maximum weight. This is what makes an unanswerable
    # question score low rather than matching on its common words alone.
    return math.log((_N + 1) / (_DF.get(term, 0) + 1)) + 1.0


def score(query: str, doc: str) -> float:
    """Recall of the query's IDF mass. This is the ABSTENTION signal: it answers
    "how much of what was asked did we find?" and nothing else."""
    q_terms = tokenize(query)
    if not q_terms:
        return 0.0
    d_terms = set(tokenize(doc))
    total = sum(idf(t) for t in q_terms)
    hit = sum(idf(t) for t in q_terms if t in d_terms)
    return hit / total if total else 0.0


# Ranking must not reuse the abstention score. Recall has no length penalty, so
# a long passage matches more query terms by sheer surface area: asked for
# India's national capital, a 62-token Indian Railways summary beat the 7-token
# "New Delhi is the capital of India" 0.822 to 0.646, purely on length. Only
# documents LONGER than average are penalised - being concise is never punished.
LENGTH_PENALTY = 0.5


def rank_score(query: str, doc: str) -> float:
    base = score(query, doc)
    if base <= 0.0:
        return 0.0
    n = len(tokenize(doc))
    excess = max(0.0, (n - _AVG_LEN) / _AVG_LEN) if _AVG_LEN else 0.0
    return base / (1.0 + LENGTH_PENALTY * excess)


def retrieve(query: str, k: int = 3, min_score: float = 0.34) -> List[Evidence]:
    """Return evidence above the threshold, best first. Empty list means abstain."""
    scored = [Evidence(i, t, score(query, t)) for i, t in CORPUS.items()]
    # Sort on the length-normalised score, but gate on raw recall: ranking picks
    # WHICH document, the threshold decides WHETHER we have one at all.
    scored.sort(key=lambda e: -rank_score(query, e.text))
    return [e for e in scored[:k] if e.score >= min_score]


def explain(query: str, k: int = 3) -> List[Evidence]:
    """Same as retrieve but ignores the threshold - for tuning and debugging."""
    scored = [Evidence(i, t, score(query, t)) for i, t in CORPUS.items()]
    scored.sort(key=lambda e: -rank_score(query, e.text))
    return scored[:k]
