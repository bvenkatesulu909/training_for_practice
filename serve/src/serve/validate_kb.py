"""Check a built knowledge base for claims that contradict each other.

Why this exists
---------------
The first India build put both of these in the store:

    "Bengaluru, also called Bangalore, is the capital of ... Karnataka."
    "Belgaum is the capital of Karnataka, a state of India."

Retrieval cannot resolve that. Whichever document happens to rank higher becomes
the answer, so the system is right or wrong by accident - and the frozen suite
still reported 100%, because it only asks about Karnataka once and the correct
document happened to win on length.

A knowledge base is only as trustworthy as its internal consistency, and nothing
in a retrieval score checks that. This does.

    python -m serve.validate_kb india_kb.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# "X is the capital of Y" -> (X, Y). The subject may be a comma-qualified list
# ("Bengaluru, also called Bangalore"), so the subject is taken up to the first
# comma and the object up to the first comma or period.
CAPITAL_RE = re.compile(
    r"^(?P<cap>[^,.]+?)(?:,[^.]*?)? is the capital of (?:the Indian state of )?"
    r"(?P<place>[^,.]+)",
    re.I,
)

# Districts carry exactly the same hazard one level down: two headquarters for
# one district is as unresolvable as two capitals for one state.
HQ_RE = re.compile(
    r"^(?P<hq>[^,.]+?) is the administrative headquarters of (?P<place>[^,]+)",
    re.I,
)


def _extract(docs: Dict[str, str], rx, subject: str) -> Dict[str, List[Tuple[str, str]]]:
    """place -> [(subject_value, doc_id)]. Only EXCLUSIVE assertions are parsed;
    a document saying "A and B are both recorded as..." deliberately makes no
    exclusive claim and so cannot contradict anything."""
    out: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for doc_id, text in docs.items():
        m = rx.match(text.strip())
        if not m:
            continue
        val = m.group(subject).strip()
        place = m.group("place").strip().rstrip(".")
        out[place.lower()].append((val, doc_id))
    return out


def extract_capital_claims(docs: Dict[str, str]) -> Dict[str, List[Tuple[str, str]]]:
    return _extract(docs, CAPITAL_RE, "cap")


def extract_hq_claims(docs: Dict[str, str]) -> Dict[str, List[Tuple[str, str]]]:
    return _extract(docs, HQ_RE, "hq")


def validate(docs: Dict[str, str]) -> dict:
    problems: List[str] = []

    claims = extract_capital_claims(docs)
    hq_claims = extract_hq_claims(docs)
    for label, group in (("capitals", claims), ("headquarters", hq_claims)):
        for place, entries in sorted(group.items()):
            distinct = {c for c, _ in entries}
            if len(distinct) > 1:
                where = ", ".join(f"{c} ({d})" for c, d in entries)
                problems.append(f"contradiction: {place} has {label} -> {where}")

    # A state described as a city means the type constraint let an administrative
    # entity through the city query.
    for doc_id, text in docs.items():
        if re.search(r"\bis a major city in India, India\b", text):
            problems.append(f"malformed: {doc_id}: {text[:70]}")
        if re.search(r"\ba (\w+ )?state of India of India\b", text, re.I):
            problems.append(f"malformed: {doc_id}: {text[:70]}")
        if re.search(r"\((?:1[89]|20)\d\d", text):
            problems.append(f"historical entity: {doc_id}: {text[:70]}")

    # Same sentence twice under two ids inflates the store and can outvote truth.
    seen: Dict[str, str] = {}
    for doc_id, text in docs.items():
        key = text.strip().lower()
        if key in seen:
            problems.append(f"duplicate: {doc_id} repeats {seen[key]}")
        seen[key] = doc_id

    return {"docs": len(docs), "claims": len(claims),
            "hq_claims": len(hq_claims), "problems": problems}


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "india_kb.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    docs = {k: (v["text"] if isinstance(v, dict) else v)
            for k, v in raw.get("docs", {}).items()}
    rep = validate(docs)
    print(f"{path}: {rep['docs']} documents, {rep['claims']} capital claims, "
          f"{rep['hq_claims']} headquarters claims")
    if rep["problems"]:
        for p in rep["problems"]:
            print(f"  FAIL  {p}")
        print(f"\n{len(rep['problems'])} problem(s) - this KB would answer inconsistently")
        return 1
    print("  clean: no contradictions, malformed rows, or duplicates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
