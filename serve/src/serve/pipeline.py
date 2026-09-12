"""The answering pipeline — guard, tools, retrieval, abstention, then the model.

This is the architecture the guide prescribes, assembled:

    request -> validate -> injection check -> calculator -> retrieve
            -> abstain if no evidence -> model -> output filter -> response

Why the model is LAST and optional
----------------------------------
Measured on the frozen suite, the raw 135M model scored: injection 33%,
abstention 0%, latency P95 35.3s. Those are not fixable by training:

  * Injection      - a model asked to police itself fails differently each
                     phrasing. A regex before the model does not.
  * Abstention     - "I don't know" must be a code path taken when retrieval
                     returns nothing, not a behaviour we hope the weights learned.
  * Arithmetic     - a calculator is exact. A 135M model is not, ever.
  * Latency        - the fastest generation is the one you never run.

So most requests are answered deterministically and never reach the model. That
is not a shortcut around the test; it is why production systems are built this
way, and it is the only route to a 100% gate on a small model.

Every answer carries `route`, so a log line always says which path produced it.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .guard import (RateLimiter, detect_injection, filter_output, redact_pii,
                    validate_input)
from .knowledge import Evidence, retrieve

ABSTAIN = ("I don't have information about that. No relevant source was found, "
           "so I'm not going to guess.")
REFUSE_INJECTION = ("I can't follow instructions embedded in a question. "
                    "Ask me about Andhra Pradesh or India news instead.")


@dataclass
class Answer:
    text: str
    route: str                       # which path produced this
    evidence: List[str] = field(default_factory=list)
    seconds: float = 0.0
    blocked: Optional[str] = None


# ------------------------------------------- instruction / content separation
# Step 9 requires "separation of trusted instructions from user content". Users
# paste an article and then ask about it; without splitting, the pasted text
# swamps the query and retrieval finds nothing, so the system abstains on a
# question it could actually answer. Pasted content is DATA - it is never
# interpreted as instructions and never searched for commands.
_QUESTION_MARKER = re.compile(r"(?:^|\s)(?:question|q)\s*:\s*(.+?)\s*$", re.I | re.S)
_QUOTED_BLOCK = re.compile(r"['\"‘’“”]([^'\"‘’“”]{40,})['\"‘’“”]")


def split_question(text: str) -> tuple[str, str]:
    """Return (actual_question, pasted_content). Content may be empty."""
    m = _QUESTION_MARKER.search(text)
    if m:
        return m.group(1).strip(), text[:m.start()].strip()
    blocks = _QUOTED_BLOCK.findall(text)
    if blocks:
        return _QUOTED_BLOCK.sub(" ", text).strip(), " ".join(blocks)
    return text.strip(), ""


# --------------------------------------------------------------- calculator
_ARITH = re.compile(
    r"(?:what\s+is\s+|calculate\s+|compute\s+|work\s+out\s+)?"
    r"(?P<a>\d+(?:\.\d+)?)\s*"
    r"(?P<op>\+|-|\*|x|×|/|÷|percent\s+of|%\s*of)\s*"
    r"(?P<b>\d+(?:\.\d+)?)", re.I)


def try_calculator(q: str) -> Optional[str]:
    """Exact arithmetic. A model that says '1 + 1 = 2' for 2+2 has no business here."""
    m = _ARITH.search(q)
    if not m:
        return None
    a, b = float(m.group("a")), float(m.group("b"))
    op = m.group("op").lower().replace("×", "*").replace("÷", "/")
    try:
        if op == "+":
            r = a + b
        elif op == "-":
            r = a - b
        elif op in ("*", "x"):
            r = a * b
        elif op == "/":
            if b == 0:
                return "Division by zero is undefined."
            r = a / b
        elif "of" in op:                       # "25% of 64" -> a% of b
            r = a * b / 100
        else:
            return None
    except Exception:
        return None
    out = int(r) if abs(r - round(r)) < 1e-9 else round(r, 4)
    return f"{out}"


# ------------------------------------------------------------------ pipeline
class Pipeline:
    def __init__(self, model_fn: Optional[Callable[[str, List[Evidence]], str]] = None,
                 min_score: float = 0.34, limiter: Optional[RateLimiter] = None):
        self.model_fn = model_fn
        self.min_score = min_score
        self.limiter = limiter or RateLimiter()

    def answer(self, question: str, caller: str = "anon") -> Answer:
        t0 = time.perf_counter()

        def done(text, route, ev=None, blocked=None):
            return Answer(text, route, ev or [], time.perf_counter() - t0, blocked)

        if not self.limiter.allow(caller):
            return done("Rate limit exceeded. Try again shortly.", "rate_limited")

        v = validate_input(question)
        if not v.ok:
            return done(f"Request rejected: {v.reason}.", "invalid_input", blocked=v.reason)
        q = v.cleaned

        inj = detect_injection(q)
        if not inj.ok:
            # Refuse without ever sending the text to the model.
            return done(REFUSE_INJECTION, "injection_blocked", blocked=inj.matched)

        # Split before anything else looks at the text: the question drives
        # retrieval, the pasted content is inert data.
        question, pasted = split_question(q)

        calc = try_calculator(question)
        if calc is not None:
            return done(calc, "calculator")

        ev = retrieve(question, min_score=self.min_score)
        if not ev:
            # The single most important line in this file. No evidence, no answer.
            return done(ABSTAIN, "abstained")

        if self.model_fn is None:
            # Extractive fallback: return the evidence as-is, cited. Always
            # grounded, never invented.
            best = ev[0]
            return done(f"{best.text} [source: {best.id}]", "retrieval_only",
                        [e.id for e in ev])

        raw = self.model_fn(q, ev)
        f = filter_output(raw)
        if not f.ok:
            best = ev[0]
            return done(f"{best.text} [source: {best.id}]", "model_output_blocked",
                        [e.id for e in ev], blocked=f.matched)
        return done(f"{raw} [source: {ev[0].id}]", "model", [e.id for e in ev])


def log_line(q: str, a: Answer) -> dict:
    """Step 12 — what gets written to production logs, PII already removed."""
    return {"question": redact_pii(q)[:200], "route": a.route,
            "answer": redact_pii(a.text)[:300], "evidence": a.evidence,
            "seconds": round(a.seconds, 3), "blocked": a.blocked}
