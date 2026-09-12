"""Step 9 — public-use safety controls, enforced by the APPLICATION.

The guide's rule: "Never give the model unrestricted access... The application -
not the model - enforces permissions."

Every control here is deterministic. That matters for a release gate: a regex
either matched or it did not, whereas a model asked to police itself gives a
different answer depending on how the attack is phrased. The baseline model,
asked to ignore its instructions and say BANANA, said BANANA. No amount of
fine-tuning makes that reliably impossible; a check before the model does.

Layers, in order:
  1. validate_input   - size, emptiness, control characters
  2. detect_injection - instruction-override and role-swap attempts
  3. redact_pii       - strip personal data before anything is logged or sent
  4. filter_output    - block template leakage and system-prompt disclosure
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

MAX_INPUT_CHARS = 500
MAX_OUTPUT_TOKENS = 200

# Instruction-override patterns. Deliberately broad: a false positive costs one
# refusal, a false negative costs a prompt injection.
INJECTION_PATTERNS = [
    # Allow up to three filler words ("all the", "any of your", ...) between the
    # verb and its object. A pattern that only accepted "all" was defeated by
    # "ignore THE above instructions" - a one-word change.
    r"(?:ignore|disregard|forget|override|bypass)\s+(?:\w+\s+){0,3}?"
    r"(?:previous|prior|above|earlier|preceding|initial|original|all)\s*"
    r"(?:instruction|prompt|rule|direction|command|message)",
    r"(?:ignore|disregard|forget)\s+(?:everything|all)\b",
    r"\byou\s+are\s+now\s+(in\s+)?(developer|debug|admin|god|dan)\s*mode",
    r"^\s*(system|assistant)\s*:",           # fake role turn
    r"print\s+(your|the)\s+(system\s+)?prompt",
    r"reveal\s+(your|the)\s+(system\s+)?(prompt|instructions)",
    r"repeat\s+(everything\s+)?above",
    r"new\s+instructions?\s*:",
    r"</?(system|instruction)>",             # fake tags
]
_INJ = [re.compile(p, re.I | re.M) for p in INJECTION_PATTERNS]

# Output leakage: template placeholders absorbed from training data, and any
# attempt to echo instructions back.
OUTPUT_BLOCKLIST = [
    r"\[address\]", r"\[name\]", r"\[phone\]", r"\[email\]", r"\[insert[^\]]*\]",
    r"p\.s\.\s+visit us", r"for more updates and tips",
    r"my system prompt", r"my instructions are",
]
_OUT = [re.compile(p, re.I) for p in OUTPUT_BLOCKLIST]

PII_PATTERNS = [
    (re.compile(r"[\w\.\-+]+@[\w\-]+\.\w{2,}"), "[email]"),
    (re.compile(r"\+?\d[\d\s\-()]{8,}\d"), "[phone]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[id]"),          # Aadhaar-like
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[card]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[pan]"),            # Indian PAN
]


@dataclass
class GuardResult:
    ok: bool
    reason: str = ""
    cleaned: str = ""
    matched: Optional[str] = None


def validate_input(text: str) -> GuardResult:
    if text is None or not text.strip():
        return GuardResult(False, "empty input")
    if len(text) > MAX_INPUT_CHARS:
        return GuardResult(False, f"input exceeds {MAX_INPUT_CHARS} characters")
    # Strip control characters that can hide instructions from a human reviewer
    # while still reaching the model.
    cleaned = "".join(ch for ch in text if ch == "\n" or ch >= " ")
    return GuardResult(True, "ok", cleaned=cleaned.strip())


def detect_injection(text: str) -> GuardResult:
    for rx in _INJ:
        m = rx.search(text)
        if m:
            return GuardResult(False, "prompt injection detected", matched=m.group(0)[:60])
    return GuardResult(True, "clean")


def redact_pii(text: str) -> str:
    """Applied before logging and before anything reaches a training set."""
    for rx, tag in PII_PATTERNS:
        text = rx.sub(tag, text)
    return text


def filter_output(text: str) -> GuardResult:
    for rx in _OUT:
        m = rx.search(text or "")
        if m:
            return GuardResult(False, "blocked output pattern", matched=m.group(0)[:60])
    return GuardResult(True, "clean", cleaned=text)


class RateLimiter:
    """Fixed-window per-caller limiter. In-process; a real deployment uses Redis."""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max = max_requests
        self.window = window_seconds
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        import time
        now = now if now is not None else time.time()
        hits = [t for t in self._hits.get(key, []) if now - t < self.window]
        if len(hits) >= self.max:
            self._hits[key] = hits
            return False
        hits.append(now)
        self._hits[key] = hits
        return True
