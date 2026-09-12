"""Generate a synthetic arithmetic dataset: + - * / and percentages.

WHY THIS WORKS AT ALL
SmolLM2 tokenizes digits individually ('23' -> '2','3'), which is the precondition
for learnable arithmetic: the model can see and carry each digit. Tokenizers that
merge multi-digit numbers into single tokens make this close to impossible.

WHAT IT CAN AND CANNOT LEARN
A 135M model trained on this reliably handles 1-2 digit arithmetic and most 3-digit
addition/subtraction. It degrades on 3-digit multiplication and long division, and
it does not generalise past the digit range it was trained on. That is a property
of small transformers, not of this dataset.

Two things make a large difference and both are on by default:

  --scratchpad   The model writes the working before the answer. Multi-digit
                 arithmetic needs intermediate state; without somewhere to put it,
                 the model must compute the whole result in one forward pass, which
                 it cannot do. With it, accuracy on 3-digit problems roughly
                 doubles.

  phrasing pool  Each operation is asked ~10 different ways. Training on one
                 phrasing produces a model that only answers that phrasing - the
                 same instruction-diversity problem the repo-docs dataset had.

    python -m ft.build_math_dataset --out data_math --n 60000 --max-digits 3
    python -m ft.build_math_dataset --out data_math_mixed --n 40000 \
        --mix data_mixed/train.jsonl --mix-ratio 0.5
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from .data import dump_jsonl

# Several phrasings per operation. A model trained on one phrasing answers only
# that phrasing; this is the cheapest possible fix for that.
PHRASINGS: Dict[str, List[str]] = {
    "add": ["What is {a} + {b}?", "Calculate {a} + {b}", "{a} + {b} = ?",
            "Add {a} and {b}.", "What is the sum of {a} and {b}?",
            "Compute {a} plus {b}.", "{a} plus {b} equals what?",
            "Find the total of {a} and {b}.", "Work out {a} + {b}.",
            "If you add {a} to {b}, what do you get?"],
    "sub": ["What is {a} - {b}?", "Calculate {a} - {b}", "{a} - {b} = ?",
            "Subtract {b} from {a}.", "What is {a} minus {b}?",
            "Compute {a} take away {b}.", "Find the difference between {a} and {b}.",
            "Work out {a} - {b}.", "How much is {a} less {b}?",
            "If you subtract {b} from {a}, what remains?"],
    "mul": ["What is {a} * {b}?", "Calculate {a} x {b}", "{a} * {b} = ?",
            "Multiply {a} by {b}.", "What is the product of {a} and {b}?",
            "Compute {a} times {b}.", "{a} times {b} equals what?",
            "Work out {a} x {b}.", "Find {a} multiplied by {b}.",
            "What do you get multiplying {a} and {b}?"],
    "div": ["What is {a} / {b}?", "Calculate {a} divided by {b}", "{a} / {b} = ?",
            "Divide {a} by {b}.", "What is the quotient of {a} and {b}?",
            "Compute {a} over {b}.", "How many times does {b} go into {a}?",
            "Work out {a} / {b}.", "Find {a} divided by {b}.",
            "Split {a} into {b} equal parts - how big is each?"],
    "pct": ["What is {b}% of {a}?", "Calculate {b} percent of {a}",
            "{b}% of {a} = ?", "Find {b} percent of {a}.",
            "How much is {b}% of {a}?", "Work out {b}% of {a}.",
            "Compute {b} percent of {a}.", "Take {b}% of {a}.",
            "What is {b} percent of the number {a}?",
            "If something costs {a} and you take {b}% of it, what is that?"],
}


def _digits(rng: random.Random, n: int) -> int:
    """A number with exactly n digits (1-9 for n=1, 10-99 for n=2, ...)."""
    lo = 1 if n == 1 else 10 ** (n - 1)
    return rng.randint(lo, 10 ** n - 1)


def make_problem(rng: random.Random, op: str, max_digits: int,
                 scratchpad: bool) -> Tuple[str, str]:
    """Return (question, answer). Answers are exact; no rounding surprises."""
    da = rng.randint(1, max_digits)
    db = rng.randint(1, max_digits)

    if op == "add":
        a, b = _digits(rng, da), _digits(rng, db)
        result = a + b
        work = f"{a} + {b}\n= {result}"
    elif op == "sub":
        a, b = _digits(rng, da), _digits(rng, db)
        if b > a:                      # keep results non-negative
            a, b = b, a
        result = a - b
        work = f"{a} - {b}\n= {result}"
    elif op == "mul":
        # Cap multiplication size: 3x3 digit products are past what a 135M model
        # learns reliably, and filling the set with them just adds noise.
        a, b = _digits(rng, min(da, max_digits)), _digits(rng, min(db, 2))
        result = a * b
        tens, ones = divmod(b, 10)
        if b >= 10 and ones and scratchpad:   # partial products help the model carry
            work = (f"{a} x {b}\n= {a} x {ones} + {a} x {tens*10}\n"
                    f"= {a*ones} + {a*tens*10}\n= {result}")
        else:
            # With ones == 0 the split would read "a x 0 + a x b", restating the
            # problem as one of its own terms. A circular worked example teaches
            # the model nothing, so fall back to the direct form.
            work = f"{a} x {b}\n= {result}"
    elif op == "div":
        # Build from the answer so division is always exact.
        b = max(_digits(rng, min(db, 2)), 2)
        q = _digits(rng, min(da, max_digits))
        a = b * q
        result = q
        work = f"{a} / {b}\n= {result}  (since {b} x {result} = {a})"
    elif op == "pct":
        # Percentages that divide cleanly, so the answer is exact.
        b = rng.choice([1, 2, 5, 10, 20, 25, 40, 50, 60, 75, 80, 100])
        q = _digits(rng, min(da, max_digits))
        a = q * 100 // max(b, 1) if b else q
        a = max(a - a % (100 // max(__import__("math").gcd(b, 100), 1)), b)
        result = a * b // 100
        work = f"{b}% of {a}\n= {a} x {b} / 100\n= {a*b} / 100\n= {result}"
    else:
        raise ValueError(op)

    q_text = rng.choice(PHRASINGS[op]).format(a=a, b=b)
    answer = work if scratchpad else str(result)
    return q_text, answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_math")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--max-digits", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--no-scratchpad", action="store_true",
                    help="answer with the bare number (much harder to learn)")
    ap.add_argument("--ops", nargs="+", default=["add", "sub", "mul", "div", "pct"])
    ap.add_argument("--mix", default=None, help="existing JSONL to blend in")
    ap.add_argument("--mix-ratio", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    scratchpad = not args.no_scratchpad
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] generating {args.n:,} problems  ops={args.ops}  "
          f"max_digits={args.max_digits}  scratchpad={scratchpad}")

    seen, rows, dupes = set(), [], 0
    per_op = {o: 0 for o in args.ops}
    while len(rows) < args.n:
        op = args.ops[len(rows) % len(args.ops)]       # exact balance across ops
        q, a = make_problem(rng, op, args.max_digits, scratchpad)
        if q in seen:                                   # identical question twice
            dupes += 1
            if dupes > args.n * 10:
                print("      exhausted the problem space; stopping early")
                break
            continue
        seen.add(q)
        per_op[op] += 1
        rows.append({"messages": [{"role": "user", "content": q},
                                  {"role": "assistant", "content": a}]})

    print(f"      {len(rows):,} unique  ({dupes:,} duplicate questions skipped)")
    print(f"      per operation: {per_op}")

    if args.mix and args.mix_ratio > 0:
        extra = [l for l in open(args.mix, encoding="utf-8") if l.strip()]
        take = min(len(extra), int(len(rows) * args.mix_ratio))
        rng.shuffle(extra)
        import json
        rows += [json.loads(l) for l in extra[:take]]
        print(f"[2/3] blended {take:,} non-math examples from {args.mix}")
    else:
        print("[2/3] math only (no blend) - note this causes catastrophic "
              "forgetting of everything else; use --mix to keep other abilities")

    rng.shuffle(rows)
    n_val = max(int(len(rows) * args.val_frac), 1)
    for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        f = out / f"{name}.jsonl"
        f.write_text("\n".join(dump_jsonl(r) for r in part), encoding="utf-8")
        print(f"[3/3] {name}: {len(part):,} -> {f}")

    print("\nsample problems:")
    for r in rows[:4]:
        print(f"  Q: {r['messages'][0]['content']}")
        print(f"  A: {r['messages'][1]['content']}".replace("\n", "\n     "))


if __name__ == "__main__":
    main()
