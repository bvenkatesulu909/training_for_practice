"""A small, factually-checked knowledge set for the news app's domain.

The point of this file is to be honest about what fine-tuning on facts achieves.
Training on these pairs WILL make the model answer these questions correctly - that
is memorisation, and it works. It will NOT make the model knowledgeable about India,
because nothing here generalises to a fact the model never saw.

The held-out block at the bottom is never trained on. After fine-tuning, compare
accuracy on TRAIN facts (should be high) against HELD_OUT facts (should stay poor).
That difference is the whole lesson.

Every fact below is uncontested. Genuinely disputed items - such as which city is
Andhra Pradesh's working capital, which has been through multiple legal reversals -
are deliberately excluded rather than asserted.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from ft.data import dump_jsonl

# (question, answer) - each fact asked several ways, because one phrasing teaches
# only that phrasing.
FACTS = [
    # India, national
    ("What is the capital of India?", "New Delhi is the capital of India."),
    ("Where is the capital of India?", "The capital of India is New Delhi."),
    ("Which city is India's capital?", "New Delhi."),
    ("Name the capital city of India.", "New Delhi."),
    ("What is India's seat of central government?",
     "New Delhi, the capital of India, is the seat of the central government."),

    # Hyderabad - the question that failed in the screenshot
    ("Where is Hyderabad in India?",
     "Hyderabad is in southern India. It is the capital of the state of Telangana."),
    ("Which state is Hyderabad in?", "Hyderabad is the capital of Telangana."),
    ("What is Hyderabad the capital of?", "Hyderabad is the capital of Telangana."),
    ("Tell me about the location of Hyderabad.",
     "Hyderabad lies on the Deccan Plateau in southern India, and is the capital of Telangana."),
    ("Is Hyderabad in north or south India?", "Hyderabad is in southern India."),

    # Telangana
    ("What is Telangana?",
     "Telangana is a state in southern India. Its capital is Hyderabad."),
    ("When was Telangana formed?",
     "Telangana was formed on 2 June 2014, when it was separated from Andhra Pradesh."),
    ("What is the capital of Telangana?", "Hyderabad."),
    ("Is Telangana a state or a city?", "Telangana is a state in southern India."),

    # Andhra Pradesh - only uncontested facts
    ("What is Andhra Pradesh?",
     "Andhra Pradesh is a state on the southeastern coast of India."),
    ("When was Andhra Pradesh divided?",
     "Andhra Pradesh was divided on 2 June 2014, when Telangana was created from it."),
    ("Which sea or bay borders Andhra Pradesh?",
     "Andhra Pradesh borders the Bay of Bengal to its east."),
    ("Name a major city in Andhra Pradesh.",
     "Visakhapatnam is the largest city in Andhra Pradesh."),

    # A few general facts, to check the model is not simply collapsing to one answer
    ("What is the capital of France?", "Paris."),
    ("What is the capital of Japan?", "Tokyo."),
    ("What is 2 + 2?", "4."),
    ("What is 10 x 10?", "100."),
]

# NEVER trained on. These measure whether anything generalised.
HELD_OUT = [
    ("What is the capital of Karnataka?", "Bengaluru"),
    ("Which state is Chennai the capital of?", "Tamil Nadu"),
    ("What is the capital of Kerala?", "Thiruvananthapuram"),
    ("What is 7 + 5?", "12"),
    ("What is the capital of Germany?", "Berlin"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_facts")
    ap.add_argument("--repeats", type=int, default=40,
                    help="times each fact appears (memorisation needs repetition)")
    ap.add_argument("--mix", default=None, help="JSONL to blend in, to limit forgetting")
    ap.add_argument("--mix-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for _ in range(args.repeats):
        for q, a in FACTS:
            rows.append({"messages": [{"role": "user", "content": q},
                                      {"role": "assistant", "content": a}]})

    print(f"[1/2] {len(FACTS)} distinct facts x {args.repeats} repeats = {len(rows):,} examples")

    if args.mix:
        import json
        extra = [l for l in open(args.mix, encoding="utf-8") if l.strip()]
        take = min(len(extra), int(len(rows) * args.mix_ratio))
        rng.shuffle(extra)
        rows += [json.loads(l) for l in extra[:take]]
        print(f"      blended {take:,} other examples to limit catastrophic forgetting")

    rng.shuffle(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Validation is the facts themselves: we are measuring memorisation on purpose.
    val = [{"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}]} for q, a in FACTS]
    for name, part in (("val", val), ("train", rows)):
        f = out / f"{name}.jsonl"
        f.write_text("\n".join(dump_jsonl(r) for r in part), encoding="utf-8")
        print(f"[2/2] {name}: {len(part):,} -> {f}")


if __name__ == "__main__":
    main()
