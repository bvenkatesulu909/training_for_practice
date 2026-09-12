"""There is only ONE thing a language model learns: P(next token | context).

Everything else - grammar, facts, style, arithmetic, "knowing" the capital of
France - is a side effect of getting good at that single objective. The weights do
not store facts as facts. They store a function that turns a context into a
probability distribution over the vocabulary.

So "does it learn weights, or does it learn to predict tokens exactly?" is not a
choice between two things. The weights ARE the token-prediction function. Training
adjusts the weights so the distribution puts its mass on the token that actually
came next in the data.

This script makes that visible. For each prompt it prints the model's top-5 next
tokens with probabilities, plus the ENTROPY of the distribution:

    low entropy  -> mass concentrated on few tokens -> the model "knows"
    high entropy -> mass spread thin               -> the model is guessing

A fact the model learned and a fact it never saw look identical in architecture.
They differ only in how peaked the distribution is.

    PYTHONPATH=src python scripts/what_weights_know.py
"""
from __future__ import annotations

import argparse
import math

import torch

from ft.data import apply_template
from ft.model_io import load_base

PROBES = [
    ("The capital of France is",            "seen constantly in pretraining"),
    ("The capital of India is",             "seen, but less consistently phrased"),
    ("2 + 2 =",                             "arithmetic - never explicitly trained"),
    ("The capital of Zzyrgquist is",        "nonsense - nothing to know"),
    ("Once upon a time, there was a little", "story template - extremely common"),
]


def probe(model, tok, text: str, k: int = 5):
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]])
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
    probs = torch.softmax(logits, -1)
    # Entropy over the FULL vocabulary, in bits. This is the honest measure of
    # how much the model has actually narrowed things down.
    ent = -(probs * torch.log2(probs.clamp_min(1e-12))).sum().item()
    top_p, top_i = probs.topk(k)
    return ent, [(tok.decode([i]), p.item()) for p, i in zip(top_p, top_i)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    args = ap.parse_args()

    model, tok = load_base(args.model)
    model.eval()
    vocab = model.config.vocab_size
    max_ent = math.log2(vocab)
    print(f"model      : {args.model}")
    print(f"vocabulary : {vocab:,} tokens")
    print(f"max entropy: {max_ent:.2f} bits  (a model that knows NOTHING scores this)")
    print()

    for text, note in PROBES:
        ent, top = probe(model, tok, text)
        certainty = 100 * (1 - ent / max_ent)
        bar = "#" * int(certainty / 4)
        print(f'"{text}..."')
        print(f"   {note}")
        print(f"   entropy {ent:6.2f} bits of {max_ent:.2f}   certainty {certainty:5.1f}% {bar}")
        print("   top-5: " + "  ".join(f"{t!r}={p:.3f}" for t, p in top))
        print()

    print("=" * 72)
    print("Same architecture, same weights, same forward pass for every line above.")
    print("The ONLY difference is how sharply the probability mass is concentrated.")
    print()
    print("That is what training does: it moves weights so that mass lands on the")
    print("token that really followed, for the data it saw. Nothing is stored as a")
    print("fact. A confident answer and a confident hallucination are produced by")
    print("exactly the same machinery - which is why the model cannot tell you")
    print("which one it is doing.")


if __name__ == "__main__":
    main()
