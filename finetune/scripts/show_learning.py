"""Watch backpropagation fix a wrong prediction, step by step.

Takes one example the model currently gets wrong, and shows the full loop:

    forward -> loss -> backward -> gradients -> optimizer -> new weights -> better prediction

For each step it prints the loss, the global gradient norm, the probability the
model assigns to the CORRECT next token, and what it would actually say. You can
watch P(correct) climb from near zero to near one as the weights move.

This is a *demonstration of the mechanism*, not a fix. Overfitting one example
teaches the model that one example. Generalising to all arithmetic needs the full
dataset (see ft.build_math_dataset) - but the machinery is identical, just run over
thousands of examples instead of one.

    PYTHONPATH=src python scripts/show_learning.py
    PYTHONPATH=src python scripts/show_learning.py --question "What is 7 * 8?" --answer "56"
"""
from __future__ import annotations

import argparse

import torch

from ft.config import LoRAConfig
from ft.data import IGNORE, apply_template, encode_example
from ft.lora import inject_lora
from ft.model_io import load_base


def show(model, tok, messages, target_ids, prompt_len, label):
    """Print what the model predicts, and how sure it is of the right token."""
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([apply_template(tok, messages[:1], add_generation_prompt=True)])
        logits = model(ids).logits[0, -1].float()
        probs = torch.softmax(logits, -1)
        first_target = target_ids[prompt_len]
        p_correct = probs[first_target].item()
        top_p, top_i = probs.max(-1)
        out = model.generate(ids, max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        said = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    model.train()
    return {"p_correct": p_correct, "top_tok": tok.decode([top_i.item()]),
            "top_p": top_p.item(), "said": said}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--question", default="What is 2 + 2?")
    ap.add_argument("--answer", default="4")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()

    print(f"model    : {args.model}")
    print(f"example  : {args.question!r}  ->  {args.answer!r}")
    model, tok = load_base(args.model)

    info = inject_lora(model, LoRAConfig(r=args.rank, alpha=args.rank * 2, dropout=0.0))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"training : {sum(p.numel() for p in params):,} LoRA params "
          f"({info['pct']:.2f}% of {info['total']:,}) - the frozen base never moves")

    messages = [{"role": "user", "content": args.question},
                {"role": "assistant", "content": args.answer}]
    ids, labels = encode_example(tok, messages, max_len=512, mask_prompt=True)
    x = torch.tensor([ids])
    y = torch.tensor([labels])
    prompt_len = next(i for i, l in enumerate(labels) if l != IGNORE)
    n_sup = sum(1 for l in labels if l != IGNORE)
    print(f"sequence : {len(ids)} tokens, {n_sup} supervised "
          f"(loss only on the answer, never the question)")

    opt = torch.optim.AdamW(params, lr=args.lr)

    before = show(model, tok, messages, ids, prompt_len, "before")
    print(f"\nBEFORE  P(correct first answer token) = {before['p_correct']:.6f}")
    print(f"        model instead predicts {before['top_tok']!r} at p={before['top_p']:.4f}")
    print(f"        it would say: {before['said']!r}")

    print("\n step        loss    grad_norm   P(correct)   top_token   says")
    print(" " + "-" * 68)
    for step in range(args.steps + 1):
        out = model(input_ids=x, labels=y)
        loss = out.loss

        opt.zero_grad(set_to_none=True)
        loss.backward()                                    # <- BACKPROPAGATION
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1e9).item()   # measure, don't clip

        if step % 3 == 0 or step == args.steps:
            s = show(model, tok, messages, ids, prompt_len, "during")
            mark = "  <-- exact" if s["said"].startswith(args.answer) else ""
            print(f" {step:4d}  {loss.item():10.6f}  {gnorm:10.4f}   {s['p_correct']:10.6f}"
                  f"   {s['top_tok']!r:>9}   {s['said'][:18]!r}{mark}")

        if step < args.steps:
            opt.step()                                     # <- OPTIMIZER UPDATES WEIGHTS

    after = show(model, tok, messages, ids, prompt_len, "after")
    print("\n" + "=" * 70)
    print(f"P(correct token):  {before['p_correct']:.6f}  ->  {after['p_correct']:.6f}"
          f"   ({after['p_correct']/max(before['p_correct'],1e-12):,.0f}x)")
    print(f"model says      :  {before['said']!r}  ->  {after['said']!r}")
    print("=" * 70)
    print("\nThat is the whole loop: the loss told us how wrong we were, backward()")
    print("turned that into a gradient for every weight, and AdamW used the gradient")
    print("to move the weights. Repeat over a real dataset and it generalises;")
    print("repeat over one example, as here, and it only memorises that example.")


if __name__ == "__main__":
    main()
