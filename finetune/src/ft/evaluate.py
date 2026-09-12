"""Side-by-side base vs fine-tuned comparison.

A loss number alone does not tell you whether the fine-tune helped in the way you
wanted, so this prints both models' answers to the same held-out prompts. Held-out
means held-out: these prompts were never trained on.

    PYTHONPATH=src python -m ft.evaluate --adapter runs/x/adapter_best.pt --n 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import SFTDataset, apply_template, collate, iter_jsonl, normalise
from .lora import merge_lora
from .merge import load_adapted
from .model_io import load_base


@torch.no_grad()
def gen(model, tok, messages, max_new=120, device="cpu"):
    ids = torch.tensor([apply_template(tok, messages, add_generation_prompt=True)],
                       device=device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def val_loss(model, ds, pad_id, device, micro=1, max_batches=40):
    model.eval()
    tot = n = 0
    for i in range(0, min(len(ds), max_batches * micro), micro):
        b = collate([ds[j] for j in range(i, min(i + micro, len(ds)))], pad_id)
        tot += model(**{k: v.to(device) for k, v in b.items()}).loss.item()
        n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--val-file", default="data/val.jsonl")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tuned, tok, cfg, sd = load_adapted(args.adapter, device=args.device)
    merge_lora(tuned); tuned.eval()
    base, _ = load_base(cfg.base_model, "float32", args.device); base.eval()

    ds = SFTDataset(args.val_file, tok, cfg.data.max_len,
                    cfg.data.on_overflow, cfg.data.mask_prompt)
    # Same guard as training: comparing two models on an almost-empty eval set
    # produces confident numbers about nothing.
    ds.warn_if_truncating("val")
    lb = val_loss(base, ds, tok.pad_token_id, args.device)
    lt = val_loss(tuned, ds, tok.pad_token_id, args.device)
    print(f"held-out loss   base {lb:.4f}  ->  tuned {lt:.4f}   ({lt-lb:+.4f})")
    print(f"perplexity      base {pow(2.718281828, lb):.1f}  ->  tuned {pow(2.718281828, lt):.1f}")

    rows = list(iter_jsonl(args.val_file))
    for r in rows[:args.n]:
        msgs = normalise(r)
        prompt = [m for m in msgs if m["role"] != "assistant"]
        print("\n" + "=" * 72)
        print("PROMPT   :", prompt[-1]["content"][:220].replace("\n", " "))
        print("-" * 72)
        print("BASE     :", gen(base, tok, prompt, args.max_new, args.device)[:400])
        print("-" * 72)
        print("TUNED    :", gen(tuned, tok, prompt, args.max_new, args.device)[:400])
        print("-" * 72)
        print("REFERENCE:", msgs[-1]["content"][:400].replace("\n", " "))


if __name__ == "__main__":
    main()
