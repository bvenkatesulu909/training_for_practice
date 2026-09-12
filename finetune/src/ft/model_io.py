"""Loading and saving, isolated because transformers renamed things in v5."""
from __future__ import annotations

import torch


def load_base(name: str, dtype: str = "float32", device: str = "cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token      # padding is masked out of the loss anyway

    td = getattr(torch, dtype)
    try:                                    # transformers >= 5
        model = AutoModelForCausalLM.from_pretrained(name, dtype=td)
    except TypeError:                       # transformers 4.x
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=td)
    return model.to(device), tok


def trainable_report(model) -> str:
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return (f"{tr:,} trainable / {tot:,} total ({100*tr/max(tot,1):.3f}%)  "
            f"| adapter ~{tr*4/2**20:.1f} MiB fp32")
