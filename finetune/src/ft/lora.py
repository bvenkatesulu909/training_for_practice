"""LoRA: low-rank adapters injected into an existing pretrained model.

Implemented directly rather than via `peft` because transformers 5.x is new enough
that adapter-library compatibility is a coin flip, and this is ~60 lines that can
be tested exactly.

    W_effective = W_frozen + (alpha/r) * B @ A        A: (r, in)  B: (out, r)

B is initialised to zero, so at step 0 the adapted model is *bit-identical* to the
base model. That property is worth having: if your first eval differs from the base
model's, something is wrong before you have wasted any compute.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.scaling = alpha / r
        # Match the base layer's dtype/device. Creating adapters in the default
        # dtype breaks the moment the base is loaded in bf16 or fp16 — which is
        # every GPU run.
        kw = {"dtype": base.weight.dtype, "device": base.weight.device}
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, **kw))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, **kw))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Kaiming on A, zeros on B -> the product BA is zero at init.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + delta * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + (self.lora_B @ self.lora_A) * self.scaling


def _walk(module: nn.Module, targets: Iterable[str], prefix: str = ""):
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and name in targets:
            yield module, name, path, child
        else:
            yield from _walk(child, targets, path)


def inject_lora(model: nn.Module, cfg) -> Dict[str, int]:
    """Freeze the base model and replace target Linear layers with LoRALinear."""
    for p in model.parameters():
        p.requires_grad = False

    replaced: List[str] = []
    for parent, name, path, child in list(_walk(model, set(cfg.target_modules))):
        setattr(parent, name, LoRALinear(child, cfg.r, cfg.alpha, cfg.dropout))
        replaced.append(path)

    if not replaced:
        raise ValueError(
            f"no modules matched {cfg.target_modules}. Inspect the architecture with "
            f"`[n for n,_ in model.named_modules()]` — projection names differ by family."
        )

    if cfg.train_embeddings:
        for mod in (model.get_input_embeddings(), model.get_output_embeddings()):
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"replaced": len(replaced), "trainable": trainable, "total": total,
            "pct": 100.0 * trainable / max(total, 1), "paths": replaced}


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Only the adapter — a few MB, versus GBs for the full model."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k}


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """Fold adapters into the base weights and restore plain nn.Linear layers.

    After this the model is a normal HF model with no custom classes, so it can be
    saved with save_pretrained and served by anything (vLLM, llama.cpp, Ollama).
    """
    merged = 0
    while True:
        found = None
        for parent, name, _, _ in _walk_lora(model):
            found = (parent, name)
            break
        if found is None:
            break
        parent, name = found
        mod: LoRALinear = getattr(parent, name)
        base = mod.base
        base.weight.data.copy_(mod.merged_weight())
        base.weight.requires_grad_(False)
        setattr(parent, name, base)
        merged += 1
    return merged


def _walk_lora(module: nn.Module, prefix: str = ""):
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, LoRALinear):
            yield module, name, path, child
        else:
            yield from _walk_lora(child, path)
