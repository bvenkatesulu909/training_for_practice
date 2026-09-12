"""Transformer/SSD hybrid block: pre-norm, mixer, pre-norm, SwiGLU MLP."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from .budget import swiglu_hidden
from .ssd import SSDMixer


class SwiGLU(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0):
        super().__init__()
        hidden = swiglu_hidden(dim, ratio)       # kept tensor-core friendly
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg, kind: str):
        super().__init__()
        self.kind = kind
        self.n1 = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.mixer = SSDMixer(cfg) if kind == "ssd" else Attention(cfg, kind)
        self.n2 = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg.dim, cfg.mlp_ratio)

    def forward(self, x, cos, sin, cache=None, offset: int = 0):
        if self.kind == "ssd":
            h, new_cache = self.mixer(self.n1(x), cache)
        else:
            h, new_cache = self.mixer(self.n1(x), cos, sin, cache, offset)
        x = x + h
        x = x + self.mlp(self.n2(x))
        return x, new_cache
