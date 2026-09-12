"""Grouped-query attention with sliding-window and global variants.

Only a minority of layers use these (see `ModelConfig.layer_pattern`), because
attention is what makes long context expensive. The split matters a lot:

  window=4096 layers  -> KV cache is O(window), constant in sequence length
  global layers       -> KV cache is O(L); these are the ones you pay for at 300k
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


def sliding_window_mask(q_len: int, kv_len: int, window: int, device) -> torch.Tensor:
    """Boolean allow-mask (q_len, kv_len): causal AND within `window` of the query."""
    qi = torch.arange(q_len, device=device).unsqueeze(1) + (kv_len - q_len)
    kj = torch.arange(kv_len, device=device).unsqueeze(0)
    return (kj <= qi) & (kj > qi - window)


class Attention(nn.Module):
    def __init__(self, cfg, kind: str = "global"):
        super().__init__()
        self.kind = kind
        self.nh, self.nkv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.rep = self.nh // self.nkv
        self.window = cfg.window if kind == "swa" else None

        self.wq = nn.Linear(cfg.dim, self.nh * self.hd, bias=False)
        self.wk = nn.Linear(cfg.dim, self.nkv * self.hd, bias=False)
        self.wv = nn.Linear(cfg.dim, self.nkv * self.hd, bias=False)
        self.wo = nn.Linear(self.nh * self.hd, cfg.dim, bias=False)
        # QK-norm: keeps logits bounded at long context, a common divergence source.
        self.q_norm = nn.RMSNorm(self.hd, eps=cfg.norm_eps)
        self.k_norm = nn.RMSNorm(self.hd, eps=cfg.norm_eps)

    def forward(self, x, cos, sin, cache=None, offset: int = 0):
        b, l, _ = x.shape
        q = self.q_norm(self.wq(x).view(b, l, self.nh, self.hd)).transpose(1, 2)
        k = self.k_norm(self.wk(x).view(b, l, self.nkv, self.hd)).transpose(1, 2)
        v = self.wv(x).view(b, l, self.nkv, self.hd).transpose(1, 2)

        q = apply_rope(q, cos, sin, offset)
        k = apply_rope(k, cos, sin, offset)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        if self.window is not None:               # bounded cache for SWA layers
            k, v = k[:, :, -self.window:], v[:, :, -self.window:]
        # The prefill pass must publish its cache too, or decoding silently
        # attends to the current token alone.
        new_cache = (k, v)

        k = k.repeat_interleave(self.rep, dim=1)
        v = v.repeat_interleave(self.rep, dim=1)

        if self.window is not None and l > 1:
            m = sliding_window_mask(l, k.shape[2], self.window, x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=(l > 1))

        out = out.transpose(1, 2).reshape(b, l, self.nh * self.hd)
        return self.wo(out), new_cache
