"""Rotary position embeddings.

1D for the language backbone, factorised 3D (time/height/width) for the vision
encoder so a video patch knows both where and *when* it is.
"""
from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 500000.0,
    scaling: float | None = None,
    device=None,
    dtype=torch.float32,
):
    """Return (cos, sin) of shape (seq_len, head_dim).

    `scaling` implements NTK-aware interpolation for context extension: train at
    4k with scaling=None, then serve at 300k with scaling=300000/4096 and only a
    short fine-tune, instead of retraining from scratch.
    """
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    if scaling is not None and scaling > 1.0:
        t = t / scaling
    freqs = torch.outer(t, inv)                    # (L, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)        # (L, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0):
    """x: (B, H, L, D). cos/sin: (>=offset+L, D)."""
    L = x.shape[-2]
    c = cos[offset:offset + L].unsqueeze(0).unsqueeze(0)
    s = sin[offset:offset + L].unsqueeze(0).unsqueeze(0)
    return (x * c) + (_rotate_half(x) * s)


def build_rope_3d(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    head_dim: int,
    theta: float = 10000.0,
    device=None,
    dtype=torch.float32,
):
    """Factorised 3D RoPE for vision patches.

    The head_dim/2 frequency pairs are split across the (t, h, w) axes, so the
    same encoder handles a still image (grid_t=1) and a 64-frame clip. Any
    remainder goes to the width axis, so head_dim need only be even.
    Returns (cos, sin) of shape (grid_t*grid_h*grid_w, head_dim).
    """
    assert head_dim % 2 == 0, "head_dim must be even"
    half = head_dim // 2
    nt = nh = half // 3
    nw = half - nt - nh

    def axis(n: int, d: int) -> torch.Tensor:
        if d == 0:
            return torch.zeros(n, 0, device=device)
        inv = 1.0 / (theta ** (torch.arange(0, d, device=device).float() / d))
        return torch.outer(torch.arange(n, device=device).float(), inv)  # (n, d)

    ft, fh, fw = axis(grid_t, nt), axis(grid_h, nh), axis(grid_w, nw)
    # Broadcast each axis over the full 3D grid, then concatenate.
    ft = ft[:, None, None, :].expand(grid_t, grid_h, grid_w, nt)
    fh = fh[None, :, None, :].expand(grid_t, grid_h, grid_w, nh)
    fw = fw[None, None, :, :].expand(grid_t, grid_h, grid_w, nw)
    freqs = torch.cat([ft, fh, fw], dim=-1).reshape(-1, half)  # (T*H*W, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)
