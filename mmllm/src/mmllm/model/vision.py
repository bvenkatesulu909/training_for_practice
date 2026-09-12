"""3D-patch vision encoder shared by images and video.

An image is just a 1-frame clip, so there is exactly one encoder and one set of
weights for both modalities. Token accounting is the whole design pressure here:
video is what eats a 300k context, so patches are merged before the LM sees them.

  image 224x224, patch 14, merge 2  ->  (16/2)^2  =  256 tokens
  video 1 fps,   temporal_patch 2   ->  128 tokens per frame
  => 300,000 ctx  ~=  2,300 frames  ~=  39 minutes of video
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import build_rope_3d, apply_rope
from .block import SwiGLU


class ViTBlock(nn.Module):
    """Bidirectional (non-causal) attention block — the encoder sees whole frames."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, eps: float = 1e-5):
        super().__init__()
        self.h, self.hd = heads, dim // heads
        self.n1 = nn.RMSNorm(dim, eps=eps)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.n2 = nn.RMSNorm(dim, eps=eps)
        self.mlp = SwiGLU(dim, mlp_ratio)

    def forward(self, x, cos, sin):
        b, l, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, l, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v)      # no causal mask
        x = x + self.proj(o.transpose(1, 2).reshape(b, l, d))
        return x + self.mlp(self.n2(x))


class VisionEncoder(nn.Module):
    def __init__(self, vcfg, out_dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.cfg = vcfg
        p, tp = vcfg.patch_size, vcfg.temporal_patch
        # Conv3d does patchification for both axes in one op.
        self.patch = nn.Conv3d(3, vcfg.dim, (tp, p, p), stride=(tp, p, p))
        self.blocks = nn.ModuleList([
            ViTBlock(vcfg.dim, vcfg.heads, vcfg.mlp_ratio, norm_eps) for _ in range(vcfg.depth)
        ])
        self.norm = nn.RMSNorm(vcfg.dim, eps=norm_eps)

        # Pixel-shuffle merge: m^2 neighbouring patches concat -> one LM token.
        m = vcfg.merge_factor
        self.merge = m
        self.proj = nn.Sequential(
            nn.Linear(vcfg.dim * m * m, out_dim, bias=True),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: (B, 3, T, H, W), T divisible by temporal_patch.
        Returns (B, n_tokens, out_dim) ready to splice into the LM sequence."""
        b = pixels.shape[0]
        z = self.patch(pixels)                            # (B, D, gt, gh, gw)
        _, d, gt, gh, gw = z.shape
        z = z.flatten(2).transpose(1, 2)                  # (B, gt*gh*gw, D)

        cos, sin = build_rope_3d(gt, gh, gw, d // self.blocks[0].h,
                                 self.cfg.rope_theta, pixels.device, z.dtype)
        for blk in self.blocks:
            z = blk(z, cos, sin)
        z = self.norm(z)

        # Spatial merge within each temporal group.
        m = self.merge
        z = z.view(b, gt, gh, gw, d)
        z = z.view(b, gt, gh // m, m, gw // m, m, d).permute(0, 1, 2, 4, 3, 5, 6)
        z = z.reshape(b, gt * (gh // m) * (gw // m), m * m * d)
        return self.proj(z)

    def n_tokens(self, frames: int, h: int, w: int) -> int:
        c = self.cfg
        gt = frames // c.temporal_patch
        return gt * (h // c.patch_size // self.merge) * (w // c.patch_size // self.merge)
