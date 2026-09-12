"""Top-level multimodal LM.

Sequence layout: text and code are ordinary BPE tokens; images and video occupy
runs of a reserved placeholder id whose embeddings are *overwritten* by encoder
output before the first block. One causal stream, four modalities.

    <|im_start|>user\n <|vision_start|> [PLACEHOLDER x N] <|vision_end|>
    what changes at 0:42? <|im_end|> <|im_start|>assistant\n ...
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block
from .budget import analytic_params, kv_cache_bytes
from .rope import build_rope_cache
from .vision import VisionEncoder

VISION_PLACEHOLDER = 3     # must match data/tokenizer.py SPECIALS


class MMLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg, k) for k in cfg.kinds()])
        self.norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.vision = VisionEncoder(cfg.vision, cfg.dim, cfg.norm_eps) if cfg.vision.enabled else None

        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim,
                                    cfg.rope_theta, cfg.rope_scaling)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init)
        # Scale residual-output projections by 1/sqrt(2*n_layers) so activation
        # variance stays ~constant with depth instead of compounding.
        s = (2 * cfg.n_layers) ** -0.5
        for n, p in self.named_parameters():
            if n.endswith(("out_proj.weight", "wo.weight", "mlp.w2.weight")):
                p.data.mul_(s)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv3d)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ---------------------------------------------------------------- forward
    def forward(self, tokens, pixels=None, targets=None, caches=None, offset: int = 0):
        h = self.embed(tokens)

        if pixels is not None and self.vision is not None:
            feats = self.vision(pixels).reshape(-1, self.cfg.dim)
            slots = tokens == VISION_PLACEHOLDER
            n = int(slots.sum())
            assert n == feats.shape[0], (
                f"{n} vision placeholder tokens but encoder produced {feats.shape[0]} features"
            )
            h = h.masked_scatter(slots.unsqueeze(-1), feats.to(h.dtype))

        new_caches = []
        for i, blk in enumerate(self.blocks):
            h, c = blk(h, self.cos, self.sin, caches[i] if caches else None, offset)
            new_caches.append(c)
        h = self.norm(h)

        if targets is None:
            return self.lm_head(h[:, -1:]), new_caches

        logits = self.lm_head(h)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1),
            ignore_index=-100,
        )
        if self.cfg.z_loss:      # penalise logit drift; prevents late-run blowups
            lse = torch.logsumexp(logits.float(), dim=-1)
            loss = loss + self.cfg.z_loss * (lse ** 2).mean()
        return logits, loss

    # --------------------------------------------------------------- sampling
    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=64, temperature=0.8, top_k=50, pixels=None):
        self.eval()
        logits, caches = self.forward(tokens, pixels=pixels)
        offset = tokens.shape[1]
        out = tokens
        for _ in range(max_new_tokens):
            l = logits[:, -1].float() / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(l, min(top_k, l.size(-1)))
                l = l.masked_fill(l < v[:, [-1]], float("-inf"))
            nxt = torch.multinomial(F.softmax(l, dim=-1), 1)
            out = torch.cat([out, nxt], dim=1)
            logits, caches = self.forward(nxt, caches=caches, offset=offset)
            offset += 1
        return out

    # ---------------------------------------------------------------- budgets
    def n_params(self, embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not embedding:
            n -= self.embed.weight.numel()
            if not self.cfg.tie_embeddings:   # untied head is an embedding too
                n -= self.lm_head.weight.numel()
        return n

    def kv_cache_bytes(self, seq_len: int, bytes_per_elem: int = 2) -> dict:
        """Inference KV-cache footprint — see model/budget.py."""
        return kv_cache_bytes(self.cfg, seq_len, bytes_per_elem)
