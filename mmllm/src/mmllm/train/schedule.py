"""Learning-rate schedules and optimizer construction."""
from __future__ import annotations

import math

import torch


def lr_at(step: int, cfg) -> float:
    """WSD (warmup-stable-decay) or cosine.

    WSD is preferred for a first real run: the long constant-LR trunk means you
    can extend training or branch several finals from one run without having
    committed to a fixed step count up front, which cosine forces you to do.
    """
    total, warm = cfg.max_steps, max(int(cfg.warmup_frac * cfg.max_steps), 1)
    lo = cfg.lr * cfg.min_lr_frac

    if step < warm:
        return cfg.lr * (step + 1) / warm
    if cfg.schedule == "cosine":
        p = (step - warm) / max(total - warm, 1)
        return lo + 0.5 * (cfg.lr - lo) * (1 + math.cos(math.pi * min(p, 1.0)))

    decay_start = int(total * (1 - cfg.decay_frac))
    if step < decay_start:
        return cfg.lr
    p = (step - decay_start) / max(total - decay_start, 1)
    return cfg.lr * (lo / cfg.lr) ** min(p, 1.0)      # exponential decay to min


def build_optimizer(model, cfg):
    """Weight decay on matrices only. Decaying norms, biases, embeddings and the
    SSD's A/D/dt parameters measurably hurts — they are not scale-free."""
    no_decay_names = ("A_log", "D", "dt_bias")
    decay, plain = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or n.endswith(no_decay_names) or "embed" in n:
            plain.append(p)
        else:
            decay.append(p)

    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": plain, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
    )
    return opt, {"decayed": sum(p.numel() for p in decay),
                 "not_decayed": sum(p.numel() for p in plain)}
