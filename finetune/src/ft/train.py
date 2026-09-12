"""LoRA supervised fine-tuning loop.

    PYTHONPATH=src python -m ft.train --config configs/smol135m_cpu.yaml
    PYTHONPATH=src python -m ft.train --config configs/smol135m_cpu.yaml --resume
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict

import torch

from .config import Config
from .data import SFTDataset, collate
from .lora import inject_lora, lora_state_dict
from .model_io import load_base, trainable_report


def lr_at(step: int, total: int, cfg) -> float:
    warm = max(int(cfg.warmup_frac * total), 1)
    if step < warm:
        return cfg.lr * (step + 1) / warm
    p = (step - warm) / max(total - warm, 1)
    lo = cfg.lr * cfg.min_lr_frac
    return lo + 0.5 * (cfg.lr - lo) * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def evaluate(model, ds, pad_id, cfg, max_batches: int) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, min(len(ds), max_batches * cfg.micro_batch), cfg.micro_batch):
        batch = collate([ds[j] for j in range(i, min(i + cfg.micro_batch, len(ds)))], pad_id)
        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        tot += model(**batch).loss.item()
        n += 1
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="override, for smoke runs")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    tc = cfg.train
    torch.manual_seed(tc.seed)
    random.seed(tc.seed)
    if tc.num_threads:
        torch.set_num_threads(tc.num_threads)
    os.makedirs(tc.out_dir, exist_ok=True)

    print(f"base model : {cfg.base_model}")
    model, tok = load_base(cfg.base_model, tc.dtype, tc.device)
    info = inject_lora(model, cfg.lora)
    print(f"lora       : r={cfg.lora.r} alpha={cfg.lora.alpha} "
          f"-> {info['replaced']} layers replaced")
    print(f"params     : {trainable_report(model)}")

    train_ds = SFTDataset(cfg.data.train_file, tok, cfg.data.max_len,
                          cfg.data.on_overflow, cfg.data.mask_prompt)
    val_ds = SFTDataset(cfg.data.val_file, tok, cfg.data.max_len,
                        cfg.data.on_overflow, cfg.data.mask_prompt)
    print(f"data       : {len(train_ds)} train / {len(val_ds)} val  {train_ds.stats}")
    train_ds.warn_if_truncating("train")
    val_ds.warn_if_truncating("val")
    sup = train_ds.supervised_fraction()
    print(f"supervised : {sup:.1%} of tokens carry loss"
          + ("   <-- SUSPICIOUS: prompt masking may be off" if sup > 0.9 else ""))

    steps_per_epoch = max(len(train_ds) // (tc.micro_batch * tc.grad_accum), 1)
    total_steps = args.max_steps or int(steps_per_epoch * tc.epochs)
    print(f"schedule   : {total_steps} steps "
          f"({steps_per_epoch}/epoch x {tc.epochs} epochs)")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay, betas=(0.9, 0.999))

    step, best = 0, float("inf")
    ckpt = os.path.join(tc.out_dir, "adapter.pt")
    if args.resume and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location=tc.device, weights_only=False)
        model.load_state_dict(sd["lora"], strict=False)
        opt.load_state_dict(sd["opt"])
        step, best = sd["step"], sd["best"]
        print(f"resumed at step {step}")

    def save(path, s, b):
        tmp = path + ".tmp"
        torch.save({"lora": lora_state_dict(model), "opt": opt.state_dict(),
                    "step": s, "best": b, "config": asdict(cfg)}, tmp)
        os.replace(tmp, path)

    base_val = evaluate(model, val_ds, tok.pad_token_id, tc, tc.eval_batches)
    # B=0 at init, so this is the untouched base model's loss. Everything after is
    # measured against it; if it does not improve, nothing else in the run matters.
    print(f"baseline   : val loss {base_val:.4f} (base model, adapter is identity)")

    logf = open(os.path.join(tc.out_dir, "log.jsonl"), "a", encoding="utf-8")
    order = list(range(len(train_ds)))
    rng = random.Random(tc.seed)
    cursor = step * tc.micro_batch * tc.grad_accum
    model.train()
    t0 = time.perf_counter()
    tok_seen = 0

    while step < total_steps:
        lr = lr_at(step, total_steps, tc)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(tc.grad_accum):
            if cursor % len(order) == 0:
                rng.shuffle(order)
            idx = [order[(cursor + k) % len(order)] for k in range(tc.micro_batch)]
            cursor += tc.micro_batch
            batch = collate([train_ds[i] for i in idx], tok.pad_token_id)
            batch = {k: v.to(tc.device) for k, v in batch.items()}
            tok_seen += int(batch["attention_mask"].sum())
            loss = model(**batch).loss
            (loss / tc.grad_accum).backward()
            acc += loss.item() / tc.grad_accum

        gnorm = torch.nn.utils.clip_grad_norm_(params, tc.grad_clip).item()
        opt.step()
        step += 1

        if step % tc.log_every == 0:
            dt = time.perf_counter() - t0
            rec = {"step": step, "loss": round(acc, 4), "lr": round(lr, 8),
                   "gnorm": round(gnorm, 3), "tok_s": round(tok_seen / dt, 1)}
            print(" ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            t0, tok_seen = time.perf_counter(), 0

        if step % tc.eval_every == 0 or step == total_steps:
            vl = evaluate(model, val_ds, tok.pad_token_id, tc, tc.eval_batches)
            delta = vl - base_val
            print(f"  val loss={vl:.4f}  (baseline {base_val:.4f}, {delta:+.4f})", flush=True)
            logf.write(json.dumps({"step": step, "val_loss": round(vl, 4),
                                   "vs_baseline": round(delta, 4)}) + "\n"); logf.flush()
            if vl < best:
                best = vl
                save(os.path.join(tc.out_dir, "adapter_best.pt"), step, best)
            t0, tok_seen = time.perf_counter(), 0

        if step % tc.save_every == 0:
            save(ckpt, step, best)

    save(ckpt, step, best)
    print(f"done. baseline {base_val:.4f} -> best {best:.4f} "
          f"({best - base_val:+.4f}) -> {tc.out_dir}")
    logf.close()


if __name__ == "__main__":
    main()
