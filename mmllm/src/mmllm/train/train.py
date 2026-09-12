"""Training loop.

Handles the things that actually kill long runs: resumable data order, loss-spike
detection with rollback, grad-norm monitoring, and atomic checkpoints.

    python -m mmllm.train.train --config configs/tiny_cpu.yaml
    python -m mmllm.train.train --config configs/tiny_cpu.yaml --resume
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import torch

from ..config import Config
from ..data.dataset import PackedDataset
from ..model import MMLLM
from .schedule import build_optimizer, lr_at


def evaluate(model, ds, cfg, iters: int) -> float:
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for i in range(iters):
            x, y = ds.batch(1_000_000 + i, cfg.micro_batch, cfg.device)
            _, loss = model(x, targets=y)
            tot += loss.item()
    model.train()
    return tot / max(iters, 1)


def save(path, model, opt, step, best, cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "best": best, "config": asdict(cfg)}, tmp)
    os.replace(tmp, path)      # atomic: a crash mid-save cannot corrupt the ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    tc, mc = cfg.train, cfg.model
    torch.manual_seed(tc.seed)
    if tc.num_threads:
        torch.set_num_threads(tc.num_threads)
    os.makedirs(tc.out_dir, exist_ok=True)

    dtype = getattr(torch, tc.dtype)
    train_ds = PackedDataset(cfg.data.train_bin, tc.seq_len, cfg.data.dtype, tc.seed)
    val_ds = PackedDataset(cfg.data.val_bin, tc.seq_len, cfg.data.dtype, tc.seed + 1)

    model = MMLLM(mc).to(tc.device, dtype=dtype)
    opt, pstats = build_optimizer(model, tc)

    tok_per_step = tc.seq_len * tc.micro_batch * tc.grad_accum
    print(f"run       : {cfg.name}")
    print(f"params    : {model.n_params():,} total / "
          f"{model.n_params(embedding=False):,} non-embedding")
    print(f"layers    : {mc.kinds()}")
    print(f"tokens/step: {tok_per_step:,}   total: {tok_per_step * tc.max_steps:,}")
    print(f"flops (6ND): {6 * model.n_params() * tok_per_step * tc.max_steps:.2e}")

    step, best = 0, float("inf")
    ckpt_path = os.path.join(tc.out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=tc.device, weights_only=False)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        step, best = sd["step"], sd["best"]
        print(f"resumed at step {step}")

    logf = open(os.path.join(tc.out_dir, "log.jsonl"), "a", encoding="utf-8")
    recent, t0, last_log = [], time.perf_counter(), step
    model.train()

    while step < tc.max_steps:
        lr = lr_at(step, tc)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for micro in range(tc.grad_accum):
            x, y = train_ds.batch(step * tc.grad_accum + micro, tc.micro_batch, tc.device)
            _, loss = model(x, targets=y)
            (loss / tc.grad_accum).backward()
            acc += loss.item() / tc.grad_accum

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()

        # Spike guard: a single bad shard can wreck a run that has been going for
        # days. Skip the update rather than let it propagate.
        med = sorted(recent)[len(recent) // 2] if len(recent) >= 20 else None
        if not math.isfinite(acc) or (med is not None and acc > med + 0.5):
            print(f"step {step}: skipping update (loss {acc:.4f} vs median {med}) ")
            opt.zero_grad(set_to_none=True)
            step += 1
            continue

        opt.step()
        recent.append(acc)
        recent = recent[-100:]

        if step % tc.log_every == 0:
            dt = time.perf_counter() - t0
            # Count steps actually timed. An eval resets the timer mid-window, so
            # assuming log_every steps here inflates throughput ~10x right after one.
            tps = tok_per_step * max(step - last_log, 1) / dt
            rec = {"step": step, "loss": round(acc, 4), "ppl": round(math.exp(min(acc, 20)), 2),
                   "lr": round(lr, 8), "gnorm": round(gnorm, 3), "tok_s": round(tps, 1)}
            print(" ".join(f"{k}={v}" for k, v in rec.items()))
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            t0, last_log = time.perf_counter(), step

        step += 1

        if step % tc.eval_every == 0:
            vl = evaluate(model, val_ds, tc, tc.eval_iters)
            print(f"  val loss={vl:.4f} ppl={math.exp(min(vl, 20)):.2f}")
            logf.write(json.dumps({"step": step, "val_loss": round(vl, 4)}) + "\n")
            logf.flush()
            if vl < best:
                best = vl
                save(os.path.join(tc.out_dir, "best.pt"), model, opt, step, best, cfg)
            t0, last_log = time.perf_counter(), step

        if step % tc.ckpt_every == 0:
            save(ckpt_path, model, opt, step, best, cfg)

    save(ckpt_path, model, opt, step, best, cfg)
    print(f"done. best val loss {best:.4f} -> {tc.out_dir}")
    logf.close()


if __name__ == "__main__":
    main()
