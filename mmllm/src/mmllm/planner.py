"""Hardware-aware training planner.

Answers the only question that matters before you start: *given this machine, how
long does this config take, and does it fit in memory?* Run it before every scale-up.

    python -m mmllm.planner --config configs/base_300k.yaml
    python -m mmllm.planner --config configs/base_300k.yaml --gflops 989000 --devices 8
"""
from __future__ import annotations

import argparse
import time

import torch

from .config import Config
from .model.budget import analytic_params, kv_cache_bytes, training_flops

YEAR = 365 * 24 * 3600


def measure_gflops(device="cpu", n=1024, dtype=torch.float32) -> float:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(3):
        a @ b
    if device != "cpu":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        a @ b
    if device != "cpu":
        torch.cuda.synchronize()
    return 2 * n ** 3 * 10 / (time.perf_counter() - t) / 1e9


def human_time(sec: float) -> str:
    if sec < 3600:
        return f"{sec/60:.1f} min"
    if sec < 86400:
        return f"{sec/3600:.1f} hours"
    if sec < YEAR:
        return f"{sec/86400:.1f} days"
    return f"{sec/YEAR:,.0f} YEARS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gflops", type=float, default=None, help="peak GFLOP/s per device")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--mfu", type=float, default=0.4, help="achieved fraction of peak")
    ap.add_argument("--tokens", type=float, default=None, help="override total token budget")
    ap.add_argument("--ctx", type=int, default=300_000, help="target inference context")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="override training seq len (use for context-extension phases)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.seq_len:
        cfg.train.seq_len = args.seq_len
    # Analytic: a 2B-param config must be priceable on a laptop that cannot hold it.
    pp = analytic_params(cfg.model)
    N, Nne = pp["total"], pp["non_embedding"]

    peak = args.gflops if args.gflops else measure_gflops()
    eff = peak * args.mfu * args.devices * 1e9

    tokens = args.tokens or (cfg.train.seq_len * cfg.train.micro_batch
                             * cfg.train.grad_accum * cfg.train.max_steps)
    fl = training_flops(cfg.model, tokens, cfg.train.seq_len)
    flops = fl["total"]

    print(f"\n=== {cfg.name} ===")
    print(f"parameters            : {N/1e6:,.1f}M total, {Nne/1e6:,.1f}M non-embedding")
    print(f"  embedding / layers / vision: {pp['embedding']/1e6:,.1f}M / "
          f"{pp['layers']/1e6:,.1f}M / {pp['vision']/1e6:,.1f}M")
    print(f"  layer mix           : {pp['layer_counts']}")
    print(f"layer pattern         : {cfg.model.kinds()}")
    print(f"train seq len         : {cfg.train.seq_len:,}")
    print(f"token budget          : {tokens:,.0f}  ({tokens/max(N,1):,.0f} tokens/param)")
    print(f"  chinchilla-optimal  : {20*N:,.0f}   inference-optimal: {200*N:,.0f}")

    print(f"\n--- compute ---")
    print(f"measured/assumed peak : {peak:,.1f} GFLOP/s x {args.devices} device(s)")
    print(f"assumed MFU           : {args.mfu:.0%}  -> {eff/1e9:,.1f} GFLOP/s effective")
    print(f"dense FLOPs (6ND)     : {fl['dense']:.3e}")
    print(f"attention FLOPs       : {fl['attention']:.3e}  "
          f"({fl['attention_frac']:.1%} of total at seq_len {cfg.train.seq_len:,})")
    print(f"TOTAL training FLOPs  : {flops:.3e}")
    print(f"WALL CLOCK            : {human_time(flops/eff)}")
    gpu_hours = flops / (peak * args.mfu * 1e9) / 3600
    print(f"GPU-hours             : {gpu_hours:,.0f}  "
          f"(~${gpu_hours*2:,.0f}-${gpu_hours*3:,.0f} at $2-3/GPU-hr)")

    print(f"\n--- memory: training (AdamW, fp32 master) ---")
    print(f"params  fp32          : {N*4/2**30:6.2f} GiB")
    print(f"grads   fp32          : {N*4/2**30:6.2f} GiB")
    print(f"adam m,v              : {N*8/2**30:6.2f} GiB")
    print(f"TOTAL (excl. activations): {N*16/2**30:6.2f} GiB")

    print(f"\n--- memory: inference KV cache @ {args.ctx:,} ctx (bf16) ---")
    kv = kv_cache_bytes(cfg.model, args.ctx)
    for k in ("global", "swa", "ssd", "total"):
        print(f"{k:22s}: {kv[k]/2**30:6.2f} GiB")
    for label, key in (("same GQA, all attention", "all_attention_equivalent"),
                       ("conventional full MHA  ", "all_mha_equivalent")):
        print(f"vs {label}: {kv[key]/2**30:8.2f} GiB "
              f"({kv[key]/max(kv['total'],1):.1f}x more)")

    v = cfg.model.vision
    if v.enabled:
        print(f"\n--- modality budget @ {args.ctx:,} tokens ---")
        print(f"code/text             : ~{args.ctx*4/1000:,.0f}k chars (~{args.ctx*4//40:,} lines)")
        print(f"images ({v.tokens_per_image} tok each)   : {args.ctx//v.tokens_per_image:,} images")
        tpf = max(v.tokens_per_frame, 1)
        print(f"video  ({tpf} tok/frame)  : {args.ctx//tpf:,} frames "
              f"= {args.ctx//tpf/60:,.0f} min @ 1 fps")


if __name__ == "__main__":
    main()
