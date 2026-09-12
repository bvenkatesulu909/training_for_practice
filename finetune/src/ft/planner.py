"""Fine-tuning cost planner.

Answers: given this base model, this dataset and this hardware, how long does the
run take and does it fit in memory? Run it before committing to an overnight CPU
run or a GPU rental.

    PYTHONPATH=src python -m ft.planner --config configs/smol135m_cpu.yaml
    PYTHONPATH=src python -m ft.planner --config configs/qwen7b_gpu.yaml \
        --gflops 989000 --devices 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .config import Config

# FLOPs per parameter per token.
#   forward                        = 2
#   backward w.r.t. activations    = 2      (needed even when weights are frozen)
#   backward w.r.t. weights        = 2      (LoRA skips this for the frozen base)
FLOPS_LORA = 4
FLOPS_FULL = 6


def measure_gflops(device="cpu", n=1024) -> float:
    a, b = torch.randn(n, n, device=device), torch.randn(n, n, device=device)
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


def activation_bytes(dims: dict, seq: int, micro_batch: int, bpp: int = 2,
                     checkpointing: bool = True) -> dict:
    """Activation memory — the term that actually decides which GPU you need.

    At 1k context it is a rounding error next to the frozen weights. At 16k it is
    the difference between fitting on a 24 GB card and not, so the planner should
    compute it rather than wave at it.

    With gradient checkpointing only layer boundaries are kept, plus the full
    working set of the single layer being recomputed.
    """
    d, m, L = dims["d"], dims["m"], dims["L"]
    tok = seq * micro_batch
    boundaries = tok * d * L * bpp                      # one tensor per layer
    one_layer = tok * (4 * d + 3 * m) * bpp             # qkv+o and gate/up/down of one block
    if checkpointing:
        return {"boundaries": boundaries, "peak_layer": one_layer,
                "total": boundaries + one_layer}
    full = tok * (d * 6 + m * 3) * L * bpp              # every layer retained
    return {"boundaries": full, "peak_layer": 0, "total": full}


def human(sec: float) -> str:
    if sec < 3600:
        return f"{sec/60:.1f} min"
    if sec < 86400:
        return f"{sec/3600:.1f} hours"
    return f"{sec/86400:.1f} days"


def model_dims(base_model: str) -> dict | None:
    """Fetch just config.json (a few KB) — never the weights.

    Pricing a 7B run must not require downloading 15 GB first.
    """
    try:
        from huggingface_hub import hf_hub_download
        import json as _json
        with open(hf_hub_download(base_model, "config.json"), encoding="utf-8") as f:
            c = _json.load(f)
    except Exception:
        return None
    d = c.get("hidden_size")
    if not d:
        return None
    nh = c.get("num_attention_heads", 1)
    hd = c.get("head_dim") or d // max(nh, 1)
    return {"d": d, "m": c.get("intermediate_size", 4 * d), "L": c.get("num_hidden_layers", 1),
            "nh": nh, "nkv": c.get("num_key_value_heads", nh), "hd": hd,
            "vocab": c.get("vocab_size", 32000), "tied": c.get("tie_word_embeddings", False)}


def shapes(dims: dict) -> dict:
    """(in_features, out_features) of every LoRA-targetable projection."""
    d, m, q, kv = dims["d"], dims["m"], dims["nh"] * dims["hd"], dims["nkv"] * dims["hd"]
    return {"q_proj": (d, q), "k_proj": (d, kv), "v_proj": (d, kv), "o_proj": (q, d),
            "gate_proj": (d, m), "up_proj": (d, m), "down_proj": (m, d)}


def adapter_params(dims: dict, r: int, targets) -> int:
    """Exact LoRA parameter count: r*(in+out) per targeted layer, per block."""
    sh = shapes(dims)
    per_layer = sum(r * (i + o) for n, (i, o) in sh.items() if n in set(targets))
    return per_layer * dims["L"]


def base_params(dims: dict) -> int:
    """Analytic parameter count of the base model from its config alone."""
    d, m, L, vocab = dims["d"], dims["m"], dims["L"], dims["vocab"]
    sh = shapes(dims)
    attn = sum(i * o for n, (i, o) in sh.items() if n.endswith(("q_proj", "k_proj", "v_proj", "o_proj")))
    mlp = sum(i * o for n, (i, o) in sh.items() if n.endswith(("gate_proj", "up_proj", "down_proj")))
    embed = vocab * d * (1 if dims["tied"] else 2)
    return embed + L * (attn + mlp + 2 * d) + d


def count_tokens(path: str, chars_per_token: float = 3.6) -> tuple[int, int]:
    """Approximate token count without loading a tokenizer (which needs the model)."""
    from .data import iter_jsonl
    n_ex = n_chars = 0
    for rec in iter_jsonl(path):
        n_ex += 1
        msgs = rec.get("messages") or []
        n_chars += sum(len(m.get("content", "")) for m in msgs) if msgs else \
            len(rec.get("instruction", "")) + len(rec.get("input", "")) + len(rec.get("output", ""))
    return n_ex, int(n_chars / chars_per_token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--params", type=float, default=None, help="base params, e.g. 7e9")
    ap.add_argument("--gflops", type=float, default=None)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--mfu", type=float, default=0.35)
    ap.add_argument("--price", type=float, default=2.5, help="$ per device-hour")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    # Parameter counts for the common bases, so the planner never needs to download
    # a model just to price it.
    KNOWN = {"SmolLM2-135M": 1.35e8, "SmolLM2-360M": 3.6e8, "Qwen2.5-0.5B": 4.9e8,
             "Qwen2.5-1.5B": 1.54e9, "Qwen2.5-3B": 3.09e9, "Qwen2.5-7B": 7.62e9,
             "Qwen3-0.6B": 6.0e8, "Qwen3-4B": 4.02e9, "Llama-3.2-1B": 1.24e9,
             "Llama-3.2-3B": 3.21e9, "Llama-3.1-8B": 8.03e9, "gemma-2-2b": 2.61e9}
    dims = model_dims(cfg.base_model)
    N = args.params
    if N is None and dims:
        N = base_params(dims)
    if N is None:
        for k, v in KNOWN.items():
            if k.lower() in cfg.base_model.lower():
                N = v
                break
    if N is None:
        raise SystemExit(f"cannot resolve {cfg.base_model}; pass --params")

    n_ex, n_tok = count_tokens(cfg.data.train_file)
    tokens = n_tok * cfg.train.epochs

    r, mods = cfg.lora.r, len(cfg.lora.target_modules)
    # Exact, from the model config — verified against the count `inject_lora`
    # reports at train time. An estimate here was wrong by 12x.
    adapter = adapter_params(dims, r, cfg.lora.target_modules) if dims else float("nan")

    peak = args.gflops or measure_gflops(cfg.train.device)
    eff = peak * args.mfu * args.devices * 1e9

    print(f"\n=== {cfg.name} ===")
    print(f"base model      : {cfg.base_model}  ({N/1e9:.2f}B params)")
    print(f"dataset         : {n_ex:,} examples, ~{n_tok:,} tokens")
    print(f"epochs          : {cfg.train.epochs}  -> {tokens:,.0f} training tokens")
    print(f"lora            : r={r}, {mods} projections  -> {adapter/1e6:.2f}M trainable "
          f"({100*adapter/N:.2f}% of base)")

    print(f"\n--- compute ---")
    print(f"peak            : {peak:,.0f} GFLOP/s x {args.devices}, MFU {args.mfu:.0%}")
    for label, fpp, n_eff in (("LoRA       ", FLOPS_LORA, N), ("full fine-tune", FLOPS_FULL, N)):
        fl = fpp * n_eff * tokens
        secs = fl / eff
        cost = secs / 3600 * args.price * args.devices
        print(f"{label}  : {fl:.2e} FLOPs -> {human(secs):>12}   ~${cost:,.2f}")

    print(f"\n--- memory ---")
    bpp = 2 if cfg.train.dtype == "bfloat16" else 4
    lora_mem = N * bpp + adapter * 16          # frozen base + fp32 master/grad/adam
    full_mem = N * bpp + N * 4 + N * 8         # weights + grads + adam m,v
    print(f"LoRA weights    : {lora_mem/2**30:6.2f} GiB  "
          f"(frozen base {N*bpp/2**30:.2f} + adapter states {adapter*16/2**30:.2f})")
    print(f"full fine-tune  : {full_mem/2**30:6.2f} GiB  ({full_mem/max(lora_mem,1):.1f}x more)")

    if dims:
        seq, mb = cfg.data.max_len, cfg.train.micro_batch
        ck = activation_bytes(dims, seq, mb, bpp, checkpointing=True)
        no = activation_bytes(dims, seq, mb, bpp, checkpointing=False)
        print()
        print(f"activations @ seq {seq:,} x micro_batch {mb}")
        print(f"  with grad checkpointing : {ck['total']/2**30:6.2f} GiB")
        print(f"  without                 : {no['total']/2**30:6.2f} GiB")
        peak = lora_mem + ck["total"]
        print()
        print(f"PEAK for LoRA (weights + activations, checkpointed): {peak/2**30:6.2f} GiB")
        for name, cap in (("24 GB (L4/4090)", 24), ("48 GB (A6000)", 48), ("80 GB (A100/H100)", 80)):
            fits = "FITS" if peak/2**30 < cap * 0.88 else "does NOT fit"
            print(f"  {name:20s}: {fits}")


if __name__ == "__main__":
    main()
