"""Fold a trained adapter into the base weights and export a standalone model.

After merging there are no custom classes left — the result is an ordinary HF
model directory that vLLM, llama.cpp or Ollama can load with no knowledge of LoRA.

    PYTHONPATH=src python -m ft.merge --adapter runs/x/adapter_best.pt --out export/
"""
from __future__ import annotations

import argparse

import torch

from .config import Config
from .lora import inject_lora, merge_lora
from .model_io import load_base


def load_adapted(adapter_path: str, device: str = "cpu", dtype: str = "float32"):
    sd = torch.load(adapter_path, map_location=device, weights_only=False)
    raw = sd["config"]
    cfg = Config.from_dict(raw) if isinstance(raw, dict) else raw
    model, tok = load_base(cfg.base_model, dtype, device)
    inject_lora(model, cfg.lora)
    missing, unexpected = model.load_state_dict(sd["lora"], strict=False)
    assert not unexpected, f"unexpected adapter keys: {unexpected[:5]}"
    assert any("lora_" in k for k in sd["lora"]), "adapter file contains no LoRA weights"
    return model, tok, cfg, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    model, tok, cfg, sd = load_adapted(args.adapter, dtype=args.dtype)
    print(f"base    : {cfg.base_model}   adapter step {sd['step']} (val {sd['best']:.4f})")

    n = merge_lora(model)
    print(f"merged  : {n} LoRA layers folded into base weights")
    assert not any("lora_" in k for k in model.state_dict()), "LoRA layers survived merge"

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"exported: {args.out}  (loadable with AutoModelForCausalLM.from_pretrained)")


if __name__ == "__main__":
    main()
