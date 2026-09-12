"""Generate from a trained checkpoint.

    PYTHONPATH=src python scripts/sample.py --ckpt runs/smoke/best.pt \
        --tokenizer data/tokenizer.json --prompt "def parse(" --tokens 80
"""
from __future__ import annotations

import argparse

import torch

from mmllm.config import Config, ModelConfig, VisionConfig
from mmllm.data.tokenizer import BOS, load_tokenizer
from mmllm.model import MMLLM


def load(ckpt_path: str, device: str = "cpu"):
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw = sd["config"]["model"]
    vis = VisionConfig(**raw.pop("vision"))
    cfg = ModelConfig(**{**raw, "vision": vis})
    model = MMLLM(cfg)
    model.load_state_dict(sd["model"])
    return model.eval().to(device), cfg, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, cfg, sd = load(args.ckpt, args.device)
    tok = load_tokenizer(args.tokenizer)

    print(f"checkpoint : {args.ckpt}  (step {sd['step']}, best val {sd['best']:.4f})")
    print(f"params     : {model.n_params():,}   layers: {cfg.kinds()}")

    ids = [BOS] + (tok.encode(args.prompt).ids if args.prompt else [])
    x = torch.tensor([ids], dtype=torch.long, device=args.device)
    out = model.generate(x, max_new_tokens=args.tokens,
                         temperature=args.temperature, top_k=args.top_k)

    print("-" * 60)
    print(tok.decode(out[0].tolist()))
    print("-" * 60)


if __name__ == "__main__":
    main()
