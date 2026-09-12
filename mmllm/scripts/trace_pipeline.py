"""Walk one real batch through every stage of the training pipeline and print it.

This exists to be *checked*, not trusted. Each section prints the actual tensors
and numbers at one stage of:

    raw text -> clean -> tokenize -> ids -> embed -> layers -> attention/NN
    -> predict next token -> compare to target -> loss -> backward -> grads
    -> optimizer -> weight update -> repeat

    PYTHONPATH=src python scripts/trace_pipeline.py
"""
from __future__ import annotations

import sys

import numpy as np

# BPE pieces contain U+0120 (the byte-level space marker); the Windows console
# defaults to cp1252 and would raise on it.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch

from mmllm.config import Config
from mmllm.data.dataset import PackedDataset
from mmllm.data.prepare import MinHashDedup, quality_ok
from mmllm.data.tokenizer import load_tokenizer
from mmllm.model import MMLLM
from mmllm.train.schedule import build_optimizer, lr_at

BAR = "=" * 78


def head(n, title):
    print(f"\n{BAR}\n[{n}] {title}\n{BAR}")


def main():
    torch.manual_seed(0)
    cfg = Config.load("configs/smoke.yaml")
    tc, mc = cfg.train, cfg.model

    # ---------------------------------------------------------------- 1 & 2
    head(1, "PLAIN TEXT -> RAW TRAINING DATA")
    raw_good = ("import os\nimport sys\n\n"
                "def tokenize(text: str) -> list[int]:\n"
                "    \"\"\"Convert text to token ids using the trained BPE vocab.\"\"\"\n"
                "    return [vocab.get(t, UNK) for t in split(text)]\n\n"
                "def detokenize(ids: list[int]) -> str:\n"
                "    return ''.join(inv_vocab[i] for i in ids)\n\n"
                "class Encoder:\n"
                "    def __init__(self, merges, specials):\n"
                "        self.merges, self.specials = merges, specials\n")
    raw_junk = "click here | click here | click here | " * 12
    print(f"good sample ({len(raw_good)} chars): {raw_good[:60]!r}...")
    print(f"junk sample ({len(raw_junk)} chars): {raw_junk[:60]!r}...")

    # -------------------------------------------------------------------- 3
    head(3, "DATA CLEANING / PREPARATION")
    from mmllm.data.prepare import dup_ngram_frac
    print(f"quality_ok(good code) = {quality_ok(raw_good, True)}   "
          f"dup-5gram={dup_ngram_frac(raw_good.split(),5):.3f}")
    print(f"quality_ok(junk)      = {quality_ok(raw_junk, False)}  "
          f"dup-5gram={dup_ngram_frac(raw_junk.split(),5):.3f}  <- repetition filter")
    dd = MinHashDedup()
    print(f"MinHash first sighting          = {dd.is_duplicate(raw_good)}  (not a duplicate)")
    print(f"MinHash second sighting         = {dd.is_duplicate(raw_good)}  <- near-dup caught")

    # -------------------------------------------------------------------- 4
    head(4, "TOKENIZER")
    tok = load_tokenizer(cfg.data.tokenizer)
    print(f"trained byte-level BPE, vocab_size = {tok.get_vocab_size()}")
    enc = tok.encode("def tokenize(text):")
    print(f"pieces = {enc.tokens}")

    # -------------------------------------------------------------------- 5
    head(5, "TOKENS -> TOKEN IDS")
    print(f"ids    = {enc.ids}")
    data = np.memmap(cfg.data.train_bin, dtype=np.uint16, mode="r")
    print(f"corpus stored as uint16 memmap: {len(data):,} tokens on disk")
    ds = PackedDataset(cfg.data.train_bin, tc.seq_len, cfg.data.dtype, tc.seed)
    x, y = ds.batch(step=0, micro_batch=2)
    print(f"batch inputs  x: {tuple(x.shape)}  dtype={x.dtype}")

    # -------------------------------------------------------------------- 6
    head(6, "EMBEDDINGS")
    model = MMLLM(mc)
    emb = model.embed(x)
    print(f"nn.Embedding({mc.vocab_size}, {mc.dim})")
    print(f"token ids {tuple(x.shape)} -> vectors {tuple(emb.shape)}")
    print(f"first token id {x[0,0].item()} -> vector[:6] = "
          f"{emb[0,0,:6].detach().numpy().round(4)}")

    # ----------------------------------------------------------------- 7 & 8
    head(7, "TRANSFORMER LAYERS")
    print(f"{mc.n_layers} blocks, each: RMSNorm -> mixer -> RMSNorm -> SwiGLU MLP")
    print(f"mixer per layer: {mc.kinds()}")
    h = emb
    for i, blk in enumerate(model.blocks):
        h, _ = blk(h, model.cos, model.sin, None, 0)
        print(f"  layer {i} ({blk.kind:6s}) -> {tuple(h.shape)}  "
              f"mean={h.mean().item():+.4f} std={h.std().item():.4f}")

    head(8, "ATTENTION + NEURAL NETWORK COMPUTATION")
    print("attention layers  : F.scaled_dot_product_attention, GQA "
          f"({mc.n_heads} q heads / {mc.n_kv_heads} kv heads)")
    print("ssd layers        : chunked linear-time state-space scan (ssd_chunked)")
    print("feed-forward      : SwiGLU  w2(silu(w1 x) * w3 x)")

    # -------------------------------------------------------------------- 9
    head(9, "PREDICT NEXT TOKEN")
    h = model.norm(h)
    logits = model.lm_head(h)
    print(f"lm_head: {tuple(h.shape)} -> {tuple(logits.shape)}   (one score per vocab entry)")
    pos = 5
    probs = torch.softmax(logits[0, pos].float(), -1)
    top = torch.topk(probs, 5)
    print(f"at position {pos}, top-5 predictions for the NEXT token:")
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"    id {i:5d}  p={p:.4f}  {tok.decode([i])!r}")

    # ------------------------------------------------------------------- 10
    head(10, "COMPARE PREDICTION WITH CORRECT TOKEN")
    print(f"input  x[0, {pos}]   = {x[0,pos].item():5d}  {tok.decode([x[0,pos].item()])!r}")
    print(f"target y[0, {pos}]   = {y[0,pos].item():5d}  {tok.decode([y[0,pos].item()])!r}")
    print(f"y is x shifted left by one -> x[0,{pos+1}] == y[0,{pos}] : "
          f"{x[0,pos+1].item() == y[0,pos].item()}")
    print(f"model gave the correct token p={probs[y[0,pos]].item():.6f}")

    # ------------------------------------------------------------------- 11
    head(11, "CALCULATE LOSS")
    _, loss = model(x, targets=y)
    print(f"cross_entropy over {x.numel():,} positions = {loss.item():.4f}")
    print(f"untrained baseline ln(vocab) = {np.log(mc.vocab_size):.4f}   "
          f"(random guessing)")
    print(f"perplexity = {np.exp(loss.item()):.1f}")

    # -------------------------------------------------------------- 12 & 13
    head(12, "BACKPROPAGATION -> GRADIENTS")
    opt, stats = build_optimizer(model, tc)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    have = [(n, p.grad.norm().item()) for n, p in model.named_parameters()
            if p.grad is not None]
    print(f"loss.backward() populated grads on {len(have)}/"
          f"{sum(1 for _ in model.parameters())} parameter tensors")
    # Scientific notation on purpose: the SSD decay parameters carry gradients
    # around 1e-7 at init, which fixed-point formatting renders as a misleading 0.
    for n, g in have[:5]:
        print(f"    {n:46s} |grad| = {g:.3e}")
    smallest = min(have, key=lambda t: t[1])
    print(f"    smallest of all {len(have)}: {smallest[0]} = {smallest[1]:.3e} "
          f"(nonzero -> no dead branch)")
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
    print(f"global grad norm = {gnorm.item():.4f}  (clipped at {tc.grad_clip})")

    # -------------------------------------------------------------- 14 & 15
    head(14, "OPTIMIZER -> UPDATE WEIGHTS")
    w = model.blocks[0].mlp.w2.weight
    before = w.detach().clone()
    lr = lr_at(0, tc)
    for g in opt.param_groups:
        g["lr"] = lr
    print(f"AdamW  lr={lr:.3e}  betas=({tc.beta1}, {tc.beta2})  wd={tc.weight_decay}")
    print(f"  decayed params {stats['decayed']:,} / not decayed {stats['not_decayed']:,}")
    opt.step()
    delta = (w.detach() - before).abs()
    print(f"weights changed: max |dW| = {delta.max().item():.3e}, "
          f"mean |dW| = {delta.mean().item():.3e}")
    assert delta.max() > 0, "optimizer did not move the weights"

    # ------------------------------------------------------------------- 16
    head(16, "REPEAT")
    _, loss2 = model(x, targets=y)
    print(f"same batch, loss before update = {loss.item():.4f}")
    print(f"same batch, loss after  update = {loss2.item():.4f}   "
          f"({loss2.item()-loss.item():+.4f})")
    print(f"\ntrain.py runs this loop: while step < max_steps  "
          f"(x {tc.grad_accum} micro-batches per step)")
    print(f"this config: {tc.max_steps} steps x "
          f"{tc.seq_len*tc.micro_batch*tc.grad_accum:,} tokens/step = "
          f"{tc.max_steps*tc.seq_len*tc.micro_batch*tc.grad_accum:,} tokens total")
    print("a frontier run repeats it for ~10^12-10^13 tokens; see README for the gap.")


if __name__ == "__main__":
    main()
