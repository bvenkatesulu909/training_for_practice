# mmllm — a 300k-context multimodal LM, built from scratch

Text, code, image and video in one causal stream, designed around a **300,000-token
context**. Pure PyTorch, no CUDA-only dependencies, runs on CPU for validation and
scales to a GPU cluster with one config swap.

---

## Read this before you start

The architecture here is buildable on any machine. **Training it is not.**

| Target | Params × Tokens | FLOPs (6ND) | On an i3-3110M @ ~10 GFLOP/s |
|---|---|---|---|
| 300k ctx, code+text+image+video, actually good | 3B × 2T | 3.6e22 | **~114,000 years** |
| Minimum viable multimodal | 1B × 100B | 6.0e20 | ~1,900 years |
| Smallest model that forms sentences | 100M × 5B | 3.0e18 | ~9.5 years |
| What fits in one day on this laptop | 6.5M × 26M | 9.6e14 | ~1 day *(measured)* |

No flag closes a 10⁷ gap. So the laptop's job is to **prove the architecture is
correct**, and the real run happens on rented hardware. `python -m mmllm.planner`
prints this table for any config against any hardware — run it before every scale-up.

The last row is measured, not projected: the `smoke` run below sustains **~300 tok/s**
on a 6.5M-param hybrid, i.e. ~11.7 GFLOP/s effective against a 24.2 GFLOP/s matmul
ceiling (48% MFU). One day at that rate is 26M tokens.

---

## Why this architecture

A 300k context is a memory problem before it is a modelling problem. Full attention
at 300k needs 9×10¹⁰ attention scores per head per layer, and for a 2B model of
this shape the KV cache alone would be 73 GiB. So most layers are not attention:

| Layer kind | Count (32-layer config) | Compute | Cache at 300k |
|---|---|---|---|
| `ssd` — Mamba-2 state space | 22 | **O(L)** | **constant** |
| `swa` — sliding-window attention (4k) | 5 | O(L·w) | O(window) |
| `global` — full GQA attention | 5 | O(L²) | O(L) |

Only the 5 global layers pay the long-context bill, and GQA with 2 KV heads keeps
even those small. Computed by `planner.py` on `base_300k.yaml` (2.06B params) at
300k context, bf16:

| Design | KV cache at 300k |
|---|---|
| **This hybrid (22 SSD / 5 SWA / 5 global, GQA-2)** | **1.47 GiB** |
| All 32 layers global, same GQA-2 | 9.16 GiB (6.2×) |
| All 32 layers global, conventional MHA | 73.24 GiB (49.8×) |

Two independent levers, and it is worth keeping them separate: the layer mix buys
6.2×, and GQA on top buys another 8×.

Everything else is current standard practice: RMSNorm pre-norm, SwiGLU, RoPE with a
large base for extrapolation, QK-norm for long-context logit stability, untied
embeddings at scale, z-loss against late-run divergence.

### Modality budget at 300k tokens

| Modality | Encoding | Fits in 300k |
|---|---|---|
| Code / text | BPE, ~4 chars/token | ~1.2M chars ≈ 30,000 lines |
| Image | 448², patch 14, 2×2 merge → 256 tok | ~1,170 images |
| Video | 1 fps, 2-frame temporal patch → 128 tok/frame | ~2,340 frames ≈ **39 min** |

Images and video share one 3D-patch ViT — an image is a 1-frame clip. Encoder output
is spliced over runs of a reserved `<|vision_pad|>` token, so the LM sees a single
causal sequence regardless of modality.

---

## Layout

```
configs/          tiny_cpu · smoke · small_1gpu · base_300k
src/mmllm/
  config.py       one dataclass tree; drives every scale
  planner.py      FLOPs / wall-clock / memory / KV-cache estimator
  model/
    ssd.py        Mamba-2 SSD: chunked-parallel + reference + single-token decode
    attention.py  GQA, sliding-window and global variants
    vision.py     3D-patch ViT for image and video
    rope.py       1D RoPE (+ NTK scaling) and factorised 3D RoPE
    model.py      top-level LM, vision splicing, cache, budgets
  data/
    hf_ingest.py  stream a HF dataset -> tokenized .bin (memory-bounded)
    tokenizer.py  byte-level BPE + FIM + multimodal control tokens
    prepare.py    quality filter → MinHash dedup → decontaminate → tokenize
    dataset.py    memmap packed loader, deterministic & resumable
  train/
    schedule.py   WSD / cosine, selective weight decay
    train.py      loop with spike guard, atomic ckpt, resume
  model/budget.py analytic params/FLOPs/KV — never instantiates a model
scripts/
  trace_pipeline.py  walks one real batch through all 16 pipeline stages
  sample.py          generate from a trained checkpoint
tests/            31 tests; the correctness argument
```

---

## Quickstart

```bash
pip install -r requirements.txt
```

Validate the architecture (~25 s):

```bash
python -m pytest tests -q
```

Get a real corpus from Hugging Face. This streams — it never holds more than one
batch in RAM, so it ingests a corpus far larger than the machine's memory:

```bash
HF_HOME=E:/hf_cache PYTHONPATH=src python -m mmllm.data.hf_ingest --repo roneneldan/TinyStories --files "data/train-00000-of-00004-*.parquet" --out data_tinystories --vocab 4096
```

Or build one from local files (loads everything into RAM — small corpora only):

```bash
python -m mmllm.data.prepare --input corpus/ --out data --vocab 8192 --val-frac 0.02
```

```bash
PYTHONPATH=src python -m mmllm.train.train --config configs/smoke.yaml
```

Watch one real batch move through every stage — raw text, cleaning, tokenizer,
ids, embeddings, layers, attention, next-token prediction, target comparison,
loss, backward, gradients, optimizer, weight update:

```bash
PYTHONPATH=src python scripts/trace_pipeline.py
```

Price any config against any hardware:

```bash
PYTHONPATH=src python -m mmllm.planner --config configs/base_300k.yaml --gflops 989000 --devices 64
```

---

## What the tests actually pin

These are the point of running anything on a laptop — each one catches a bug that
either produces silent quality loss or costs a rented-GPU run.

| Test | Catches |
|---|---|
| `ssd_chunked` vs `ssd_reference` (float64, 4 shapes) | A wrong scan makes every long-context claim fiction |
| SSD causality | Future tokens leaking backwards through the recurrence |
| SSD prefill → step equivalence | Decode path drifting from training path — invisible in the loss curve |
| Cache generation vs full forward, ×3 layer patterns | The exact bug this repo shipped with and fixed |
| LM causality | Attention mask errors |
| RoPE relative invariance | Broken position encoding; this property is what allows 4k→300k extension |
| Gradient reaches every parameter | Dead branches (found nothing, but it's the cheap check) |
| Vision splice changes logits | An encoder wired in but not actually read |
| KV cache: hybrid ≪ all-attention | The core architectural claim, asserted not assumed |
| SSD state constant in sequence length | O(1) memory claim |
| WSD schedule shape | Silent LR bugs |
| Analytic params == real model, ×6 configs | Cost estimates drifting from the modules they price |
| 2B config priceable without allocating it | Planning requiring the hardware being planned for |
| Single-line spam rejected, real content kept | A quality filter that passes `"click here \| click here \| …"` |

Four real defects were caught this way during the build:

1. **The prefill pass never published its KV cache** — decoding attended only to
   the current token. Caught by `test_cache_generation_matches_full_forward`.
2. **SSD layers had no recurrent-state cache**, so generation re-scanned from a
   zero state every step. Required writing `ssd_step` and a conv-window cache.
3. **Throughput was over-reported ~10×** right after each eval, because the timer
   reset but the log still divided by `log_every`.
4. **`6ND` under-costed long-context training by 4×** — it omits attention, which
   is 11% of FLOPs at 8k but 75% at 300k.
5. **`quality_ok` accepted single-line spam and rejected real code.** The
   line-uniqueness rule only catches repetition *across* lines, so
   `"click here | click here | …"` — one unique line — passed. Found by running
   `trace_pipeline.py`, fixed with a duplicate-5-gram check, and calibrated
   against the real corpus (median 0.012, p90 0.069, max 0.309 vs a 0.35
   threshold: **zero** genuine documents newly rejected, spam at 0.971).

All five are fixed and regression-tested. Numbers 3, 4 and 5 are the dangerous
kind: nothing crashes, you just believe a wrong number or train on worse data.

---

## Choosing a dataset

The first run here trained on this repo's own docs — 245k tokens — and produced
markdown structure but no grammar. That is a *data* ceiling, not a model one.

| Dataset | Size | Why / when |
|---|---|---|
| **`roneneldan/TinyStories`** | ~122M tok/shard | **Default for a <10M-param model.** Synthetic children's stories with a deliberately small vocabulary; the paper's result is that 1–30M-param models trained on it produce *fluent grammatical prose*. Nothing else does that at this scale. |
| `Salesforce/wikitext` (103-raw) | ~103M tok | Classic benchmark — use it to compare perplexity against published numbers. |
| `HuggingFaceFW/fineweb-edu` | 10B+ tok | Real web text. Only worth it above ~100M params, on a GPU. |
| `bigcode/the-stack-smol` | ~1M files | Add code once prose works. |

**Match data volume to model size.** Chinchilla-optimal is ~20 tokens per
parameter, so a 5.5M-param model wants ~110M tokens — which is one TinyStories
shard almost exactly. `python -m mmllm.planner` prints this for any config.

## Training stages

Do not attempt these out of order. Each one de-risks the next.

All figures below are `planner.py` output at 40% MFU on H100s ($2–3/GPU-hr), not
guesses. Budget roughly **2× the compute number** for restarts, failed runs and
data iteration — that overhead is normal, not a sign something went wrong.

| Stage | Config | Scope | GPU-hours | Compute cost |
|---|---|---|---|---|
| 0 — validate | `smoke` / `tiny_cpu` | 6.5M params, code-path proof | — | free (this laptop) |
| 1 — first real LM | `small_1gpu` | 400M × 8B tok, text+code, 4k ctx | ~230 | ~$500–700 |
| 2 — vision align | `small_1gpu` + vision | freeze LM, train encoder + projector | ~150–400 | ~$300–1,200 |
| 3 — pretrain | `base_300k` | 2.06B × 2T tok, 8k ctx | **19,643** | **$39k–59k** |
| 4 — extend ctx | `base_300k`, `rope_scaling: 36` | 8k → 300k, 10B tok | 349 | ~$700–1,050 |
| 5 — post-train | — | SFT → DPO → RLVR | ~1,000–4,000 | $2k–12k |

Stage 3 is 12.8 days on 64× H100. Note what stage 4 costs: extending 8k → 300k is
**~2% of pretraining**, which is exactly why you do it as a separate short phase
rather than pretraining at 300k. At 300k, attention is 75% of all FLOPs; at 8k it
is 11%. Pretraining directly at 300k would burn most of the budget on attention
over positions the model cannot yet use.

**If the budget is not six figures**, the honest path is continued pretraining of
an open base (Qwen3-VL, Llama, Gemma) on your domain data — 100–5,000 GPU-hours, and
it will beat anything you can pretrain from scratch for the same money. This repo is
still the right thing to have built: stages 0–2 are how you learn what stage 3 costs.

---

## Known gaps

- **Vision data loader is a stub.** `InterleavedMultimodalDataset` expects a
  `vision_store` with `.sample(i)` and `__len__`; no image/video decoder is wired up.
- **Single-device only.** No FSDP/tensor/pipeline parallel. Stage 3 needs them.
- **`ssd_chunked` is pure PyTorch.** Correct, but 3–5× slower than the `mamba-ssm`
  CUDA kernel; swap it in on GPU.
- **Image/video are input-only.** Generation would need a VQ tokenizer or diffusion
  head, neither of which is here.
