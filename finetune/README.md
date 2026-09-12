# ft — LoRA fine-tuning of an existing LLM

Adapt a pretrained model to your domain, on hardware you already have. Pure
PyTorch + transformers; LoRA implemented directly (~60 lines) rather than via
`peft`, so it is testable and does not break when adapter libraries lag a
transformers release.

This is the track the arithmetic pointed at. Pretraining the 2B multimodal model
in [`../mmllm`](../mmllm) costs ~19,600 GPU-hours; a LoRA fine-tune of an open
base that already speaks English and code costs **1–50 GPU-hours** and, for
domain adaptation, wins.

---

## What it does

| Stage | Module | Notes |
|---|---|---|
| Build a dataset from a codebase | `ft.build_dataset` | Markdown sections + Python docstrings → JSONL, exact **and** MinHash near-dedup |
| Tokenise + mask prompts | `ft.data` | Loss on assistant tokens only, derived from the tokenizer's own template |
| Inject adapters | `ft.lora` | Freeze base, replace target `nn.Linear` with `LoRALinear` |
| Train | `ft.train` | AdamW on adapters only, cosine + warmup, resume, per-eval baseline delta |
| Compare | `ft.evaluate` | Held-out loss **and** base-vs-tuned generations side by side |
| Export | `ft.merge` | Fold adapters into base weights → a plain HF model directory |
| Price a run | `ft.planner` | Time, cost and memory for any base/hardware, from `config.json` alone |
| Pull an instruction set | `ft.hf_dataset` | Convert a HF dataset to our JSONL, optionally blended with your own |

---

## Why LoRA, concretely

```
W_effective = W_frozen + (alpha/r) * B @ A          A: (r, in)   B: (out, r)
```

- **B starts at zero**, so an untrained adapter is bit-identical to the base model.
  That is a free correctness check: if your first eval differs from the base
  model's, something is wrong before you have burned any compute.
- **~1% of parameters train.** For SmolLM2-135M at r=16 that is ~1.5M of 135M.
  Optimizer state shrinks with it — AdamW on a full fine-tune needs 8 bytes per
  parameter, which is what actually puts 7B models out of reach on small GPUs.
- **Adapters are megabytes.** You can keep one per domain and swap them against a
  single base model, instead of storing a full copy of the model per task.

---

## Quickstart

```bash
pip install torch transformers pyyaml pytest
```

Validate the stack first — it takes ~10 s and catches the bugs that matter:

```bash
PYTHONPATH=src python -m pytest tests -q
```

Build a dataset from a codebase:

```bash
PYTHONPATH=src python -m ft.build_dataset --input ../ --out data --val-frac 0.06
```

Blend in human-written instructions — this fixes the diversity gap that
template-generated data has, and is usually what makes a fine-tune generalise:

```bash
PYTHONPATH=src python -m ft.hf_dataset --repo databricks/databricks-dolly-15k --out data_mixed --mix data/train.jsonl --mix-ratio 0.45
```

Train:

```bash
PYTHONPATH=src python -m ft.train --config configs/smol135m_cpu.yaml
```

Compare base vs tuned on held-out prompts:

```bash
PYTHONPATH=src python -m ft.evaluate --adapter runs/smol135m/adapter_best.pt --n 3
```

Export a standalone model that vLLM or Ollama can serve:

```bash
PYTHONPATH=src python -m ft.merge --adapter runs/smol135m/adapter_best.pt --out export/
```

Price a run before committing to it — this downloads only `config.json`, never weights:

```bash
PYTHONPATH=src python -m ft.planner --config configs/qwen7b_gpu.yaml --gflops 312000 --price 1.8
```

---

## What the tests pin

Every one of these guards a failure mode that yields a *plausible-looking but
wrong* run, not a crash. Those are the only ones worth spending laptop time on.

| Test | Failure it prevents |
|---|---|
| Prompt tokens are masked | **The** SFT bug: without masking, most of the gradient teaches the model to generate *user questions*, and the loss curve looks perfectly healthy while it happens |
| Non-prefix template rejected | Silently mislabelled data when a tokenizer's template is not prefix-consistent |
| LoRA is identity at init | Adapter wired in backwards; also gives you a free baseline |
| Merge is numerically equivalent (float64) | Shipping a different model than the one you evaluated |
| Only LoRA params trainable | Accidentally full fine-tuning, or accidentally training nothing |
| Adapter dtype matches base | Injecting fp32 adapters into a bf16 model — breaks every GPU run |
| Gradients: B flows at init, A follows | Documents real LoRA maths so a zero grad is not mistaken for a dead layer |
| Padding never supervised | Training the model to predict pad tokens |
| Overflow dropped, not truncated | Truncating mid-answer teaches the model to stop mid-sentence |
| Planner adapter count == real injection | Cost/memory estimates drifting from the code they price |
| Near-duplicate answers removed | Memorisation: exact-hash dedup missed 3.1% of this dataset |
| Config round-trips through `asdict` | A checkpoint that loads but fails later, at point of use |
| `BatchEncoding` is not a `dict` | It extends `UserDict`; an isinstance check alone passes the wrapper downstream |

Three real bugs were caught this way during the build:

1. **LoRA adapters were created in the default dtype**, not the base layer's. Fine
   on this CPU box, fatal on any bf16 GPU run.
2. **`apply_chat_template(tokenize=True)` returns a `BatchEncoding` in transformers
   5.x**, not a list of ids — so the prefix check was comparing dict *keys*. The
   guard refused to emit labels rather than training on garbage, which is exactly
   what it exists for.
3. **The planner's adapter-size estimate was a guess, and wrong by 12×** — it
   claimed 59.1M trainable params where injection actually creates 4,884,480, and
   reported an impossible 212 GiB of optimizer state for a 7B LoRA. Replaced with
   exact `r*(in+out)`-per-layer arithmetic read from the model's `config.json`, and
   pinned against a real injection.
4. **`load_adapted` rebuilt only the `lora` sub-config** from a checkpoint, leaving
   `data` and `train` as raw dicts. Training and saving both succeeded; the failure
   only appeared when `ft.evaluate` touched `cfg.data.max_len` — after the run was
   already finished. Fixed with `Config.from_dict` and an asdict round-trip test.
5. A merge-count assertion that was simply wrong about the test fixture.

Bugs 1–3 are the dangerous kind: nothing crashes, you just get a wrong number or a
wrongly-trained model and no signal that anything happened.

---

## Matching parameters to the dataset

Parameters are not a property of the model alone — they are a function of the
data. Measured on three real datasets:

| | repo docs | dolly-15k | **OpenThoughts** |
|---|---|---|---|
| median example | ~180 tok | ~250 tok | **7,746 tok** |
| `max_len` needed | 512 | 1024 | **16384** |
| survival at `max_len` 1024 | 96% | 98% | **0%** |
| base model that can learn it | 135M | 135M–1.5B | **7B+** |
| LoRA rank | 16 | 16 | **64** |
| peak GPU memory | <1 GB | <1 GB | **21.8 GiB** |

The middle row is the one that bites. At `max_len: 1024` OpenThoughts yields an
**empty** training set — no error, no loss curve, nothing. `ft.train` now refuses
to start when more than half the examples are dropped, and warns above 10%.

Reasoning data also needs a base model that can already reason: a 135M model
trained on OpenThoughts learns to emit `<|begin_of_thought|>` and then produce
noise. The R1-Distill series used Qwen2.5 at 1.5B and up for exactly this data.

## Choosing a base model

| Base | Params | Fits here? | Where it belongs |
|---|---|---|---|
| SmolLM2-135M-Instruct | 135M | yes | validating the pipeline |
| Qwen2.5-0.5B-Instruct | 0.5B | overnight | small real domain model |
| Qwen2.5-7B-Instruct | 7B | no | 1× A100, a few hours, ~$20–60 |

Rules of thumb that hold up:

- **LoRA lr is 10–50× a full fine-tune's.** 1e-4 to 2e-4 is the usual band; 2e-5
  will look like nothing is happening.
- **`alpha = 2r`** is a fine default. Raising `r` past 64 rarely helps for style
  or domain adaptation; it matters more for teaching genuinely new capability.
- **1–3 epochs.** LoRA overfits small SFT sets fast. Watch held-out loss, not train.
- **Target all seven projections**, not just `q_proj`/`v_proj`. The MLP projections
  (`gate/up/down`) carry most of a transformer's parameters and most of its
  factual knowledge.
- **`train_embeddings: true` only for genuinely new vocabulary.** It is most of the
  trainable budget and usually unnecessary.

---

## Known gaps

- **No 4-bit quantisation (QLoRA).** A 7B base in bf16 needs ~15 GB just for frozen
  weights. Fitting 7B on a 24 GB card needs `bitsandbytes` 4-bit loading, which is
  not implemented here.
- **No DPO/preference stage.** This is SFT only. Preference tuning is the step after.
- **Template-generated dataset.** `build_dataset` produces format-correct pairs with
  almost no instruction diversity, so a model trained only on it is brittle to
  rephrasing. Bootstrap with it; mix in varied instructions for anything real.
  Dedup is handled (exact + MinHash near-dup, 425 + 183 removed on this repo);
  *diversity* is the part you still have to supply — `ft.hf_dataset` is how.
  Blending `databricks/databricks-dolly-15k` at ratio 0.45 gives 17,587 examples
  at roughly 2:1 human-written to repo-domain, which is a sane starting mix.
- **Single device.** No FSDP/DeepSpeed, so multi-GPU is out of scope.
