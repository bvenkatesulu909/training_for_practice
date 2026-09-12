# training_for_practice

Two complete, tested LLM training systems, built end to end on a **2012 laptop** —
an Intel i3-3110M, 8 GB RAM, no GPU. The constraint is the point: it forces every
number to be measured rather than assumed.

| | |
|---|---|
| [`serve/`](serve) | The **serving layer** — safety guard, retrieval, abstention, release gate. |
| [`mmllm/`](mmllm) | A multimodal LLM **pretrained from scratch** — architecture, tokenizer, streaming data pipeline, training loop. |
| [`finetune/`](finetune) | **LoRA fine-tuning** of an existing model, with LoRA implemented directly rather than via `peft`. |

**Trained weights:** [huggingface.co/Venkatesulu/training_for_practice](https://huggingface.co/Venkatesulu/training_for_practice)
— this repo is code only; the 538 MB model lives on the Hub.

---

## What's actually interesting here

Not the model. It's 135M parameters trained for 60 steps and it is not good — it
learned *format*, not knowledge, and on some prompts it degenerates into a
repetition loop. That's documented honestly on the model card.

The interesting part is the **131 tests**, because every one of them pins a failure
mode that produces a *plausible-looking but wrong* run rather than a crash. Those
are the only bugs worth spending laptop compute on.

| Test | Failure it prevents |
|---|---|
| Prompt tokens are masked | Without it, most of the gradient teaches the model to generate *user questions* — and the loss curve looks perfectly healthy while it happens |
| Drop-rate guard | A `max_len` shorter than your data yields an **empty** training set, silently. On a reasoning dataset with 7,700-token median, `max_len=1024` keeps zero examples |
| JSONL line-breaker escaping | `json.dumps(ensure_ascii=False)` leaves U+0085/U+2028/U+2029 raw; `splitlines()`, jq and pandas all treat them as newlines. This shredded 11 records before it was caught |
| Merge is numerically equivalent (float64) | Shipping a different model than the one you evaluated |
| LoRA adapter dtype matches base | Harmless on CPU; fatal on every bf16 GPU run |
| Cached generation == full forward | Decode path drifting from the training path |
| SSD chunked scan == reference scan | The entire long-context argument rests on this |
| Analytic params == real model | Cost estimates drifting from the code they price |
| Knowledge base is self-consistent | Wikidata lists two capitals for Karnataka. Shipping both as *X is the capital* lets retrieval pick the answer by ranking luck — and the suite still reads 100%, because it asks once |
| Long documents don't outrank short ones | Recall has no length penalty, so a 62-token railways summary beat *New Delhi is the capital of India* on the question *India's national capital* |

Fourteen real bugs were found this way during development. **Every one was silent.**

---

## The result worth reading

The same 20-case frozen suite, run against the raw model and against a serving
pipeline wrapped around it. **Identical weights in both columns.**

| Category | Raw model | With pipeline |
|---|---|---|
| injection | 33.3% | **100%** |
| unanswerable (must abstain) | **0.0%** | **100%** |
| geography | 50% | **100%** |
| arithmetic | 75% | **100%** |
| latency P95 | 35.31 s | **0.00 s** |
| **overall** | **55.0%** | **100.0%** |
| **release gate** | **BLOCK** | **PROMOTE** |

No training involved. Each failure was fixed where it lives: injection by a check
*before* the model, abstention by a retrieval threshold, arithmetic by a
calculator, latency by not calling a model for most requests.

`serve/SPEC.md` records the targets and the RAG-vs-fine-tune decision, written
before any of it was measured.

---

## Why the facts are not in the weights

This was measured, not assumed. Fine-tuning 22 facts drove validation loss from
**1.6301 to 0.0680** — a 24x improvement — and the model still answered *what is
Karnataka's capital* wrong, because Karnataka was not one of the 22. Facts do not
generalise out of the weights they were trained into.

The same facts in a retrieval store answer correctly, cite a source, and are
corrected by editing one row. So `serve/build_india_kb.py` builds one from live
Wikidata and Wikipedia — 370 documents covering all 28 states and union
territories, 261 districts, 49 major cities, and 32 article summaries:

```bash
PYTHONPATH=serve/src python -m serve.build_india_kb --out serve/india_kb.json
PYTHONPATH=serve/src python -m serve.validate_kb serve/india_kb.json
```

Re-running it refreshes the store. No training, no GPU, no evaluation to redo.


The corpus is also published on its own as a dataset:
[Venki_data_set_Ananthapuram](https://huggingface.co/datasets/Venkatesulu/Venki_data_set_Ananthapuram) — 370 rows, licence recorded per row
(338 CC0 from Wikidata, 32 CC BY-SA 4.0 from Wikipedia).

**What the build had to defend against.** Every one of these produced confidently
wrong output that no existing test was failing on:

| Source defect | Effect if shipped |
|---|---|
| Dissolved entities carry stale capitals | *Hyderabad is the capital of Andhra Pradesh* — true until 2014, false now |
| City query required only *in India, has a population* | 23 of 40 "cities" were states, regions or a demonym: *Karnataka is a major city in India, India* |
| Karnataka has two capitals at equal rank, no qualifier | Two contradicting documents; retrieval answers by ranking luck |
| Two population statements for one city | The store disagrees with itself about Vadodara |
| Capital/headquarters URLs built from a LABEL | `Special:EntityPage/Andhra Pradesh` - a citation field pointing at a dead URL with a raw space in it |

`validate_kb.py` fails the build on all four, and CI runs it against both the KB
file and the merged store that queries actually hit.

## Quickstart

### Fine-tune an existing model

```bash
pip install -r finetune/requirements.txt
PYTHONPATH=finetune/src python -m ft.build_dataset --input /path/to/repo --out data
PYTHONPATH=finetune/src python -m ft.train --config finetune/configs/smol135m_cpu.yaml
PYTHONPATH=finetune/src python -m ft.evaluate --adapter runs/smol135m/adapter_best.pt
PYTHONPATH=finetune/src python -m ft.merge --adapter runs/smol135m/adapter_best.pt --out export/
```

Blend in human-written instructions — template-generated data has no instruction
diversity, and this is usually what makes a fine-tune generalise:

```bash
PYTHONPATH=finetune/src python -m ft.hf_dataset --repo databricks/databricks-dolly-15k --out data_mixed --mix data/train.jsonl --mix-ratio 0.45
```

Price a run before committing to it — downloads only `config.json`, never weights,
and reports peak GPU memory *including activations*:

```bash
PYTHONPATH=finetune/src python -m ft.planner --config finetune/configs/qwen7b_gpu.yaml --gflops 989000 --price 1.8
```

### Pretrain from scratch

```bash
pip install -r mmllm/requirements.txt
PYTHONPATH=mmllm/src python -m mmllm.data.hf_ingest --repo roneneldan/TinyStories --files "data/train-00000-of-00004-*.parquet" --out data_tinystories --vocab 4096
PYTHONPATH=mmllm/src python -m mmllm.train.train --config mmllm/configs/tinystories_cpu.yaml
```

Watch one real batch move through all sixteen pipeline stages — raw text, cleaning,
tokenizer, ids, embeddings, layers, attention, next-token prediction, target
comparison, loss, backward, gradients, optimizer, weight update:

```bash
PYTHONPATH=mmllm/src python mmllm/scripts/trace_pipeline.py
```

---

## Architecture note

`mmllm` targets a 300,000-token context. At that length the KV cache, not compute,
is the binding constraint — so most layers are linear-time state-space (Mamba-2 SSD)
rather than attention:

| Design | KV cache at 300k ctx |
|---|---|
| **This hybrid** (22 SSD / 5 sliding-window / 5 global, GQA-2) | **1.47 GiB** |
| All layers global attention, same GQA | 9.16 GiB (6.2×) |
| All layers conventional MHA | 73.24 GiB (49.8×) |

---

## Hardware reality

Measured, not estimated:

| | |
|---|---|
| Peak matmul throughput | 24 GFLOP/s (i3-3110M, 2 cores, CPU only) |
| Achieved in training | ~11.7 GFLOP/s (48% MFU) |
| From-scratch pretraining of a useful model | **~10⁷× beyond this machine** |

That gap is why this repo ships a fine-tuned model rather than a pretrained one, and
why `planner.py` exists. A 7B LoRA fine-tune on a rented A100 costs single-digit
dollars; pretraining a 2B model costs ~19,600 GPU-hours. For domain adaptation,
fine-tuning wins by roughly five orders of magnitude.

---

## Tests

```bash
PYTHONPATH=mmllm/src    python -m pytest mmllm/tests -q     # 34 tests
PYTHONPATH=finetune/src python -m pytest finetune/tests -q  # 38 tests
PYTHONPATH=serve/src    python -m pytest serve/tests -q     # 59 tests
```

CI runs all three on every push, plus four checks that exist because each of
these went wrong by hand during development — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## License

Apache-2.0.
