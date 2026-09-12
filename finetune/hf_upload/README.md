---
license: apache-2.0
language:
  - en
task_categories:
  - text-generation
tags:
  - code
  - instruction-tuning
  - sft
  - lora
size_categories:
  - 1K<n<10K
configs:
  - config_name: balanced
    default: true
    data_files:
      - split: train
        path: balanced/train.jsonl
      - split: validation
        path: balanced/validation.jsonl
  - config_name: full
    data_files:
      - split: train
        path: full/train.jsonl
      - split: validation
        path: full/validation.jsonl
---

# Workspace SFT Dataset

Supervised fine-tuning pairs extracted from a personal multi-project engineering
workspace: 21 projects covering RAG systems, multi-agent orchestration,
an inference server, a SAP ABAP approval engine, a React dashboard, and an ERP
schema.

Each row is a chat exchange in the format most SFT trainers expect:

```json
{"messages": [{"role": "user", "content": "..."},
             {"role": "assistant", "content": "..."}],
 "project": "saas_agent", "source_kind": "python"}
```

`project` and `source_kind` are provenance, not training signal - trainers that
read only `messages` ignore them, and they let you filter or rebalance without
rebuilding.

## Configs

| Config | Train | Validation | Notes |
|---|---|---|---|
| `balanced` | 2,706 | 142 | capped at 400/project |
| `full` | 5,723 | 301 | every extracted row |

`balanced` caps each project at 400 rows so the two largest cannot dominate; in
`full` they are 34% of the data. `balanced` is the default.

```python
from datasets import load_dataset
ds = load_dataset("<your-hf-username>/Venki_data_set", "balanced")
```

## How it was built

Five extractors over the workspace tree:

| Source | Pair shape |
|---|---|
| Markdown | each `##`/`###` section -> a question about that section |
| Python | documented function/class -> docstring, and signature+docstring -> implementation |
| ABAP | class ABAP-Doc -> description; `METHOD`...`ENDMETHOD` -> implementation |
| JS/TS/JSX | declaration -> implementation, plus JSDoc where present |
| SQL | each `CREATE TABLE`/`VIEW` -> its DDL |

Deduplication is exact (blake2b on the answer) then near-duplicate (banded
MinHash-LSH, 64 permutations / 16 bands). Vendored code, build output and a
duplicate git worktree of the workspace are excluded, so the rows are
first-party work.

## Known limitations

Read these before training on it.

- **Template-generated questions.** Five phrasings across all extractors. This
  teaches facts and output format but gives almost no instruction diversity, so a
  model trained only on this stays brittle to rephrasing.
- **The Markdown half asks for recall, not reasoning.** A question like
  *"In X Readme, describe Workflow State"* identifies a section by title and
  expects its content reproduced. That is memorisation of ~1.5k sections. In a
  135M-parameter reference fine-tune the code pairs worked well while these
  produced correctly-formatted but fabricated content. The code, ABAP, JS and SQL
  pairs carry their answer's context in the question and behave much better.
- **Long rows are dropped by the trainer, not here.** At `max_len` 512 about 5%
  of rows overflow; raising the limit recovers them.

## Reference fine-tune

LoRA (r=16, alpha=32, 7 projections; 4.88M trainable of 139M) on
SmolLM2-135M-Instruct, 1 epoch over the `balanced` config, CPU:

| Metric | Base | Tuned |
|---|---|---|
| Validation loss | 3.1047 | 2.2719 |
| Held-out loss | 2.7133 | 2.0401 |
| Perplexity | 15.1 | 7.7 |

Generate with a repetition penalty (~1.15). Greedy decoding with none sends a
model this small into repetition loops that are an artifact of decoding rather
than of the fine-tune.

## Redaction

A GCP project id, its Workload-Identity service account and a GKE LoadBalancer IP were replaced with documentation-range placeholders (49 occurrences). No credentials were present: the corpus was scanned for AWS, Anthropic, OpenAI, GitHub, Google and Slack key formats and private-key headers, with zero matches. Strings that look like keys (`sk-ant-...`, `YOUR_DD_API_KEY`, `sk_test_123`) are placeholders in documentation and test fixtures.

## Licence

Apache-2.0, covering the extracted text. Rows derive from first-party project
code and documentation.
