"""Assemble one upload-ready Hugging Face dataset folder from the built splits.

Everything a dataset repo needs lands in a single directory: both variants as
named configs, a dataset card with the YAML frontmatter the Hub parses, and the
build manifests. Nothing here rebuilds data - it packages what
`ft.build_workspace_dataset` already produced.

Redaction is on by default and is the reason this is a script rather than a copy
command. The corpus carries a live GCP project id, its Workload-Identity service
account, and a GKE LoadBalancer IP. None is a credential and none teaches the
model anything, but uploading is public and hard to walk back, so they are
replaced with documentation-range placeholders here while the local originals
stay untouched. `--no-redact` keeps them verbatim.

    python scripts/package_for_hf.py --out hf_upload
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

# Real identifiers -> placeholders. Ordered: the project id appears inside the
# service-account address, so the address must be rewritten first.
REDACTIONS = [
    (re.compile(r"github-actions@venkatesh-68cb677e\.iam\.gserviceaccount\.com"),
     "github-actions@example-project.iam.gserviceaccount.com"),
    (re.compile(r"venkatesh-68cb677e"), "example-project"),
    (re.compile(r"\b34\.135\.216\.73\b"), "203.0.113.10"),      # TEST-NET-3, RFC 5737
]

VARIANTS = {
    "balanced": "data_workspace_balanced",
    "full": "data_workspace",
}

CARD = """---
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
{configs}---

# Workspace SFT Dataset

Supervised fine-tuning pairs extracted from a personal multi-project engineering
workspace: {n_projects} projects covering RAG systems, multi-agent orchestration,
an inference server, a SAP ABAP approval engine, a React dashboard, and an ERP
schema.

Each row is a chat exchange in the format most SFT trainers expect:

```json
{{"messages": [{{"role": "user", "content": "..."}},
             {{"role": "assistant", "content": "..."}}],
 "project": "saas_agent", "source_kind": "python"}}
```

`project` and `source_kind` are provenance, not training signal - trainers that
read only `messages` ignore them, and they let you filter or rebalance without
rebuilding.

## Configs

| Config | Train | Validation | Notes |
|---|---|---|---|
{table}

`balanced` caps each project at 400 rows so the two largest cannot dominate; in
`full` they are 34% of the data. `balanced` is the default.

```python
from datasets import load_dataset
ds = load_dataset("{repo}", "balanced")
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

{redaction_note}

## Licence

Apache-2.0, covering the extracted text. Rows derive from first-party project
code and documentation.
"""


def redact(text: str, counts: Counter) -> str:
    for pattern, replacement in REDACTIONS:
        text, n = pattern.subn(replacement, text)
        if n:
            counts[pattern.pattern.replace("\\", "")[:40]] += n
    return text


def copy_split(src: Path, dst: Path, do_redact: bool, counts: Counter) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            fout.write((redact(line, counts) if do_redact else line) + "\n")
            rows += 1
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="hf_upload")
    ap.add_argument("--repo", default="your-username/workspace-sft",
                    help="repo id used in the card's load_dataset example")
    ap.add_argument("--no-redact", action="store_true",
                    help="keep the real GCP project id, service account and IP")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    do_redact = not args.no_redact
    counts: Counter = Counter()
    stats = {}

    for config, folder in VARIANTS.items():
        src = Path(folder)
        if not src.exists():
            raise SystemExit(f"missing {src} - build it before packaging")
        n_train = copy_split(src / "train.jsonl", out / config / "train.jsonl", do_redact, counts)
        n_val = copy_split(src / "val.jsonl", out / config / "validation.jsonl", do_redact, counts)
        manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        (out / config / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        stats[config] = {"train": n_train, "validation": n_val, "manifest": manifest}
        print(f"{config:9s} train {n_train:5d}  validation {n_val:4d}")

    configs_yaml = ""
    for i, config in enumerate(VARIANTS):
        configs_yaml += f"  - config_name: {config}\n"
        if i == 0:
            configs_yaml += "    default: true\n"
        configs_yaml += "    data_files:\n"
        configs_yaml += f"      - split: train\n        path: {config}/train.jsonl\n"
        configs_yaml += f"      - split: validation\n        path: {config}/validation.jsonl\n"

    table = "\n".join(
        f"| `{c}` | {s['train']:,} | {s['validation']:,} | "
        f"{'capped at 400/project' if c == 'balanced' else 'every extracted row'} |"
        for c, s in stats.items())

    if do_redact and counts:
        note = ("A GCP project id, its Workload-Identity service account and a GKE "
                "LoadBalancer IP were replaced with documentation-range placeholders "
                f"({sum(counts.values())} occurrences). No credentials were present: the "
                "corpus was scanned for AWS, Anthropic, OpenAI, GitHub, Google and Slack "
                "key formats and private-key headers, with zero matches. Strings that look "
                "like keys (`sk-ant-...`, `YOUR_DD_API_KEY`, `sk_test_123`) are placeholders "
                "in documentation and test fixtures.")
    else:
        note = "None applied - this copy is verbatim."

    n_projects = len(stats["full"]["manifest"]["by_project"])
    (out / "README.md").write_text(
        CARD.format(configs=configs_yaml, table=table, repo=args.repo,
                    redaction_note=note, n_projects=n_projects),
        encoding="utf-8")

    print(f"\nredactions applied: {sum(counts.values())}")
    for k, n in counts.most_common():
        print(f"  {n:4d}  {k}")
    print(f"\nfolder ready: {out.resolve()}")


if __name__ == "__main__":
    main()
