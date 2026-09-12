"""Convert an instruction dataset into our SFT JSONL format.

Reads from the Hub or from parquet/jsonl already on disk, and handles the three
layouts that cover almost everything: `messages`, `instruction/input/output`, and
the ShareGPT `{system, conversations[{from,value}]}` used by reasoning datasets.

Writes **streaming**. OpenThoughts averages 34,000 characters per example, so
accumulating 40,000 of them costs well over a gigabyte of Python strings — which
is most of this machine's free RAM. Rows go straight to disk instead.

    python -m ft.hf_dataset --repo databricks/databricks-dolly-15k --out data_dolly
    python -m ft.hf_dataset --local ../_tmp_bucket/data --conversations --out data_ot
    python -m ft.hf_dataset --repo databricks/databricks-dolly-15k --out data_mixed \
        --mix data/train.jsonl --mix-ratio 0.45
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator, List, Optional

KNOWN = {
    "databricks/databricks-dolly-15k": ("instruction", "context", "response"),
    "yahma/alpaca-cleaned":            ("instruction", "input", "output"),
    "tatsu-lab/alpaca":                ("instruction", "input", "output"),
}


# ------------------------------------------------------------------ sources
def load_local(root: str, pattern: str = "*.parquet") -> Iterator[dict]:
    """Read parquet/jsonl already on disk, one batch at a time."""
    paths = sorted(Path(root).glob(pattern))
    if not paths:
        raise SystemExit(f"no files matching {pattern} under {root}")
    for p in paths:
        print(f"  reading {p.name}", flush=True)
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq
            for rb in pq.ParquetFile(p).iter_batches(batch_size=100):
                for row in rb.to_pylist():
                    yield row
        else:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)


def load_rows(repo: str, files: Optional[List[str]]) -> Iterator[dict]:
    import fnmatch

    from huggingface_hub import HfApi, hf_hub_download

    names = [s.rfilename for s in HfApi().dataset_info(repo).siblings]
    picked = ([n for n in names if any(fnmatch.fnmatch(n, p) for p in files)] if files
              else [n for n in names if n.endswith((".jsonl", ".json", ".parquet"))
                    and "test" not in n.lower()])
    if not picked:
        raise SystemExit(f"no data files in {repo}; available: {names[:20]}")
    for name in sorted(picked):
        print(f"  fetching {name}", flush=True)
        p = hf_hub_download(repo, name, repo_type="dataset")
        if name.endswith(".parquet"):
            import pyarrow.parquet as pq
            for rb in pq.ParquetFile(p).iter_batches(batch_size=500):
                for row in rb.to_pylist():
                    yield row
        else:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)


# --------------------------------------------------------------- conversion
def to_messages(row: dict, cols) -> Optional[dict]:
    instr_c, ctx_c, out_c = cols
    instr = (row.get(instr_c) or "").strip()
    ctx = (row.get(ctx_c) or "").strip() if ctx_c else ""
    out = (row.get(out_c) or "").strip()
    if not instr or not out:
        return None
    user = f"{instr}\n\n{ctx}" if ctx else instr
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": out}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="HF dataset id")
    ap.add_argument("--local", default=None, help="directory of parquet/jsonl on disk")
    ap.add_argument("--pattern", default="*.parquet")
    ap.add_argument("--conversations", action="store_true",
                    help="source uses {system, conversations[{from,value}]}")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--out", default="data_hf")
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--input", dest="input_col", default=None)
    ap.add_argument("--output", dest="output_col", default=None)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--min-output-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=0,
                    help="skip examples longer than this (0 = keep all)")
    ap.add_argument("--mix", default=None, help="existing JSONL to append")
    ap.add_argument("--mix-ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.repo and not args.local:
        raise SystemExit("pass --repo or --local")

    if args.conversations:
        cols = None
        print(f"[1/3] reading {args.local or args.repo}  layout=conversations")
    elif args.instruction:
        cols = (args.instruction, args.input_col, args.output_col)
        print(f"[1/3] reading {args.local or args.repo}  columns={cols}")
    elif args.repo in KNOWN:
        cols = KNOWN[args.repo]
        print(f"[1/3] reading {args.repo}  columns={cols}")
    else:
        raise SystemExit("unknown layout; pass --conversations or --instruction/--input/--output")

    from .data import dump_jsonl, normalise

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    src = load_local(args.local, args.pattern) if args.local else load_rows(args.repo, args.files)

    n = dict(kept=0, val=0, short=0, long=0, bad=0)
    chars = 0
    # Straight to disk. The training loop shuffles indices every epoch, so a
    # global pre-shuffle here would buy nothing and cost all of RAM.
    with open(out / "train.jsonl", "w", encoding="utf-8") as ftr, \
         open(out / "val.jsonl", "w", encoding="utf-8") as fva:
        for row in src:
            try:
                msgs = normalise(row) if cols is None else None
                m = {"messages": msgs} if cols is None else to_messages(row, cols)
            except ValueError:
                n["bad"] += 1
                continue
            if m is None or len(m["messages"]) < 2:
                n["bad"] += 1
                continue
            size = sum(len(x["content"]) for x in m["messages"])
            if len(m["messages"][-1]["content"]) < args.min_output_chars:
                n["short"] += 1
                continue
            if args.max_chars and size > args.max_chars:
                n["long"] += 1
                continue

            line = dump_jsonl(m)
            if rng.random() < args.val_frac:
                fva.write(line + "\n"); n["val"] += 1
            else:
                ftr.write(line + "\n"); n["kept"] += 1
            chars += size

            total = n["kept"] + n["val"]
            if total % 5000 == 0:
                print(f"      {total:,} written · {chars/1e6:,.0f} M chars", flush=True)
            if args.max_examples and total >= args.max_examples:
                break

    print(f"      kept {n['kept']:,} train / {n['val']:,} val "
          f"(skipped short-{n['short']:,} long-{n['long']:,} bad-{n['bad']:,})")
    avg = chars / max(n["kept"] + n["val"], 1)
    print(f"      mean {avg:,.0f} chars/example  ~= {avg/3.6:,.0f} tokens")

    if args.mix:
        extra = [l for l in open(args.mix, encoding="utf-8") if l.strip()]
        take = min(len(extra), int(n["kept"] * args.mix_ratio))
        rng.shuffle(extra)
        with open(out / "train.jsonl", "a", encoding="utf-8") as ftr:
            for l in extra[:take]:
                ftr.write(l if l.endswith("\n") else l + "\n")
        print(f"[2/3] appended {take:,} examples from {args.mix}")
    else:
        print("[2/3] no blend requested")

    print(f"[3/3] wrote {out}/train.jsonl and {out}/val.jsonl")


if __name__ == "__main__":
    main()
