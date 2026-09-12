"""Build an SFT dataset from a local codebase and its docs.

Produces JSONL of {"messages":[...]} from two extractors:
  * Markdown  — each `##` section becomes a question/answer pair.
  * Python    — each documented function/class becomes "what does X do" plus a
                signature+docstring -> implementation pair.

Both are TEMPLATE-generated, and that limitation is real: template data teaches
format and facts but has almost no instruction diversity, so a model trained only
on it becomes brittle to rephrasing. Use it to bootstrap and to smoke-test the
pipeline; for a production run, mix in genuinely varied instructions.

    python -m ft.build_dataset --input ../ --out data --val-frac 0.05
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from pathlib import Path

from .data import dump_jsonl
from typing import Iterator, List

import numpy as np

SKIP_DIRS = {".git", "__pycache__", "node_modules", "site-packages", "runs",
             "corpus", ".venv", "venv", "data"}

Q_TEMPLATES = [
    "What does the \"{h}\" section of {doc} cover?",
    "Explain {h} in {doc}.",
    "In {doc}, describe {h}.",
]


class MinHashDedup:
    """Banded MinHash-LSH near-duplicate filter.

    Exact-hash dedup only catches byte-identical answers. A repo full of
    `*_SUMMARY.md` / `*_GUIDE.md` variants produces answers that differ by a few
    words and are otherwise the same — measured at 3.1% of this dataset. Training
    on those just teaches the model to memorise them.

    Self-contained on purpose: this package does not depend on the pretraining repo.
    """

    def __init__(self, perms: int = 64, bands: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 2 ** 61 - 1, perms, dtype=np.int64)
        self.b = rng.integers(0, 2 ** 61 - 1, perms, dtype=np.int64)
        self.perms, self.bands, self.rows = perms, bands, perms // bands
        self.buckets = [dict() for _ in range(bands)]

    @staticmethod
    def _shingle_hash(text: str) -> int:
        """Stable across processes, unlike the builtin `hash()`.

        `hash()` on a str is salted by PYTHONHASHSEED, so shingle values — and
        therefore every dedup decision — changed run to run: the same corpus
        dropped between 189 and 211 near-duplicates across builds, and the
        near-dup test passed or failed depending on the seed the interpreter
        happened to start with. `--seed` fixes the permutations, not this.
        """
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % (2 ** 61 - 1)

    def _shingles(self, text: str, k: int = 9) -> np.ndarray:
        w = re.findall(r"\w+", text.lower())
        sh = {self._shingle_hash(" ".join(w[i:i + k])) for i in range(max(len(w) - k + 1, 1))}
        return np.fromiter(sh, dtype=np.int64)

    def is_duplicate(self, text: str) -> bool:
        sh = self._shingles(text)
        if sh.size == 0:
            return False
        sig = ((sh[None, :] * self.a[:, None] + self.b[:, None]) % (2 ** 61 - 1)).min(axis=1)
        hit = False
        for i in range(self.bands):
            key = hashlib.blake2b(
                sig[i * self.rows:(i + 1) * self.rows].tobytes(), digest_size=8
            ).digest()
            if key in self.buckets[i]:
                hit = True
            self.buckets[i][key] = True
        return hit


def iter_files(root: Path, suffix: str) -> Iterator[Path]:
    for p in root.rglob(f"*{suffix}"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file():
            yield p


def markdown_pairs(text: str, doc: str, min_chars: int, max_chars: int, rng) -> Iterator[dict]:
    """Section pairs from one document's text. `doc` is its display title."""
    # Split on ## / ### headings, keeping the heading with its body.
    parts = re.split(r"^(#{2,3})\s+(.+)$", text, flags=re.M)
    for i in range(1, len(parts) - 2, 3):
        heading, body = parts[i + 1].strip(), parts[i + 2].strip()
        if not (min_chars <= len(body) <= max_chars):
            continue
        if body.count("|") > 40:            # giant tables answer nothing useful
            continue
        yield {"messages": [
            {"role": "user", "content": rng.choice(Q_TEMPLATES).format(h=heading, doc=doc)},
            {"role": "assistant", "content": body},
        ]}


def doc_title(p: Path) -> str:
    return p.stem.replace("_", " ").title()


def from_markdown(root: Path, min_chars: int, max_chars: int, rng) -> Iterator[dict]:
    for p in iter_files(root, ".md"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        yield from markdown_pairs(text, doc_title(p), min_chars, max_chars, rng)


def python_pairs(src: str, filename: str, min_chars: int, max_chars: int) -> Iterator[dict]:
    """Docstring and implementation pairs from one module's source."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return
    lines = src.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc or len(doc) < 40:
            continue
        kind = "class" if isinstance(node, ast.ClassDef) else "function"

        yield {"messages": [
            {"role": "user",
             "content": f"What does the `{node.name}` {kind} in `{filename}` do?"},
            {"role": "assistant", "content": doc.strip()},
        ]}

        # Signature + docstring -> implementation. This is the pair that
        # actually teaches the codebase's idioms rather than just its prose.
        seg = "\n".join(lines[node.lineno - 1:node.end_lineno])
        if min_chars <= len(seg) <= max_chars:
            yield {"messages": [
                {"role": "user",
                 "content": f"Implement `{node.name}` in `{filename}`.\n\n"
                            f"Docstring:\n{doc.strip()}"},
                {"role": "assistant", "content": f"```python\n{seg}\n```"},
            ]}


def from_python(root: Path, min_chars: int, max_chars: int) -> Iterator[dict]:
    for p in iter_files(root, ".py"):
        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        yield from python_pairs(src, p.name, min_chars, max_chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--max-examples", type=int, default=0, help="0 = no cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-near-dedup", action="store_true",
                    help="keep near-duplicate answers (exact dedup still runs)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.input)
    rows: List[dict] = []
    rows += list(from_markdown(root, args.min_chars, args.max_chars, rng))
    n_md = len(rows)
    rows += list(from_python(root, args.min_chars, args.max_chars))
    print(f"extracted {n_md} markdown, {len(rows)-n_md} python -> {len(rows)} raw")

    # Exact-dedup on the assistant answer: docs repeat themselves a lot, and
    # duplicated targets are the fastest route to a memorising model.
    seen, uniq, n_exact = set(), [], 0
    near = None if args.no_near_dedup else MinHashDedup(seed=args.seed)
    n_near = 0
    for r in rows:
        answer = r["messages"][-1]["content"]
        key = hashlib.blake2b(answer.encode(), digest_size=16).digest()
        if key in seen:
            n_exact += 1
            continue
        seen.add(key)
        if near is not None and near.is_duplicate(answer):
            n_near += 1
            continue
        uniq.append(r)
    print(f"after dedup: {len(uniq)}  (exact -{n_exact}, near -{n_near})")

    rng.shuffle(uniq)
    if args.max_examples:
        uniq = uniq[:args.max_examples]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_val = max(int(len(uniq) * args.val_frac), 1)
    for name, part in (("val", uniq[:n_val]), ("train", uniq[n_val:])):
        f = out / f"{name}.jsonl"
        f.write_text("\n".join(dump_jsonl(r) for r in part),
                     encoding="utf-8")
        print(f"  {name}: {len(part)} examples -> {f}")


if __name__ == "__main__":
    main()
