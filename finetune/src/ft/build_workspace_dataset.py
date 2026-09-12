"""Build ONE SFT dataset spanning every project in the workspace.

`ft.build_dataset` reads Markdown and Python only. Run at the workspace root it
therefore silently drops whole projects: `sap-abap-pr-approval` is 14 ABAP
classes, `Multi-Orchestration-agents/pm-dashboard` is a JS/React app, and both
contributed zero rows. This module reuses that module's per-file extractors and
adds ABAP, JS/TS and SQL, so "all my projects" means all of them.

Two things it adds beyond file types:

  * **Provenance.** Every row carries `"project"` (the top-level directory it
    came from) and `"source_kind"`. One dataset, still auditable: you can count,
    filter or rebalance per project afterwards, and `--max-per-project` stops the
    biggest repo from drowning the rest. `ft.data.normalise` reads only
    `messages`, so the extra keys are inert during training.
  * **Vendored exclusion.** `abapGit/` is 756 ABAP files of someone else's
    open-source client, `dist/` is minified bundles, and `.claude/worktrees/` is
    a second checkout of this very workspace. None of it is your work and all of
    it would distort the mix. Excluded by default; `--include-vendored` to
    override.

Provenance is taken from the path at extraction time rather than by matching the
file name quoted in the question. Names collide across projects - a dozen of
these have a `README.md` - so name matching mis-attributes silently, which it did
until this was rewritten.

    python -m ft.build_workspace_dataset --input .. --out data_workspace
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from .data import dump_jsonl
from typing import Iterator, List

from ft.build_dataset import (SKIP_DIRS, MinHashDedup, doc_title,
                              markdown_pairs, python_pairs)

# Third-party checkouts, build output, and duplicate checkouts of this same repo.
# `.claude/worktrees/` is the one that actually distorts the mix: it is a second
# copy of the workspace whose files differ from the originals by a few edits, so
# exact dedup misses them and near-dedup only catches some. Left in, it
# contributed 636 rows of the project restating itself.
VENDORED_DIRS = {"abapGit", "abapGit-build", "dist", "build",
                 ".ragstack_index", ".pytest_cache", "worktrees"}

JS_SUFFIXES = (".js", ".jsx", ".ts", ".tsx")

# A minified bundle is syntactically valid and semantically useless as training
# data. Line length separates it from hand-written source more reliably than any
# filename convention. Applied to code only - prose documents legitimately carry
# a single very long line, and filtering those would drop real guides.
MAX_LINE_CHARS = 400

QUOTES = ("\"", "'", "`")


def skip_dirs(include_vendored: bool) -> set:
    return SKIP_DIRS if include_vendored else SKIP_DIRS | VENDORED_DIRS


def project_of(path: Path, root: Path) -> str:
    """Top-level directory under the workspace root; root-level files get its name."""
    root = root.resolve()                            # `--input ..` must not become ".."
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        return root.name
    return rel.parts[0] if len(rel.parts) > 1 else root.name


def iter_source(root: Path, suffix: str, include_vendored: bool) -> Iterator[Path]:
    skip = skip_dirs(include_vendored)
    for p in root.rglob(f"*{suffix}"):
        if any(part in skip for part in p.parts):
            continue
        if p.is_file():
            yield p


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def read_code(path: Path) -> str | None:
    """Source text, or None when empty, unreadable, or machine-generated."""
    text = read_text(path)
    if text is None or not text.strip():
        return None
    if max((len(line) for line in text.splitlines()), default=0) > MAX_LINE_CHARS:
        return None                                  # minified / generated
    return text


def pair(user: str, assistant: str) -> dict:
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


# --------------------------------------------------------------------------- ABAP

ABAP_DOC = re.compile(r'^\s*"!\s?(.*)$')
ABAP_CLASS_DEF = re.compile(r"^\s*CLASS\s+(\S+)\s+DEFINITION", re.I)
ABAP_METHOD = re.compile(r"^\s*METHOD\s+(\S+?)\.\s*$", re.I)
ABAP_ENDMETHOD = re.compile(r"^\s*ENDMETHOD\.", re.I)


def _abap_doc_above(lines: List[str], idx: int) -> str:
    """Contiguous ABAP Doc (`"!`) block immediately above line `idx`."""
    out: List[str] = []
    i = idx - 1
    while i >= 0:
        m = ABAP_DOC.match(lines[i])
        if not m:
            break
        out.append(m.group(1).rstrip())
        i -= 1
    out.reverse()
    # Drop the synchronized-shorttext wrapper; it is markup, not prose.
    cleaned = [re.sub(r"<[^>]+>", "", ln).strip() for ln in out]
    return "\n".join(ln for ln in cleaned if ln).strip()


def abap_pairs(text: str, filename: str, min_chars: int, max_chars: int) -> Iterator[dict]:
    """Class-doc and method-implementation pairs from one ABAP include."""
    lines = text.splitlines()

    for i, line in enumerate(lines):
        m = ABAP_CLASS_DEF.match(line)
        if m:
            doc = _abap_doc_above(lines, i)
            if len(doc) >= 40:
                yield pair(f"What does the `{m.group(1)}` ABAP class in `{filename}` do?", doc)
            break                                    # one class per include

    # METHOD ... ENDMETHOD from the IMPLEMENTATION section. ABAP does not nest
    # these, so a linear scan is exact rather than merely adequate.
    i = 0
    while i < len(lines):
        m = ABAP_METHOD.match(lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not ABAP_ENDMETHOD.match(lines[j]):
            j += 1
        body = "\n".join(lines[i:j + 1])
        if min_chars <= len(body) <= max_chars:
            yield pair(f"Implement the `{m.group(1)}` method in `{filename}` (ABAP).",
                       f"```abap\n{body}\n```")
        i = j + 1


# ----------------------------------------------------------------------- JS / TS

JS_DECL = re.compile(
    r"^[ \t]*(?:export\s+)?(?:default\s+)?"
    r"(?:async\s+)?function\s+(?P<fn>[A-Za-z_$][\w$]*)"
    r"|^[ \t]*(?:export\s+)?const\s+(?P<const>[A-Za-z_$][\w$]*)\s*="
    r"\s*(?:async\s*)?\([^)]*\)\s*=>",
    re.M)


def _match_delimiter(src: str, search_from: int, opener: str, closer: str) -> int:
    """Index of the delimiter closing the first `opener` at/after `search_from`.

    String- and comment-aware: a brace inside a template literal or a comment
    would otherwise unbalance the count and swallow the rest of the file.
    """
    i = src.find(opener, search_from)
    if i < 0:
        return -1
    depth, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in QUOTES:
            quote, i = c, i + 1
            while i < n and src[i] != quote:
                i += 2 if src[i] == "\\" else 1
        elif c == "/" and i + 1 < n and src[i + 1] == "/":
            i = src.find("\n", i)
            if i < 0:
                return -1
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            i = src.find("*/", i)
            if i < 0:
                return -1
            i += 1
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _function_source(src: str, m: re.Match) -> str | None:
    """Declaration at `m` through the closing brace of its body.

    The body brace is not simply the next one after the match: a destructured
    parameter list (`function Auth({ onLogin }) {`) opens and closes a brace
    first, and matching that returns the signature alone - which silently
    truncated every React component here to a few dozen characters. So for a
    `function` declaration the parameter parens are skipped first. The arrow
    branch of JS_DECL already consumes through `=>`, so its body is next.
    """
    if m.group("const"):
        search_from = m.end()
    else:
        params_end = _match_delimiter(src, m.end(), "(", ")")
        if params_end < 0:
            return None
        search_from = params_end + 1
    close = _match_delimiter(src, search_from, "{", "}")
    return None if close < 0 else src[m.start():close + 1]


def _jsdoc_above(src: str, start: int) -> str:
    """JSDoc block immediately preceding `start`, stripped of its comment markup."""
    head = src[:start].rstrip()
    if not head.endswith("*/"):
        return ""
    open_at = head.rfind("/**")
    if open_at < 0:
        return ""
    lines = [re.sub(r"^\s*\*\s?", "", ln).strip()
             for ln in head[open_at + 3:-2].splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def _fence_lang(suffix: str) -> str:
    return {".tsx": "tsx", ".jsx": "jsx", ".ts": "typescript"}.get(suffix, "javascript")


def js_pairs(src: str, filename: str, suffix: str,
             min_chars: int, max_chars: int) -> Iterator[dict]:
    """Implementation pairs (plus a doc pair where JSDoc exists) from one module."""
    lang = _fence_lang(suffix)
    for m in JS_DECL.finditer(src):
        name = m.group("fn") or m.group("const")
        if not name:
            continue
        seg = _function_source(src, m)
        if seg is None or not (min_chars <= len(seg) <= max_chars):
            continue
        doc = _jsdoc_above(src, m.start())
        if len(doc) >= 40:
            yield pair(f"What does the `{name}` function in `{filename}` do?", doc)
            yield pair(f"Implement `{name}` in `{filename}`.\n\nDocstring:\n{doc}",
                       f"```{lang}\n{seg}\n```")
        else:
            yield pair(f"Implement `{name}` in `{filename}`.", f"```{lang}\n{seg}\n```")


# ---------------------------------------------------------------------------- SQL

SQL_CREATE = re.compile(
    r"CREATE\s+(?:TABLE|VIEW|INDEX)\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"[`\"\[]?(\w+)[`\"\]]?[\s\S]*?;",
    re.I)


def sql_pairs(src: str, filename: str, min_chars: int, max_chars: int) -> Iterator[dict]:
    """One pair per CREATE statement: the object's name -> its DDL."""
    for m in SQL_CREATE.finditer(src):
        stmt = m.group(0).strip()
        if min_chars <= len(stmt) <= max_chars:
            yield pair(f"Write the SQL schema for `{m.group(1)}` in `{filename}`.",
                       f"```sql\n{stmt}\n```")


# --------------------------------------------------------------------------- main

def collect(root: Path, min_chars: int, max_chars: int, rng,
            include_vendored: bool) -> tuple[List[dict], Counter]:
    """Every extractor over every project, each row tagged with where it came from."""
    rows: List[dict] = []
    counts: Counter = Counter()

    def emit(pairs: Iterator[dict], path: Path, kind: str) -> None:
        for r in pairs:
            r["project"] = project_of(path, root)
            r["source_kind"] = kind
            rows.append(r)
            counts[kind] += 1

    for p in iter_source(root, ".md", include_vendored):
        text = read_text(p)
        if text is not None:
            emit(markdown_pairs(text, doc_title(p), min_chars, max_chars, rng), p, "markdown")

    for p in iter_source(root, ".py", include_vendored):
        text = read_text(p)
        if text is not None:
            emit(python_pairs(text, p.name, min_chars, max_chars), p, "python")

    for p in iter_source(root, ".abap", include_vendored):
        text = read_code(p)
        if text is not None:
            emit(abap_pairs(text, p.name, min_chars, max_chars), p, "abap")

    for suffix in JS_SUFFIXES:
        for p in iter_source(root, suffix, include_vendored):
            text = read_code(p)
            if text is not None:
                emit(js_pairs(text, p.name, suffix, min_chars, max_chars), p, "js")

    for p in iter_source(root, ".sql", include_vendored):
        text = read_code(p)
        if text is not None:
            emit(sql_pairs(text, p.name, min_chars, max_chars), p, "sql")

    return rows, counts


def dedupe(rows: List[dict], seed: int, near_dedup: bool) -> tuple[List[dict], int, int]:
    """Exact then near-duplicate removal on the assistant answer."""
    seen, uniq, n_exact, n_near = set(), [], 0, 0
    near = MinHashDedup(seed=seed) if near_dedup else None
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
    return uniq, n_exact, n_near


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="workspace root holding every project")
    ap.add_argument("--out", default="data_workspace")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--max-per-project", type=int, default=0,
                    help="cap rows per project so one repo cannot dominate (0 = uncapped)")
    ap.add_argument("--max-examples", type=int, default=0, help="0 = no cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-vendored", action="store_true",
                    help="also mine abapGit/, dist/ and other third-party checkouts")
    ap.add_argument("--no-near-dedup", action="store_true",
                    help="keep near-duplicate answers (exact dedup still runs)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.input)

    rows, counts = collect(root, args.min_chars, args.max_chars, rng, args.include_vendored)
    print("extracted " + ", ".join(f"{k} {n}" for k, n in counts.most_common())
          + f" -> {len(rows)} raw")

    uniq, n_exact, n_near = dedupe(rows, args.seed, not args.no_near_dedup)
    print(f"after dedup: {len(uniq)}  (exact -{n_exact}, near -{n_near})")

    rng.shuffle(uniq)

    if args.max_per_project:
        per, capped = Counter(), []
        for r in uniq:
            if per[r["project"]] >= args.max_per_project:
                continue
            per[r["project"]] += 1
            capped.append(r)
        print(f"per-project cap {args.max_per_project}: {len(uniq)} -> {len(capped)}")
        uniq = capped

    if args.max_examples:
        uniq = uniq[:args.max_examples]

    by_project = Counter(r["project"] for r in uniq)
    print(f"\n{len(by_project)} projects in the dataset:")
    for proj, n in by_project.most_common():
        print(f"  {n:6d}  {proj}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_val = max(int(len(uniq) * args.val_frac), 1)
    for name, part in (("val", uniq[:n_val]), ("train", uniq[n_val:])):
        f = out / f"{name}.jsonl"
        f.write_text("\n".join(dump_jsonl(r) for r in part),
                     encoding="utf-8")
        print(f"  {name}: {len(part)} examples -> {f}")

    manifest = {"total": len(uniq),
                "by_project": dict(by_project),
                "by_source": dict(Counter(r["source_kind"] for r in uniq)),
                "dedup": {"exact": n_exact, "near": n_near},
                "include_vendored": args.include_vendored,
                "seed": args.seed}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
