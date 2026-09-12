"""Instruction dataset: chat templating, prompt masking, batching.

The load-bearing part is `encode_example`. In SFT the loss must be computed on the
assistant's tokens ONLY. If the prompt is not masked, the model spends most of its
gradient learning to generate *user questions*, and the run still produces a
smooth, plausible-looking loss curve while doing it. This is the most common silent
SFT bug, so the masking is derived from the tokenizer's own chat template rather
than from hand-written string markers, and it is pinned by tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch

IGNORE = -100


LINE_BREAKERS = (chr(0x85), chr(0x2028), chr(0x2029))


def dump_jsonl(obj) -> str:
    """Serialise one record as a JSONL line that every reader can split.

    `json.dumps(ensure_ascii=False)` escapes everything below U+0020 but leaves
    U+0085 (NEL), U+2028 (LINE SEPARATOR) and U+2029 (PARAGRAPH SEPARATOR) raw.
    Those are legal inside a JSON string, and Python's `str.splitlines()` -- plus
    jq, pandas.read_json(lines=True) and HuggingFace datasets -- all treat them as
    newlines. Our reader copes; other people's tools do not, so escape them on the
    way out and the artifact stays portable.

    Escaping happens inside string literals only: these characters are not JSON
    structural syntax, so json.dumps can never emit them anywhere else.
    """
    line = json.dumps(obj, ensure_ascii=False)
    for ch in LINE_BREAKERS:
        line = line.replace(ch, "\\u%04x" % ord(ch))
    return line


def iter_jsonl(path: str):
    """Read JSONL by real newlines only.

    The mirror of dump_jsonl: iterating the file object splits on "\\n" alone,
    which is what JSONL actually means. Reading with `read_text().splitlines()`
    silently shredded 11 OpenThoughts records into unparseable fragments.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ShareGPT-style role labels seen in the wild, mapped to chat-template roles.
ROLE_MAP = {"user": "user", "human": "user", "prompter": "user",
            "assistant": "assistant", "gpt": "assistant", "bot": "assistant",
            "system": "system"}


def normalise(rec: dict) -> List[dict]:
    """Accept {"messages":[...]}, {"instruction","input","output"}, or the
    ShareGPT/OpenThoughts {"system", "conversations":[{"from","value"}]} layout."""
    if "messages" in rec:
        return rec["messages"]

    if "conversations" in rec:
        msgs = []
        if rec.get("system"):
            msgs.append({"role": "system", "content": rec["system"]})
        for t in rec["conversations"]:
            role = ROLE_MAP.get((t.get("from") or "").lower())
            if role is None:
                raise ValueError(f"unknown conversation role {t.get('from')!r}")
            msgs.append({"role": role, "content": t.get("value") or ""})
        return msgs

    instr = rec.get("instruction", "")
    inp = rec.get("input", "")
    user = f"{instr}\n\n{inp}".strip() if inp else instr
    msgs = []
    if rec.get("system"):
        msgs.append({"role": "system", "content": rec["system"]})
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": rec["output"]})
    return msgs


def apply_template(tok, messages: List[dict], add_generation_prompt: bool = False) -> List[int]:
    """Render a conversation to a flat list of token ids.

    transformers 4.x returns List[int] here; 5.x returns a BatchEncoding dict.
    Normalising in one place avoids silently comparing dict *keys* downstream —
    which is exactly the bug the prefix check caught during development. Note
    BatchEncoding extends UserDict, so `isinstance(out, dict)` is False; the
    hasattr check has to come first.
    """
    out = tok.apply_chat_template(messages, tokenize=True,
                                  add_generation_prompt=add_generation_prompt)
    if hasattr(out, "input_ids") or isinstance(out, dict):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):     # batched render
        out = out[0]
    return [int(t) for t in out]


def encode_example(tok, messages: List[dict], max_len: int,
                   mask_prompt: bool = True) -> Tuple[List[int], List[int]]:
    """Return (input_ids, labels) with non-assistant tokens set to IGNORE.

    Spans are located by re-rendering prefixes of the conversation through the
    tokenizer's own template, so this works for ChatML, Llama-3, Gemma, and
    anything else without per-family string constants.
    """
    ids = apply_template(tok, messages)
    if not mask_prompt:
        return ids[:max_len], ids[:max_len]

    labels = [IGNORE] * len(ids)
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        head = apply_template(tok, messages[:i], add_generation_prompt=True)
        upto = apply_template(tok, messages[:i + 1])
        # The incremental renders must be prefixes of the full render, or the span
        # arithmetic is meaningless. Fail loudly rather than train on wrong labels.
        if ids[:len(head)] != head or ids[:len(upto)] != upto:
            raise ValueError(
                "chat template is not prefix-consistent; cannot derive assistant "
                "spans safely for this tokenizer"
            )
        labels[len(head):len(upto)] = ids[len(head):len(upto)]

    return ids, labels


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, tok, max_len: int,
                 on_overflow: str = "drop", mask_prompt: bool = True):
        self.rows: List[Tuple[List[int], List[int]]] = []
        self.stats = {"total": 0, "dropped_long": 0, "dropped_empty": 0}

        for rec in iter_jsonl(path):
            self.stats["total"] += 1
            ids, labels = encode_example(tok, normalise(rec), max_len, mask_prompt)
            if len(ids) > max_len:
                if on_overflow == "drop":
                    self.stats["dropped_long"] += 1
                    continue
                ids, labels = ids[:max_len], labels[:max_len]
            # An example with no supervised token contributes nothing but does
            # produce NaN if it is ever alone in a batch.
            if all(l == IGNORE for l in labels):
                self.stats["dropped_empty"] += 1
                continue
            self.rows.append((ids, labels))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

    def drop_rate(self) -> float:
        """Share of source examples discarded for exceeding max_len."""
        return self.stats["dropped_long"] / max(self.stats["total"], 1)

    def warn_if_truncating(self, name: str = "dataset") -> None:
        """A dataset whose examples are mostly longer than max_len produces an
        empty or unrepresentative training set. At OpenThoughts' median of ~7,700
        tokens, max_len=1024 keeps literally nothing — and nothing about that is
        visible in a loss curve, because there is no loss curve."""
        r = self.drop_rate()
        if r > 0.5:
            raise ValueError(
                f"{name}: {r:.1%} of examples exceed max_len and were dropped "
                f"({len(self.rows):,} of {self.stats['total']:,} remain). Raise "
                f"data.max_len, or set on_overflow: truncate if you accept "
                f"cutting answers mid-sentence."
            )
        if r > 0.1:
            print(f"  WARNING {name}: {r:.1%} of examples dropped for exceeding max_len")

    def supervised_fraction(self) -> float:
        """Share of tokens that actually carry loss. Typically 0.2-0.6; if this is
        ~1.0 your prompt masking is off."""
        sup = sum(sum(1 for l in lab if l != IGNORE) for _, lab in self.rows)
        tot = sum(len(i) for i, _ in self.rows)
        return sup / max(tot, 1)


def collate(batch, pad_id: int):
    n = max(len(i) for i, _ in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = n - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [IGNORE] * pad)      # padding never contributes loss
        attn.append([1] * len(ids) + [0] * pad)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}
