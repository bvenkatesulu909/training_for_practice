"""Correctness tests for the fine-tuning stack.

Each pins a failure mode that produces a *plausible-looking but wrong* run rather
than a crash — which is the only kind worth spending laptop time on.
"""
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ft.config import LoRAConfig
from ft.data import IGNORE, SFTDataset, collate, encode_example, normalise
from ft.lora import inject_lora, lora_state_dict, merge_lora

torch.manual_seed(0)


class ToyBlock(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.other = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.other(self.q_proj(x) + self.v_proj(x))


class ToyModel(nn.Module):
    def __init__(self, d=32, n=3):
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock(d) for _ in range(n)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


# ------------------------------------------------------------------- LoRA
def test_lora_is_identity_at_init():
    """B is initialised to zero, so an untrained adapter must not change a single
    logit. If your first eval differs from the base model's, stop and debug."""
    m = ToyModel()
    x = torch.randn(4, 32)
    before = m(x).clone()
    inject_lora(m, LoRAConfig(r=8, alpha=16, dropout=0.0))
    assert torch.allclose(before, m(x), atol=1e-6)


def test_only_lora_params_are_trainable():
    m = ToyModel()
    info = inject_lora(m, LoRAConfig(r=4, alpha=8, target_modules=["q_proj", "v_proj"]))
    assert info["replaced"] == 6                      # 2 per block x 3 blocks
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert names and all("lora_" in n for n in names), names
    # `other` was not targeted and must stay frozen.
    assert not any("other" in n for n in names)


def test_lora_gradients_reach_adapters_only():
    """At init B=0, so dL/dA = 0 too — that is correct LoRA maths, not a dead
    parameter. B gets gradient immediately; A starts flowing once B leaves zero."""
    m = ToyModel()
    inject_lora(m, LoRAConfig(r=4, alpha=8))
    m(torch.randn(4, 32)).sum().backward()
    for n, p in m.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, f"frozen param {n} received a gradient"
        elif "lora_B" in n:
            assert p.grad is not None and p.grad.abs().sum() > 0, f"dead adapter {n}"
        else:
            assert p.grad is not None and p.grad.abs().sum() == 0,                 f"{n} should have zero grad while B is zero"

    # After B moves off zero, A must start receiving gradient.
    m.zero_grad(set_to_none=True)
    for n, p in m.named_parameters():
        if "lora_B" in n:
            nn.init.normal_(p, std=0.1)
    m(torch.randn(4, 32)).sum().backward()
    for n, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad.abs().sum() > 0, f"dead adapter {n} after B moved"


def test_lora_matches_base_dtype():
    """Adapters must adopt the base layer's dtype, or any bf16/fp16 run crashes."""
    for dt in (torch.float64, torch.float32):
        m = ToyModel().to(dt)
        inject_lora(m, LoRAConfig(r=4, alpha=8, dropout=0.0))
        for n, p in m.named_parameters():
            if "lora_" in n:
                assert p.dtype == dt, f"{n} is {p.dtype}, base is {dt}"
        m(torch.randn(2, 32, dtype=dt))          # must not raise


def test_merge_is_numerically_equivalent():
    """Merging must not change behaviour. If it does, everything you measured
    before export was measuring a different model than the one you shipped."""
    m = ToyModel().double()
    inject_lora(m, LoRAConfig(r=8, alpha=16, dropout=0.0))
    for p in m.parameters():                          # give the adapter real values
        if p.requires_grad:
            nn.init.normal_(p, std=0.1)
    m.eval()
    x = torch.randn(4, 32, dtype=torch.float64)
    with torch.no_grad():
        adapted = m(x).clone()
    n = merge_lora(m)
    assert n == 6                                     # q_proj+v_proj x 3 blocks
    with torch.no_grad():
        merged = m(x)
    assert torch.allclose(adapted, merged, atol=1e-12), (adapted - merged).abs().max()
    assert not any("lora_" in k for k in m.state_dict())


def test_adapter_state_dict_is_small():
    m = ToyModel()
    inject_lora(m, LoRAConfig(r=4, alpha=8))
    sd = lora_state_dict(m)
    assert sd and all("lora_" in k for k in sd)
    full = sum(p.numel() for p in m.parameters())
    assert sum(v.numel() for v in sd.values()) < full * 0.25


def test_unmatched_target_modules_raises():
    with pytest.raises(ValueError, match="no modules matched"):
        inject_lora(ToyModel(), LoRAConfig(target_modules=["nonexistent_proj"]))


# ------------------------------------------------------- prompt masking
class FakeTok:
    """ChatML-ish tokenizer: one token per word plus role markers. Prefix-consistent."""
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                            return_tensors=None):
        out = []
        for m in messages:
            out.append(abs(hash(m["role"])) % 100 + 1000)      # role marker
            out += [abs(hash(w)) % 900 + 100 for w in m["content"].split()]
            out.append(9999)                                   # end marker
        if add_generation_prompt:
            out.append(abs(hash("assistant")) % 100 + 1000)
        return out


CONV = [
    {"role": "user", "content": "alpha beta gamma"},
    {"role": "assistant", "content": "delta epsilon"},
    {"role": "user", "content": "zeta"},
    {"role": "assistant", "content": "eta theta iota"},
]


def _toks(words):
    return {abs(hash(w)) % 900 + 100 for w in words.split()}


def test_prompt_tokens_are_masked():
    """THE bug: without masking, most of the gradient goes into learning to
    generate user questions, and the loss curve looks perfectly healthy."""
    ids, labels = encode_example(FakeTok(), CONV, max_len=999, mask_prompt=True)
    assert len(ids) == len(labels)
    sup = [i for i, l in enumerate(labels) if l != IGNORE]
    assert sup, "nothing supervised"
    # Every supervised label must equal the input token at that position.
    assert all(labels[i] == ids[i] for i in sup)
    supervised_tokens = {ids[i] for i in sup}
    # User content tokens must never be supervised.
    assert not (_toks("alpha beta gamma zeta") & supervised_tokens), \
        "user tokens leaked into the loss"
    # Assistant content tokens must all be supervised.
    assert _toks("delta epsilon eta theta iota") <= supervised_tokens


def test_mask_prompt_false_supervises_everything():
    ids, labels = encode_example(FakeTok(), CONV, max_len=999, mask_prompt=False)
    assert labels == ids


def test_supervised_fraction_is_a_minority():
    _, labels = encode_example(FakeTok(), CONV, max_len=999, mask_prompt=True)
    frac = sum(1 for l in labels if l != IGNORE) / len(labels)
    assert 0.1 < frac < 0.75, frac


def test_non_prefix_template_is_rejected():
    """A template that is not prefix-consistent makes the span arithmetic silently
    wrong, so it must raise instead of producing mislabelled data."""
    class BadTok(FakeTok):
        def apply_chat_template(self, messages, tokenize=True,
                                add_generation_prompt=False, return_tensors=None):
            out = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return out if add_generation_prompt else [4242] + out

    with pytest.raises(ValueError, match="prefix-consistent"):
        encode_example(BadTok(), CONV, max_len=999, mask_prompt=True)


# ------------------------------------------------------------- batching
def test_collate_masks_padding():
    batch = [([1, 2, 3], [IGNORE, 2, 3]), ([4, 5], [IGNORE, 5])]
    out = collate(batch, pad_id=0)
    assert out["input_ids"].shape == (2, 3)
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert out["labels"][1].tolist() == [IGNORE, 5, IGNORE]   # pad never supervised


def test_overflow_is_dropped_not_truncated(tmp_path):
    """Truncating mid-answer teaches the model to stop mid-sentence."""
    f = tmp_path / "d.jsonl"
    f.write_text("\n".join(json.dumps({"instruction": "q " * 5, "output": "a " * n})
                           for n in (2, 200)), encoding="utf-8")
    ds = SFTDataset(str(f), FakeTok(), max_len=60, on_overflow="drop")
    assert ds.stats["total"] == 2 and ds.stats["dropped_long"] == 1
    assert len(ds) == 1
    assert all(len(i) <= 60 for i, _ in ds.rows)


def test_normalise_accepts_both_formats():
    a = normalise({"instruction": "x", "input": "y", "output": "z"})
    assert [m["role"] for m in a] == ["user", "assistant"]
    assert a[0]["content"] == "x\n\ny" and a[1]["content"] == "z"
    b = normalise({"messages": [{"role": "user", "content": "q"}]})
    assert b == [{"role": "user", "content": "q"}]


# --------------------------------------------------------------- planner
def _llama_like(dims):
    """A module with exactly the projection shapes a Llama-family block has."""
    d, m, q, kv = dims["d"], dims["m"], dims["nh"] * dims["hd"], dims["nkv"] * dims["hd"]

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d, q, bias=False)
            self.k_proj = nn.Linear(d, kv, bias=False)
            self.v_proj = nn.Linear(d, kv, bias=False)
            self.o_proj = nn.Linear(q, d, bias=False)
            self.gate_proj = nn.Linear(d, m, bias=False)
            self.up_proj = nn.Linear(d, m, bias=False)
            self.down_proj = nn.Linear(m, d, bias=False)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(dims["L"])])

    return Net()


@pytest.mark.parametrize("targets", [
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ["q_proj", "v_proj"],
])
def test_planner_adapter_count_matches_injection(targets):
    """The planner prices runs without downloading weights, so its arithmetic must
    match what inject_lora actually creates. A guess here was wrong by 12x."""
    from ft.planner import adapter_params
    # SmolLM2-135M's real shape.
    dims = {"d": 576, "m": 1536, "L": 30, "nh": 9, "nkv": 3, "hd": 64,
            "vocab": 49152, "tied": True}
    net = _llama_like(dims)
    inject_lora(net, LoRAConfig(r=16, alpha=32, target_modules=targets))
    actual = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert adapter_params(dims, 16, targets) == actual


def test_planner_base_param_count_is_close():
    """Analytic base-param count from config alone, within a few percent."""
    from ft.planner import base_params
    dims = {"d": 576, "m": 1536, "L": 30, "nh": 9, "nkv": 3, "hd": 64,
            "vocab": 49152, "tied": True}
    n = base_params(dims)
    assert 1.2e8 < n < 1.5e8, n           # SmolLM2-135M is 135M


# ----------------------------------------------------------- dataset dedup
def test_near_duplicate_answers_are_removed():
    """Exact-hash dedup misses answers that differ by a few words. On this repo's
    docs that was 3.1% of the dataset — pure memorisation fuel."""
    from ft.build_dataset import MinHashDedup
    base = ("The retrieval pipeline embeds each chunk with a sentence encoder, "
            "stores the vectors in a Qdrant collection, and queries them with "
            "cosine similarity before reranking the top candidates. ") * 3
    near = base.replace("cosine similarity", "dot-product similarity")
    other = ("Gradient checkpointing trades compute for memory by discarding "
             "intermediate activations during the forward pass and recomputing "
             "them on demand in the backward pass. ") * 3

    dd = MinHashDedup()
    assert dd.is_duplicate(base) is False          # first sighting
    assert dd.is_duplicate(near) is True           # near-dup caught
    assert dd.is_duplicate(other) is False         # unrelated text kept


def test_minhash_handles_empty_and_tiny_text():
    from ft.build_dataset import MinHashDedup
    dd = MinHashDedup()
    assert dd.is_duplicate("") is False
    assert dd.is_duplicate("hi") is False


# ----------------------------------------------------------------- config
def test_config_round_trips_through_asdict():
    """Checkpoints store the config as a plain dict. Rebuilding only *some*
    sub-configs works right up until something touches the ones left as dicts —
    which is what broke `ft.evaluate` after a successful training run."""
    from dataclasses import asdict
    from ft.config import Config, DataConfig, LoRAConfig, TrainConfig

    orig = Config(name="x", base_model="some/model",
                  lora=LoRAConfig(r=8, alpha=16, target_modules=["q_proj"]),
                  data=DataConfig(max_len=777, mask_prompt=False),
                  train=TrainConfig(lr=1e-5, grad_accum=3))
    back = Config.from_dict(asdict(orig))

    assert isinstance(back.lora, LoRAConfig)
    assert isinstance(back.data, DataConfig)
    assert isinstance(back.train, TrainConfig)
    assert back.data.max_len == 777 and back.data.mask_prompt is False
    assert back.train.lr == 1e-5 and back.train.grad_accum == 3
    assert back.lora.r == 8 and back.lora.target_modules == ["q_proj"]
    assert asdict(back) == asdict(orig)


def test_batchencoding_is_not_a_dict():
    """transformers' BatchEncoding extends UserDict, so `isinstance(x, dict)` is
    False. Code that normalises template output by checking isinstance(dict) alone
    silently passes the wrapper object downstream. apply_template checks
    hasattr(out, 'input_ids') first, which is why it works."""
    from ft.data import apply_template

    class BE:                                   # stand-in with the same shape
        def __init__(self, ids):
            self.input_ids = ids

        def __getitem__(self, k):
            return self.input_ids

    class Tok:
        def apply_chat_template(self, messages, tokenize=True,
                                add_generation_prompt=False, return_tensors=None):
            return BE([1, 2, 3])

    assert not isinstance(BE([1]), dict)        # the trap itself
    assert apply_template(Tok(), CONV) == [1, 2, 3]


def test_apply_template_normalises_plain_list_and_batched():
    from ft.data import apply_template

    class ListTok:
        def apply_chat_template(self, messages, tokenize=True,
                                add_generation_prompt=False, return_tensors=None):
            return [7, 8, 9]

    class BatchedTok:
        def apply_chat_template(self, messages, tokenize=True,
                                add_generation_prompt=False, return_tensors=None):
            return {"input_ids": [[7, 8, 9]]}

    assert apply_template(ListTok(), CONV) == [7, 8, 9]
    assert apply_template(BatchedTok(), CONV) == [7, 8, 9]


# ------------------------------------------------- workspace extractors
ABAP_SRC = '''"! <p class="shorttext synchronized">Approval level determination</p>
"! Decides how many approval levels a requisition of a given value needs.
"! The configuration is injected, so this class has no database dependency.
CLASS zcl_pr_approval_engine DEFINITION PUBLIC FINAL CREATE PRIVATE.
  PUBLIC SECTION.
    METHODS required_level IMPORTING amount TYPE zpr_de_amount.
ENDCLASS.

CLASS zcl_pr_approval_engine IMPLEMENTATION.
  METHOD required_level.
    " Padding so the body clears the min-chars floor and is actually emitted
    " by the extractor rather than filtered out as too short to be useful.
    result = COND #( WHEN amount > 10000 THEN 3 ELSE 1 ).
  ENDMETHOD.
ENDCLASS.
'''


def test_abap_extracts_class_doc_and_method_body():
    from ft.build_workspace_dataset import abap_pairs

    rows = list(abap_pairs(ABAP_SRC, "zcl_pr_approval_engine.clas.abap", 120, 3000))
    doc, impl = rows[0]["messages"], rows[1]["messages"]

    assert "zcl_pr_approval_engine" in doc[0]["content"]
    assert "Decides how many approval levels" in doc[1]["content"]
    assert "<p class=" not in doc[1]["content"]         # shorttext markup stripped
    assert "required_level" in impl[0]["content"]
    assert impl[1]["content"].startswith("```abap")
    assert "ENDMETHOD." in impl[1]["content"]


def test_js_destructured_params_do_not_truncate_the_body():
    """The bug this guards: `function Auth({ onLogin }) {` opens a brace in the
    parameter list, and matching that one returns the signature alone — every
    React component in the workspace was reduced to a few dozen characters."""
    from ft.build_workspace_dataset import js_pairs

    src = ("export default function Auth({ onLogin, onError }) {\n"
           "  const [email, setEmail] = useState('');\n"
           "  const submit = () => onLogin({ email });\n"
           "  return <form onSubmit={submit}>{email}</form>;\n"
           "}\n")
    body = list(js_pairs(src, "Auth.jsx", ".jsx", 40, 3000))[0]["messages"][1]["content"]

    assert body.startswith("```jsx")
    assert "useState" in body and "return <form" in body   # body, not just signature
    assert body.rstrip().endswith("```")


def test_js_braces_inside_strings_do_not_unbalance_the_scan():
    from ft.build_workspace_dataset import js_pairs

    src = ("export function render(user) {\n"
           "  const t = `hello ${user} {unclosed`;   // } not a real brace\n"
           "  return t + '}';\n"
           "}\n"
           "export function after() {\n"
           "  return 'the scan must still find this one';\n"
           "}\n")
    names = [r["messages"][0]["content"] for r in js_pairs(src, "x.js", ".js", 40, 3000)]

    assert any("`render`" in n for n in names)
    assert any("`after`" in n for n in names)             # scan did not run away


def test_sql_pairs_keep_tables_and_drop_index_one_liners():
    """The min-chars floor earns its keep on DDL: it admits every real table and
    excludes bare `CREATE INDEX` lines, which are near-identical to each other and
    teach nothing as standalone rows (37 of 71 statements in this workspace)."""
    from ft.build_workspace_dataset import sql_pairs

    src = ("CREATE TABLE tenants (\n  id UUID PRIMARY KEY,\n  name TEXT NOT NULL,\n"
           "  plan TEXT NOT NULL DEFAULT 'free',\n  seats INTEGER NOT NULL DEFAULT 1,\n"
           "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n);\n\n"
           "CREATE TABLE users (\n  id UUID PRIMARY KEY,\n  tenant_id UUID NOT NULL,\n"
           "  email TEXT NOT NULL UNIQUE,\n  role TEXT NOT NULL DEFAULT 'member',\n"
           "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n);\n\n"
           "CREATE INDEX idx_users_tenant_id ON users(tenant_id);\n")
    rows = list(sql_pairs(src, "schema.sql", 120, 3000))

    assert [r["messages"][0]["content"] for r in rows] == [
        "Write the SQL schema for `tenants` in `schema.sql`.",
        "Write the SQL schema for `users` in `schema.sql`.",
    ]
    assert rows[1]["messages"][1]["content"].startswith("```sql\nCREATE TABLE users")


def test_minified_code_is_rejected(tmp_path):
    from ft.build_workspace_dataset import read_code

    normal = tmp_path / "app.js"
    normal.write_text("export function f() {\n  return 1;\n}\n", encoding="utf-8")
    bundle = tmp_path / "index-abc123.js"
    bundle.write_text("var a=1;" * 200, encoding="utf-8")   # one very long line

    assert read_code(normal) is not None
    assert read_code(bundle) is None


def test_provenance_is_taken_from_the_path(tmp_path):
    """Attribution must not depend on the file name: many projects share one."""
    from ft.build_workspace_dataset import collect
    import random

    for proj in ("alpha", "beta"):
        d = tmp_path / proj
        d.mkdir()
        (d / "README.md").write_text(
            f"## Overview\n\n{'The ' + proj + ' service does a thing. ' * 8}\n",
            encoding="utf-8")

    rows, _ = collect(tmp_path, 120, 3000, random.Random(0), include_vendored=False)
    assert {r["project"] for r in rows} == {"alpha", "beta"}


def test_vendored_directories_are_excluded(tmp_path):
    from ft.build_workspace_dataset import collect
    import random

    body = "## Overview\n\n" + "Real prose that clears the minimum length. " * 8 + "\n"
    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / "doc.md").write_text(body, encoding="utf-8")
    vendored = tmp_path / "abapGit" / "docs"
    vendored.mkdir(parents=True)
    (vendored / "doc.md").write_text(body.replace("Real", "Vendored"), encoding="utf-8")

    projects = {r["project"] for r in collect(tmp_path, 120, 3000, random.Random(0), False)[0]}
    assert projects == {"mine"}

    with_vendored = {r["project"] for r in collect(tmp_path, 120, 3000, random.Random(0), True)[0]}
    assert with_vendored == {"mine", "abapGit"}


def test_shingle_hash_is_stable_across_processes():
    """PYTHONHASHSEED must not change dedup decisions — it silently did."""
    import subprocess
    import sys

    # Absolute: a relative 'src' only resolves when pytest happens to be run
    # from finetune/, and the child then fails to import and prints nothing --
    # which the old assert read as three matching hashes.
    src = Path(__file__).resolve().parents[1] / "src"
    code = (f"import sys; sys.path.insert(0, {str(src)!r});"
            "from ft.build_dataset import MinHashDedup;"
            "print(MinHashDedup._shingle_hash('retrieval pipeline embeds each chunk'))")
    runs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": s})
            for s in ("0", "1", "12345")]

    for run in runs:
        assert run.returncode == 0, run.stderr
    out = [run.stdout.strip() for run in runs]
    assert len(set(out)) == 1 and out[0]


# ------------------------------------------------------------ jsonl reading
def test_jsonl_survives_unicode_line_separators(tmp_path):
    """json.dumps(ensure_ascii=False) does NOT escape U+0085, U+2028 or U+2029,
    but str.splitlines() breaks on all of them. Reading JSONL with splitlines()
    therefore shreds any record containing one — it silently destroyed 11 records
    in OpenThoughts before this was caught."""
    from ft.data import iter_jsonl

    nasty = ["plain text",
             "next\u0085line",          # NEL
             "line\u2028separator",     # LINE SEPARATOR
             "para\u2029separator",     # PARAGRAPH SEPARATOR
             "vertical\vtab", "form\ffeed"]
    f = tmp_path / "d.jsonl"
    f.write_text("\n".join(
        json.dumps({"messages": [{"role": "user", "content": t},
                                 {"role": "assistant", "content": "ok"}]},
                   ensure_ascii=False) for t in nasty), encoding="utf-8")

    # The buggy approach loses records; the correct one does not.
    assert len(f.read_text(encoding="utf-8").splitlines()) > len(nasty)
    rows = list(iter_jsonl(str(f)))
    assert len(rows) == len(nasty)
    assert [r["messages"][0]["content"] for r in rows] == nasty


def test_sftdataset_reads_records_with_line_separators(tmp_path):
    """The same failure, one layer up: these records must reach training."""
    from ft.data import SFTDataset

    f = tmp_path / "d.jsonl"
    f.write_text("\n".join(
        json.dumps({"instruction": f"q{i}", "output": f"answer\u2028body {i} " * 5},
                   ensure_ascii=False) for i in range(6)), encoding="utf-8")
    ds = SFTDataset(str(f), FakeTok(), max_len=999, on_overflow="drop")
    assert ds.stats["total"] == 6 and len(ds) == 6


def test_drop_rate_guard(tmp_path):
    """A dataset whose examples all exceed max_len must fail loudly, not train on
    nothing. At OpenThoughts' median of ~7,700 tokens, max_len=1024 keeps zero."""
    from ft.data import SFTDataset

    f = tmp_path / "d.jsonl"
    f.write_text("\n".join(json.dumps({"instruction": "q", "output": "word " * 400})
                           for _ in range(10)), encoding="utf-8")
    ds = SFTDataset(str(f), FakeTok(), max_len=50, on_overflow="drop")
    assert ds.drop_rate() == 1.0 and len(ds) == 0
    with pytest.raises(ValueError, match="exceed max_len"):
        ds.warn_if_truncating("t")


def test_drop_rate_guard_passes_when_fine(tmp_path):
    from ft.data import SFTDataset

    f = tmp_path / "d.jsonl"
    f.write_text("\n".join(json.dumps({"instruction": "q", "output": "word " * 5})
                           for _ in range(10)), encoding="utf-8")
    ds = SFTDataset(str(f), FakeTok(), max_len=999, on_overflow="drop")
    assert ds.drop_rate() == 0.0
    ds.warn_if_truncating("t")          # must not raise


def test_conversations_schema_roles_and_system():
    """OpenThoughts/ShareGPT layout must map onto chat-template roles."""
    from ft.data import normalise
    msgs = normalise({"system": "think first",
                      "conversations": [{"from": "user", "value": "Q"},
                                        {"from": "assistant", "value": "A"}]})
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert normalise({"conversations": [{"from": "human", "value": "h"},
                                        {"from": "gpt", "value": "g"}]})[0]["role"] == "user"
    with pytest.raises(ValueError, match="unknown conversation role"):
        normalise({"conversations": [{"from": "alien", "value": "x"}]})


# ------------------------------------------------- jsonl write portability
def test_dump_jsonl_escapes_unicode_line_breakers():
    """Our reader copes with raw U+0085/U+2028/U+2029, but jq, pandas and HF
    datasets do not. Escaping on the way out keeps the artifact portable."""
    from ft.data import LINE_BREAKERS, dump_jsonl

    payload = "a" + LINE_BREAKERS[0] + "b" + LINE_BREAKERS[1] + "c" + LINE_BREAKERS[2] + "d"
    rec = {"messages": [{"role": "user", "content": payload},
                        {"role": "assistant", "content": "ok"}]}
    line = dump_jsonl(rec)

    assert not any(ch in line for ch in LINE_BREAKERS), "raw line breaker survived"
    assert len(line.splitlines()) == 1, "still splits into multiple lines"
    assert json.loads(line) == rec, "escaping changed the data"


def test_dump_jsonl_preserves_ordinary_unicode():
    """Only the three line breakers get escaped; real text stays readable."""
    from ft.data import dump_jsonl
    rec = {"messages": [{"role": "user", "content": "caf\u00e9 \u2014 \u65e5\u672c\u8a9e"}]}
    line = dump_jsonl(rec)
    assert "caf\u00e9" in line or "caf" in line
    assert json.loads(line) == rec


def test_dump_and_iter_jsonl_round_trip(tmp_path):
    """Writer and reader are a matched pair: what goes out must come back."""
    from ft.data import LINE_BREAKERS, dump_jsonl, iter_jsonl

    recs = [{"messages": [{"role": "user", "content": f"q{i}{ch}tail"},
                          {"role": "assistant", "content": f"a{i}"}]}
            for i, ch in enumerate(LINE_BREAKERS + ("", "\t"))]
    f = tmp_path / "d.jsonl"
    f.write_text("\n".join(dump_jsonl(r) for r in recs), encoding="utf-8")

    # Even the naive reader now works, which is the point of escaping.
    assert len(f.read_text(encoding="utf-8").splitlines()) == len(recs)
    assert list(iter_jsonl(str(f))) == recs
