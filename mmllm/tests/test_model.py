"""Correctness tests. These are the point of running anything on the laptop:
prove the architecture is right before renting a GPU by the hour.

    python -m pytest tests -q
"""
import math
from pathlib import Path

import pytest
import torch

from mmllm.config import Config, ModelConfig, VisionConfig
from mmllm.model import MMLLM, VISION_PLACEHOLDER
from mmllm.model.ssd import ssd_chunked, ssd_reference
from mmllm.model.attention import sliding_window_mask
from mmllm.model.rope import build_rope_3d, build_rope_cache, apply_rope
from mmllm.train.schedule import lr_at

torch.manual_seed(0)


def tiny_cfg(**kw):
    base = dict(vocab_size=256, dim=64, n_layers=6, n_heads=4, n_kv_heads=2,
                window=8, ssd_heads=4, ssd_head_dim=16, ssd_state=16,
                ssd_chunk=8, max_seq_len=64,
                vision=VisionConfig(enabled=False))
    base.update(kw)
    return ModelConfig(**base)


# ---------------------------------------------------------------- SSD kernel
@pytest.mark.parametrize("L,chunk", [(32, 8), (30, 8), (64, 16), (7, 8)])
def test_ssd_chunked_matches_reference(L, chunk):
    """The chunked scan is the whole long-context argument. If it disagrees with
    the literal recurrence, every downstream number is fiction."""
    b, h, p, n, g = 2, 4, 16, 8, 2
    x = torch.randn(b, L, h, p, dtype=torch.float64)
    dt = torch.rand(b, L, h, dtype=torch.float64) * 0.1 + 0.01
    A = -torch.rand(h, dtype=torch.float64) - 0.5
    B = torch.randn(b, L, g, n, dtype=torch.float64)
    C = torch.randn(b, L, g, n, dtype=torch.float64)
    D = torch.randn(h, dtype=torch.float64)

    ref = ssd_reference(x, dt, A, B, C, D)
    got = ssd_chunked(x, dt, A, B, C, D, chunk=chunk)
    assert torch.allclose(ref, got, atol=1e-8, rtol=1e-6), (ref - got).abs().max()


def test_ssd_is_causal():
    """Changing token t must not alter any output before t."""
    b, L, h, p, n = 1, 24, 2, 8, 8
    args = lambda x: (x, torch.rand(b, L, h) * .1 + .01, -torch.rand(h) - .5,
                      torch.randn(b, L, 1, n), torch.randn(b, L, 1, n), torch.randn(h))
    torch.manual_seed(1)
    x = torch.randn(b, L, h, p)
    a = args(x)
    y1 = ssd_chunked(*a, chunk=8)
    x2 = x.clone()
    x2[:, 12:] += 5.0
    y2 = ssd_chunked(x2, *a[1:], chunk=8)
    assert torch.allclose(y1[:, :12], y2[:, :12], atol=1e-6)
    assert not torch.allclose(y1[:, 12:], y2[:, 12:], atol=1e-3)


# ------------------------------------------------------------------ masking
def test_sliding_window_mask():
    m = sliding_window_mask(4, 4, window=2, device="cpu")
    expect = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]).bool()
    assert torch.equal(m, expect)


# --------------------------------------------------------------------- rope
def test_rope_3d_shape_and_image_case():
    cos, sin = build_rope_3d(1, 4, 4, head_dim=32)
    assert cos.shape == (16, 32)
    cos, sin = build_rope_3d(8, 4, 4, head_dim=30)   # not divisible by 6
    assert cos.shape == (128, 30)


def test_rope_relative_invariance():
    """RoPE must encode relative position: shifting both q and k leaves the dot
    product unchanged. This is what lets a 4k-trained model extend to 300k."""
    cos, sin = build_rope_cache(64, 16, theta=10000.0)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)
    d1 = (apply_rope(q, cos, sin, 5) * apply_rope(k, cos, sin, 3)).sum()
    d2 = (apply_rope(q, cos, sin, 25) * apply_rope(k, cos, sin, 23)).sum()
    assert torch.allclose(d1, d2, atol=1e-4)


# -------------------------------------------------------------------- model
def test_forward_and_loss():
    cfg = tiny_cfg()
    m = MMLLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = m(x, targets=x)
    assert logits.shape == (2, 32, cfg.vocab_size)
    assert math.isfinite(loss.item())
    # An untrained model should sit near ln(vocab).
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 1.5


def test_lm_is_causal():
    cfg = tiny_cfg()
    m = MMLLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 24))
    with torch.no_grad():
        a, _ = m(x, targets=x)
        x2 = x.clone(); x2[:, 16:] = (x2[:, 16:] + 7) % cfg.vocab_size
        b, _ = m(x2, targets=x2)
    assert torch.allclose(a[:, :16], b[:, :16], atol=1e-4), "future tokens leaked into the past"


def test_gradients_flow_everywhere():
    cfg = tiny_cfg()
    m = MMLLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    _, loss = m(x, targets=x)
    loss.backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not dead, f"no gradient reached: {dead}"


def test_ssd_prefill_then_step_matches_full_scan():
    """The chunked prefill and the single-token decode path are two separate
    implementations of one recurrence. If they drift, generation degrades in a
    way no loss curve will ever show you."""
    from mmllm.model.ssd import SSDMixer
    cfg = tiny_cfg()
    mix = SSDMixer(cfg).eval().double()
    u = torch.randn(2, 17, cfg.dim, dtype=torch.float64)
    with torch.no_grad():
        full, _ = mix(u)
        pre, cache = mix(u[:, :-1])
        step, _ = mix(u[:, -1:], cache)
    assert torch.allclose(full[:, :-1], pre, atol=1e-9)
    assert torch.allclose(full[:, -1:], step, atol=1e-9), (full[:, -1:] - step).abs().max()


@pytest.mark.parametrize("pattern", [
    ["swa", "global"],                    # attention only
    ["ssd", "swa", "ssd", "global"],      # the real hybrid
    ["ssd"],                              # SSD only
])
def test_cache_generation_matches_full_forward(pattern):
    """Cached decoding must reproduce the uncached logits for every mixer type."""
    cfg = tiny_cfg(layer_pattern=pattern, n_layers=4, window=64)
    m = MMLLM(cfg).eval().double()
    x = torch.randint(0, cfg.vocab_size, (1, 20))
    with torch.no_grad():
        full, _ = m(x, targets=x)
        _, caches = m(x[:, :-1])
        step, _ = m(x[:, -1:], caches=caches, offset=19)
    assert torch.allclose(full[:, -1], step[:, -1], atol=1e-8), \
        (full[:, -1] - step[:, -1]).abs().max()


def test_generate_runs_and_extends():
    cfg = tiny_cfg()
    m = MMLLM(cfg)
    out = m.generate(torch.randint(0, cfg.vocab_size, (2, 8)), max_new_tokens=6, top_k=10)
    assert out.shape == (2, 14)
    assert out.max() < cfg.vocab_size


def test_vision_splice_and_token_accounting():
    v = VisionConfig(enabled=True, image_size=28, patch_size=7, temporal_patch=2,
                     dim=48, depth=2, heads=4, merge_factor=2)
    cfg = tiny_cfg(vision=v)
    m = MMLLM(cfg)
    n_tok = m.vision.n_tokens(frames=2, h=28, w=28)
    assert n_tok == 4                                  # (28/7/2)^2 * (2/2)

    x = torch.randint(16, cfg.vocab_size, (1, 20))
    x[0, 5:5 + n_tok] = VISION_PLACEHOLDER
    pixels = torch.randn(1, 3, 2, 28, 28)
    logits, loss = m(x, pixels=pixels, targets=x)
    assert math.isfinite(loss.item())

    # Vision features must actually reach the output.
    with torch.no_grad():
        a, _ = m(x, pixels=pixels, targets=x)
        b, _ = m(x, pixels=pixels * 3 + 1, targets=x)
    assert not torch.allclose(a, b, atol=1e-5), "vision input had no effect on logits"


def test_video_is_more_tokens_than_image():
    v = VisionConfig(enabled=True, image_size=28, patch_size=7, temporal_patch=2,
                     dim=48, depth=2, heads=4, merge_factor=2)
    m = MMLLM(tiny_cfg(vision=v))
    assert m.vision.n_tokens(2, 28, 28) < m.vision.n_tokens(16, 28, 28)


# ------------------------------------------------------------------ budgets
def test_hybrid_beats_full_attention_on_kv_cache():
    cfg = ModelConfig(dim=512, n_layers=12, n_heads=8, n_kv_heads=2, window=4096,
                      vision=VisionConfig(enabled=False))
    m = MMLLM(cfg)
    kv = m.kv_cache_bytes(300_000)
    assert kv["total"] < kv["all_attention_equivalent"] / 4
    assert kv["ssd"] < kv["global"]        # SSD state is O(1) in sequence length


def test_ssd_state_is_constant_in_seq_len():
    m = MMLLM(tiny_cfg())
    assert m.kv_cache_bytes(1_000)["ssd"] == m.kv_cache_bytes(1_000_000)["ssd"]


# ---------------------------------------------------------------- schedules
def test_wsd_schedule_shape():
    from mmllm.config import TrainConfig
    tc = TrainConfig(lr=1e-3, max_steps=1000, warmup_frac=0.02,
                     decay_frac=0.2, min_lr_frac=0.1, schedule="wsd")
    assert lr_at(0, tc) < tc.lr
    assert math.isclose(lr_at(500, tc), tc.lr)                  # stable trunk
    assert lr_at(999, tc) < tc.lr * 0.15                        # decayed
    assert all(lr_at(i, tc) > 0 for i in range(0, 1000, 37))


# ------------------------------------------------------- analytic accounting
@pytest.mark.parametrize("kw", [
    dict(),
    dict(tie_embeddings=False),
    dict(layer_pattern=["ssd"]),
    dict(layer_pattern=["global"], n_layers=3),
    dict(mlp_ratio=2.5, dim=96, n_heads=6, n_kv_heads=3),
    dict(vision=VisionConfig(enabled=True, image_size=28, patch_size=7,
                             temporal_patch=2, dim=48, depth=3, heads=4,
                             merge_factor=2, mlp_ratio=3.0)),
])
def test_analytic_params_matches_real_model(kw):
    """planner.py prices configs too large to instantiate. If these formulas drift
    from the modules, every cost estimate is silently wrong."""
    from mmllm.model.budget import analytic_params
    cfg = tiny_cfg(**kw)
    m = MMLLM(cfg)
    pp = analytic_params(cfg)
    assert pp["total"] == m.n_params(), (pp["total"], m.n_params())
    assert pp["non_embedding"] == m.n_params(embedding=False)


def test_analytic_scales_without_allocating():
    """The 2B target config must be priceable on a machine that cannot hold it."""
    from mmllm.config import Config
    from mmllm.model.budget import analytic_params, kv_cache_bytes
    # Anchored to the package, not the working directory: the suite is run
    # from the repo root as often as from mmllm/.
    cfg = Config.load(Path(__file__).resolve().parents[1] / "configs" / "base_300k.yaml")
    pp = analytic_params(cfg.model)
    assert 1.0e9 < pp["total"] < 4.0e9
    assert pp["vision"] > 0
    kv = kv_cache_bytes(cfg.model, 300_000)
    assert kv["total"] / 2**30 < 3.0                      # hybrid fits on one GPU
    # Layer mix alone buys ~6x; stacking GQA on top of it buys ~50x vs full MHA.
    assert kv["all_attention_equivalent"] > 5 * kv["total"]
    assert kv["all_mha_equivalent"] > 40 * kv["total"]


# --------------------------------------------------------- data quality
def test_single_line_spam_is_rejected():
    """Line-uniqueness alone misses spam that lives on ONE line. Caught by the
    pipeline trace: quality_ok used to accept this and reject real code."""
    from mmllm.data.prepare import quality_ok, dup_ngram_frac
    spam = "click here | click here | click here | " * 12
    assert dup_ngram_frac(spam.split(), 5) > 0.9
    assert not quality_ok(spam, is_code=False)


def test_real_content_survives_the_repetition_filter():
    """The repetition rule must not eat legitimate text. On the real corpus the
    worst genuine document scored 0.309 against a 0.35 threshold."""
    from mmllm.data.prepare import quality_ok, dup_ngram_frac
    prose = ("The tokenizer converts raw text into integer identifiers. Each one "
             "indexes a row of the embedding matrix, which the transformer refines "
             "layer by layer until a distribution over the vocabulary can be read "
             "off the final hidden state. Training adjusts those matrices so the "
             "distribution concentrates on whatever token actually came next.")
    assert dup_ngram_frac(prose.split(), 5) < 0.05
    assert quality_ok(prose, is_code=False)

    code = ("import os\nimport sys\n\n"
            "def tokenize(text: str) -> list[int]:\n"
            "    return [vocab.get(t, UNK) for t in split(text)]\n\n"
            "def detokenize(ids: list[int]) -> str:\n"
            "    return ''.join(inv_vocab[i] for i in ids)\n\n"
            "class Encoder:\n"
            "    def __init__(self, merges, specials):\n"
            "        self.merges, self.specials = merges, specials\n")
    assert quality_ok(code, is_code=True)


def test_dup_ngram_frac_edges():
    from mmllm.data.prepare import dup_ngram_frac
    assert dup_ngram_frac([], 5) == 0.0
    assert dup_ngram_frac(["a"] * 3, 5) == 0.0          # too short to judge
    assert dup_ngram_frac(["a"] * 40, 5) > 0.9          # maximally repetitive
    assert dup_ngram_frac([str(i) for i in range(40)], 5) == 0.0


# ------------------------------------------------------ streaming ingest
def test_binwriter_roundtrip_across_flushes(tmp_path):
    """The streaming writer must reproduce exactly what was appended, in order,
    regardless of where its buffer happened to flush. A corpus larger than RAM is
    the whole reason this class exists, so an off-by-one here silently corrupts
    every token after the first flush."""
    import numpy as np
    from mmllm.data.hf_ingest import BinWriter

    path = tmp_path / "t.bin"
    # Tiny buffer forces many flushes mid-stream.
    w = BinWriter(str(path), np.uint16, buffer_tokens=50)
    chunks = [np.arange(i, i + 7, dtype=np.uint16) for i in range(0, 300, 7)]
    for c in chunks:
        w.add(c)
    total = w.close()

    expected = np.concatenate(chunks)
    got = np.fromfile(path, dtype=np.uint16)
    assert total == expected.size == got.size
    assert np.array_equal(got, expected)


def test_binwriter_truncates_previous_run(tmp_path):
    """Re-ingesting must not append to a stale file — that would silently double
    the corpus and mix two tokenizers' ids."""
    import numpy as np
    from mmllm.data.hf_ingest import BinWriter

    path = tmp_path / "t.bin"
    w = BinWriter(str(path), np.uint16); w.add(np.arange(10, dtype=np.uint16)); w.close()
    w2 = BinWriter(str(path), np.uint16); w2.add(np.arange(4, dtype=np.uint16))
    assert w2.close() == 4
    assert np.fromfile(path, dtype=np.uint16).size == 4


def test_binwriter_handles_empty(tmp_path):
    import numpy as np
    from mmllm.data.hf_ingest import BinWriter
    w = BinWriter(str(tmp_path / "e.bin"), np.uint16)
    assert w.close() == 0
