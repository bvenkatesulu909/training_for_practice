"""Analytic parameter, FLOP and KV-cache accounting.

Everything here is computed from the config alone — no model is instantiated.
That is not an optimisation: `base_300k` is ~2B params and would need ~8 GiB of
RAM just to be counted, which the machine you plan on may not have. Planning must
never require the hardware you are planning for.

`test_analytic_params_matches_real_model` pins these formulas against an actually
constructed model, so they cannot drift from the modules.
"""
from __future__ import annotations


def swiglu_hidden(dim: int, ratio: float) -> int:
    """Shared by SwiGLU and the analytic count so the two cannot disagree."""
    h = int(2 * ratio * dim / 3)
    return 64 * ((h + 63) // 64)


def attention_params(cfg) -> int:
    nh, nkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    return (cfg.dim * nh * hd            # wq
            + cfg.dim * nkv * hd * 2     # wk, wv
            + nh * hd * cfg.dim          # wo
            + 2 * hd)                    # q_norm, k_norm


def ssd_params(cfg) -> int:
    d_inner = cfg.ssd_heads * cfg.ssd_head_dim
    gn = cfg.ssd_groups * cfg.ssd_state
    conv_dim = d_inner + 2 * gn
    return (cfg.dim * (2 * d_inner + 2 * gn + cfg.ssd_heads)   # in_proj
            + conv_dim * cfg.ssd_conv + conv_dim               # depthwise conv + bias
            + 3 * cfg.ssd_heads                                # A_log, D, dt_bias
            + d_inner                                          # norm
            + d_inner * cfg.dim)                               # out_proj


def vision_params(cfg) -> int:
    v = cfg.vision
    if not v.enabled:
        return 0
    hid = swiglu_hidden(v.dim, v.mlp_ratio)
    per_block = (2 * v.dim                            # n1, n2
                 + 3 * v.dim * v.dim + 3 * v.dim      # qkv + bias
                 + v.dim * v.dim + v.dim              # proj + bias
                 + 3 * v.dim * hid)                   # SwiGLU
    patch = 3 * v.dim * v.temporal_patch * v.patch_size ** 2 + v.dim
    m = v.merge_factor
    projector = (v.dim * m * m * cfg.dim + cfg.dim + cfg.dim * cfg.dim + cfg.dim)
    return patch + v.depth * per_block + v.dim + projector


def analytic_params(cfg) -> dict:
    """Parameter breakdown for a ModelConfig, without building anything."""
    hid = swiglu_hidden(cfg.dim, cfg.mlp_ratio)
    mlp = 3 * cfg.dim * hid
    per_kind = {"ssd": ssd_params(cfg), "swa": attention_params(cfg),
                "global": attention_params(cfg)}

    layers = 0
    counts = {"ssd": 0, "swa": 0, "global": 0}
    for k in cfg.kinds():
        layers += per_kind[k] + mlp + 2 * cfg.dim   # + n1, n2
        counts[k] += 1

    embed = cfg.vocab_size * cfg.dim
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.dim
    vis = vision_params(cfg)
    total = embed + head + layers + cfg.dim + vis

    return {"total": total, "embedding": embed + head, "layers": layers,
            "vision": vis, "final_norm": cfg.dim,
            "non_embedding": total - embed - head, "layer_counts": counts}


def kv_cache_bytes(cfg, seq_len: int, bytes_per_elem: int = 2) -> dict:
    """Inference KV-cache footprint — the real gate on long context."""
    per_tok = 2 * cfg.n_kv_heads * cfg.head_dim * bytes_per_elem
    kinds = cfg.kinds()
    g = kinds.count("global") * per_tok * seq_len
    s = kinds.count("swa") * per_tok * min(seq_len, cfg.window)
    # SSD carries a fixed-size state, independent of seq_len.
    ssd = kinds.count("ssd") * cfg.ssd_heads * cfg.ssd_head_dim * cfg.ssd_state * 4
    # Two honest baselines. The first isolates the *layer-mix* win alone; the
    # second is what a conventional full-MHA model of this shape would cost.
    full_gqa = cfg.n_layers * per_tok * seq_len
    full_mha = cfg.n_layers * 2 * cfg.n_heads * cfg.head_dim * bytes_per_elem * seq_len
    return {"global": g, "swa": s, "ssd": ssd, "total": g + s + ssd,
            "all_attention_equivalent": full_gqa,
            "all_mha_equivalent": full_mha}


def training_flops(cfg, tokens: float, seq_len: int) -> dict:
    """Training FLOPs, attention included.

    The usual 6ND shortcut drops the attention term. That is fine at 8k, where it
    is ~10% — and badly wrong at 300k, where the quadratic term dominates and
    would make a context-extension run look ~10x cheaper than it is.

    Per attention layer per sequence: ~12 * L^2 * d (QK^T and AV, fwd+bwd).
    Sliding-window layers see min(L, window) instead of L, and SSD layers are
    linear, so they contribute nothing quadratic at all.
    """
    N = analytic_params(cfg)["total"]
    dense = 6 * N * tokens

    kinds = cfg.kinds()
    eff_len = (kinds.count("global") * seq_len
               + kinds.count("swa") * min(seq_len, cfg.window))
    attn = 12 * eff_len * cfg.dim * tokens

    return {"dense": dense, "attention": attn, "total": dense + attn,
            "attention_frac": attn / (dense + attn)}
