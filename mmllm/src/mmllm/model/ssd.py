"""State-space-duality (Mamba-2) mixer, pure PyTorch.

This is the layer that makes a 300k context affordable. Attention costs
O(L^2) compute and carries an O(L) KV cache; this costs O(L) compute and
carries a *constant* (H, P, N) state regardless of sequence length.

Two implementations are provided and are numerically equivalent:
  - `ssd_reference`: the literal recurrence, O(L) python loop. Slow, obviously
    correct, used as the test oracle.
  - `ssd_chunked`:   the chunkwise-parallel form. Loops only over L/Q chunks and
    does the rest in batched matmuls. This is what trains.

Recurrence being computed, per head h:
    h_t = exp(dt_t * A_h) * h_{t-1} + dt_t * (B_t x_t^T)
    y_t = C_t . h_t + D_h * x_t
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _expand_groups(t: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, L, G, N) -> (B, L, H, N) by repeating each group H//G times."""
    g = t.shape[2]
    if g == n_heads:
        return t
    return t.repeat_interleave(n_heads // g, dim=2)


def ssd_reference(x, dt, A, B, C, D):
    """Literal sequential scan. x:(b,l,h,p) dt:(b,l,h) A:(h,) B/C:(b,l,g,n) D:(h,)"""
    b, l, h, p = x.shape
    B = _expand_groups(B, h)
    C = _expand_groups(C, h)
    n = B.shape[-1]
    state = x.new_zeros(b, h, p, n)
    outs = []
    for t in range(l):
        dA = torch.exp(dt[:, t] * A)                      # (b, h)
        # state <- dA * state + dt * (x_t outer B_t)
        state = state * dA[..., None, None] + (
            dt[:, t][..., None, None] * x[:, t][..., :, None] * B[:, t][..., None, :]
        )
        outs.append(torch.einsum("bhpn,bhn->bhp", state, C[:, t]))
    y = torch.stack(outs, dim=1)                          # (b, l, h, p)
    return y + x * D[None, None, :, None]


def ssd_chunked(x, dt, A, B, C, D, chunk: int = 128, return_state: bool = False):
    """Chunkwise-parallel SSD. Same signature and result as `ssd_reference`.

    With `return_state`, also returns the final (b, h, p, n) recurrent state so
    generation can continue from it instead of re-scanning the prefix.
    """
    b, l, h, p = x.shape
    B = _expand_groups(B, h)
    C = _expand_groups(C, h)
    n = B.shape[-1]

    pad = (chunk - l % chunk) % chunk
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
    L = x.shape[1]
    nc = L // chunk

    rs = lambda t, *tail: t.reshape(b, nc, chunk, *tail)
    xc, dtc, Bc, Cc = rs(x, h, p), rs(dt, h), rs(B, h, n), rs(C, h, n)

    # Cumulative log-decay inside each chunk. A < 0 and dt > 0, so every
    # exponent below is <= 0 -> exp() is bounded by 1 and cannot overflow.
    a = dtc * A                                            # (b,nc,q,h)
    cs = torch.cumsum(a, dim=2)
    csh = cs.permute(0, 1, 3, 2)                           # (b,nc,h,q)

    # --- intra-chunk: strictly causal, fully parallel ---
    diff = csh.unsqueeze(-1) - csh.unsqueeze(-2)           # [...,i,j] = cs_i - cs_j
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=x.device).tril()
    decay = torch.where(causal, torch.exp(diff), torch.zeros_like(diff))
    CB = torch.einsum("bcihn,bcjhn->bchij", Cc, Bc)
    xdt = xc * dtc.unsqueeze(-1)                           # fold dt into the value
    y = torch.einsum("bchij,bcjhp->bcihp", CB * decay, xdt)

    # --- inter-chunk: sequential over L/Q chunks only ---
    total = cs[:, :, -1, :]                                # (b,nc,h) log-decay per chunk
    w = torch.exp(total.unsqueeze(2) - cs)                 # (b,nc,q,h)
    contrib = torch.einsum("bcjh,bcjhp,bcjhn->bchpn", w, xdt, Bc)

    states = x.new_zeros(b, nc, h, p, n)
    s = x.new_zeros(b, h, p, n)
    dec = torch.exp(total)                                 # (b,nc,h)
    for c in range(nc):
        states[:, c] = s
        s = s * dec[:, c][..., None, None] + contrib[:, c]

    y = y + torch.einsum("bcihn,bchpn->bcihp", Cc, states) * torch.exp(cs).unsqueeze(-1)

    y = y.reshape(b, L, h, p)[:, :l]
    out = y + x[:, :l] * D[None, None, :, None]
    # Padded positions carry dt=0, so they leave `s` untouched — the final state
    # is exactly the state after token l-1.
    return (out, s) if return_state else out


def ssd_step(x_t, dt_t, A, B_t, C_t, D, state):
    """Single-token SSD update for autoregressive decoding.

    x_t:(b,h,p) dt_t:(b,h) B_t/C_t:(b,g,n) state:(b,h,p,n) -> (y_t, new_state).
    Cost and memory are constant in sequence length — this is why the model can
    decode at 300k context without an ever-growing cache.
    """
    h = x_t.shape[1]
    B_t = _expand_groups(B_t.unsqueeze(1), h).squeeze(1)
    C_t = _expand_groups(C_t.unsqueeze(1), h).squeeze(1)
    dA = torch.exp(dt_t * A)                                   # (b,h)
    state = state * dA[..., None, None] + (
        dt_t[..., None, None] * x_t[..., :, None] * B_t[..., None, :]
    )
    y = torch.einsum("bhpn,bhn->bhp", state, C_t) + x_t * D[None, :, None]
    return y, state


class SSDMixer(nn.Module):
    """Mamba-2 style block: gated conv -> SSD scan -> gated output projection."""

    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.ssd_heads
        self.p = cfg.ssd_head_dim
        self.n = cfg.ssd_state
        self.g = cfg.ssd_groups
        self.chunk = cfg.ssd_chunk
        d_inner = self.h * self.p

        # z (gate) | x (values) | B | C | dt
        self.in_proj = nn.Linear(cfg.dim, 2 * d_inner + 2 * self.g * self.n + self.h, bias=False)

        conv_dim = d_inner + 2 * self.g * self.n
        self.conv = nn.Conv1d(conv_dim, conv_dim, cfg.ssd_conv,
                              groups=conv_dim, padding=cfg.ssd_conv - 1, bias=True)

        # A initialised so heads span a range of timescales (short -> long memory).
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.h + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.h))
        # dt bias in softplus^-1 space -> initial dt spread over [0.001, 0.1]
        dt = torch.exp(torch.rand(self.h) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.norm = nn.RMSNorm(d_inner, eps=cfg.norm_eps)
        self.out_proj = nn.Linear(d_inner, cfg.dim, bias=False)

    def forward(self, u: torch.Tensor, cache=None):
        """cache = (conv_window, ssm_state); returns (y, new_cache).

        Both branches below must agree exactly — `test_ssd_prefill_then_step`
        pins that, because a mismatch shows up only as degraded generation.
        """
        b, l, _ = u.shape
        d_inner, gn = self.h * self.p, self.g * self.n
        k = self.conv.kernel_size[0]

        z, xBC, dt = self.in_proj(u).split([d_inner, d_inner + 2 * gn, self.h], dim=-1)
        A = -torch.exp(self.A_log)                         # strictly negative
        dt = F.softplus(dt + self.dt_bias)                 # strictly positive

        if cache is not None and l == 1:
            conv_win, state = cache
            conv_win = torch.roll(conv_win, -1, dims=-1)
            conv_win[:, :, -1] = xBC[:, 0]
            w = self.conv.weight.squeeze(1)                # (conv_dim, k) depthwise
            act = F.silu(torch.einsum("bdk,dk->bd", conv_win, w) + self.conv.bias)
            x, B, C = act.split([d_inner, gn, gn], dim=-1)
            y, state = ssd_step(x.view(b, self.h, self.p), dt[:, 0], A,
                                B.view(b, self.g, self.n), C.view(b, self.g, self.n),
                                self.D, state)
            y = y.reshape(b, 1, d_inner)
            new_cache = (conv_win, state)
        else:
            raw = xBC.transpose(1, 2)                      # (b, conv_dim, l)
            act = F.silu(self.conv(raw)[..., :l].transpose(1, 2))
            x, B, C = act.split([d_inner, gn, gn], dim=-1)
            y, state = ssd_chunked(x.view(b, l, self.h, self.p), dt, A,
                                   B.view(b, l, self.g, self.n),
                                   C.view(b, l, self.g, self.n),
                                   self.D, chunk=self.chunk, return_state=True)
            y = y.reshape(b, l, d_inner)
            conv_win = F.pad(raw, (max(k - l, 0), 0))[:, :, -k:]  # left-pad if l < k
            new_cache = (conv_win.contiguous(), state)

        y = self.norm(y) * F.silu(z)
        return self.out_proj(y), new_cache
