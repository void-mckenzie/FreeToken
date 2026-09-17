"""Reference-PyTorch ops for the DFlash2 speculative-draft drafter.

Milestone-1 (plan T4/T5/T6/T7/T8) is *correct-by-construction* reference PyTorch on top of
FreeToken's existing primitives: no new CUDA/Triton kernel. Every op here composes plain PyTorch
(matmul, rsqrt, softmax, einsum/reshape, argsort/multinomial) so it runs unmodified on the real box
(RTX 5090) and can be validated against an FP64 oracle before any Triton conversion (T11/T12/T13).

Ops by task:
* T5 ``dynamic_grouped_conv_prepare``/``_finish``; T6 ``draft_attention``/``noncausal_sliding_attention``/
  ``full_attention``; plus the shared norm/rope helpers (``plain_rmsnorm``, ``rope_1d_split_half``,
  ``head_rmsnorm_rope``, ``repeat_kv``).
* T4 ``materialize_context_kv``: target residual features -> per-layer context K/V (K bf16,
  V ``FP16_RNE(BF16(v_raw))``). Ring storage lives in ``kvcache/dflash2_ring_pool.py``.
* T7 ``candidate_selector``: top-16 shortlist + conditional left-to-right walk (greedy first).
* T8 ``accept_sparse_drafts``: verify/accept (greedy ship-first; temperature > 0 documented).

Source of truth for the math: ninfer ``docs/maintainer/dflash.md`` ("DFlash2 dynamic
convolution", "Query block and attention masks", "DFlash backbone and proposal", "Target
conditioning and context") and the op contracts ``dynamic_grouped_conv.h``,
``sliding_window_attention.h`` / ``softmax_attention.h``, and ``rmsnorm_rope.h`` / ``rope.h``.

Precision choices (documented for the T5/T6 oracles):
* RMSNorm / RoPE: reduction + rotation in fp32, cast back to the input (bf16) dtype -- matches
  the FreeToken triton ``rmsnorm`` ("reductions accumulate in fp32; output cast back") and the
  ninfer "no observable bf16 materialization of n" norm-then-rope contract.
* Dynamic grouped conv: the delta projection and the two-sided conv run in fp32 (approximating
  the FP64 oracle from the represented bf16 inputs); only the observable outputs (``prepared``
  and the ``finish`` residual contribution) are cast to bf16. The output-side delta
  (``finish_delta``) is computed ONCE in ``prepare`` from the normalized input and *reused* in
  ``finish`` -- it is NOT re-projected from the branch output.
* Non-causal sliding / full attention: scores, stable softmax and the value reduction are done
  in fp32 (stable softmax over exactly the visible set), the output cast to the q dtype (bf16).
  The visible-set rule is applied as an explicit additive float mask.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def plain_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Plain RMSNorm: ``w * x / sqrt(mean(x^2, over last dim) + eps)``.

    This is the plain (NOT Gemma ``(1+w)``) form used everywhere in DFlash2 (input_norm,
    post_attention_norm, context_norm, final_norm, and the full-head q/k norm). The reduction is
    done in fp32 and the result cast back to ``x``'s dtype, matching FreeToken's triton ``rmsnorm``
    and ``ninfer`` ``plain_rmsnorm(x,w) = w * x / sqrt(mean(x^2) + eps)``.
    """
    x_f = x.float()
    inv = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
    return (x_f * inv * weight.float()).to(x.dtype)


def rope_1d_split_half(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """1-D split-half (rotate-half / NeoX) RoPE over the last dim, full-head.

    ``x`` is ``(..., D)`` with even ``D``; ``positions`` is broadcastable to ``x.shape[:-1]``
    (e.g. ``(N,)`` for a ``(N, heads, D)`` tensor). For pair ``i in [0, D/2)``::

        angle(i) = position * theta ** (-2*i/D)
        out[i]      = x[i]    * cos(angle(i)) - x[i+D/2] * sin(angle(i))
        out[i + D/2] = x[i+D/2] * cos(angle(i)) + x[i]    * sin(angle(i))

    This is the full-head 1-D RoPE the draft uses (theta = 1e7), NOT the target's partial
    3-axis MRoPE. Rotation is in fp32; output cast to ``x``'s dtype.
    """
    D = x.shape[-1]
    assert D % 2 == 0, f"rope_1d_split_half needs an even head dim, got {D}"
    half = D // 2
    inv_freq = theta ** (-2.0 * torch.arange(half, device=x.device, dtype=torch.float32) / D)  # (half,)
    # positions is (N,) for x of shape (N, *, D): one angle set per leading row, shared across the
    # intermediate (head) dims. Shape positions to (N, 1, 1, ..., 1) (x.ndim-1 dims) so it broadcasts
    # over the intermediate dims and inv_freq (last dim).
    n_ones = x.ndim - 2  # dims between the position dim (N) and the last dim (D)
    pos = positions.to(torch.float32).reshape((positions.shape[0],) + (1,) * (n_ones + 1))
    angle = pos * inv_freq  # (N, 1, ..., half)
    cos, sin = angle.cos(), angle.sin()
    x_f = x.float()
    lo, hi = x_f[..., :half], x_f[..., half:]
    out = torch.cat((lo * cos - hi * sin, hi * cos + lo * sin), dim=-1)
    return out.to(x.dtype)


def head_rmsnorm_rope(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
    eps: float,
) -> torch.Tensor:
    """Full-head per-head plain RMSNorm then 1-D split-half RoPE (``ninfer rmsnorm_rope``).

    ``x`` is ``(N, heads, D)``; ``norm_weight`` is ``(D,)``; ``positions`` is ``(N,)`` absolute.
    Each head is normalized independently over its ``D`` dims (full-head norm, the same weight
    vector applied per head), then rotated. Normalization + rotation are one semantic op -- there
    is no observable bf16 materialization of the normalized tensor; the whole thing is computed in
    fp32 and cast to ``x``'s dtype at the end.
    """
    D = x.shape[-1]
    x_f = x.float()
    inv = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)  # (N, heads, 1)
    n = x_f * inv * norm_weight.float()[None, None, :]  # (N, heads, D) fp32
    half = D // 2
    inv_freq = theta ** (-2.0 * torch.arange(half, device=x.device, dtype=torch.float32) / D)
    angle = positions.to(torch.float32).reshape(-1, 1, 1) * inv_freq[None, None, :]  # (N, 1, half)
    cos, sin = angle.cos(), angle.sin()
    lo, hi = n[..., :half], n[..., half:]
    out = torch.cat((lo * cos - hi * sin, hi * cos + lo * sin), dim=-1)  # (N, heads, D) fp32
    return out.to(x.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Grouped-query expansion: ``(T, n_kv, D)`` -> ``(T, n_kv * n_rep, D)``.

    Query head ``h`` reads KV head ``h // n_rep``; ``expand`` tiles each KV head ``n_rep`` times
    (KV head ``j`` becomes query heads ``[j*n_rep, (j+1)*n_rep)``). Free (stride 0) until the
    downstream matmul forces a copy.
    """
    if n_rep == 1:
        return x
    T, n_kv, D = x.shape
    return x[:, :, None, :].expand(T, n_kv, n_rep, D).reshape(T, n_kv * n_rep, D)


def _prev(x: torch.Tensor) -> torch.Tensor:
    """Shift one step along the position (leading) dim with a zero at the origin.

    ``out[0] = 0`` and ``out[i] = x[i-1]`` for ``i >= 1`` -- encodes the "position 0 has no
    preceding tap" rule of the dynamic grouped conv.
    """
    out = torch.zeros_like(x)
    if x.shape[0] > 1:
        out[1:] = x[:-1]
    return out


def dynamic_grouped_conv_prepare(
    base_kernel: torch.Tensor,
    kernel_weight: torch.Tensor,
    n: torch.Tensor,
    *,
    taps: int,
    groups: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Input-side dynamic grouped convolution of a normalized branch input (``prepare``).

    ``base_kernel`` is the runtime view ``[H, taps, side]`` (channel, tap, side); ``kernel_weight``
    is the ``[2*taps*groups, H]`` projection whose rows are indexed ``row(s, t, g) = (s*taps + t)
    * groups + g``; ``n`` is the normalized branch input ``[W, H]`` (positions, channels). For
    channel ``h = group_size*g + j``::

        d[s, t, g, i]   = sum_h kernel_weight[row(s, t, g), h] * n[h, i]
        prepared[h, i]  = (base[h, 0, 0] + d[0, 0, g, i]) * n[h, i]
                         + 1{i>0} * (base[h, 1, 0] + d[0, 1, g, i]) * n[h, i-1]

    Returns ``(prepared, finish_delta)`` where ``prepared`` is bf16 ``[W, H]`` and
    ``finish_delta`` is the *output-side* delta ``d[1, :, :, :]`` (fp32 ``[taps, groups, W]``),
    computed once from the normalized input and reused by ``finish`` (NOT re-projected). The conv
    arithmetic runs in fp32 for oracle parity; ``prepared`` is cast to ``n``'s dtype.
    """
    W, H = n.shape
    n_f = n.float()
    base = base_kernel.float()  # (H, taps, 2)
    base_g = base.reshape(groups, group_size, taps, 2)  # (group, j, tap, side)
    # delta projection: d[s, t, g, i] = sum_h kernel_weight[row(s, t, g), h] * n[h, i]
    proj = n_f @ kernel_weight.float().T  # (W, 2*taps*groups)
    proj = proj.reshape(W, 2, taps, groups)  # (W, side, tap, group)
    delta = proj.permute(1, 2, 3, 0)  # (side, tap, group, W)
    n_g = n_f.reshape(W, groups, group_size)  # (W, group, j)
    base_in_t0 = base_g[:, :, 0, 0]  # (group, j): base[channel, tap0, side0] (input, current)
    base_in_t1 = base_g[:, :, 1, 0]  # (group, j): base[channel, tap1, side0] (input, previous)
    d_in_t0 = delta[0, 0].transpose(0, 1)  # (W, group)
    d_in_t1 = delta[0, 1].transpose(0, 1)  # (W, group)
    prev = _prev(n_g)  # (W, group, j), [0] = 0
    # base_in_t0[None] (1, group, j) + d_in_t0[:, :, None] (W, group, 1) -> (W, group, j)
    cur = (base_in_t0[None] + d_in_t0[:, :, None]) * n_g
    prv = (base_in_t1[None] + d_in_t1[:, :, None]) * prev
    prepared = (cur + prv).reshape(W, H).to(n.dtype)
    finish_delta = delta[1]  # (taps, group, W) fp32 -- output-side delta, kept from prepare
    return prepared, finish_delta


def dynamic_grouped_conv_finish(
    base_kernel: torch.Tensor,
    z: torch.Tensor,
    finish_delta: torch.Tensor,
    *,
    taps: int,
    groups: int,
    group_size: int,
) -> torch.Tensor:
    """Output-side dynamic grouped convolution of a branch output ``z`` (``finish``).

    Applies the output side (``side = 1``) of the dynamic conv to the branch output ``z``
    (``[W, H]``) using the ``finish_delta`` produced by ``prepare`` (``[taps, groups, W]``,
    re-projected NO further). For channel ``h = group_size*g + j``::

        out[h, i] = (base[h, 0, 1] + finish_delta[0, g, i]) * z[h, i]
                   + 1{i>0} * (base[h, 1, 1] + finish_delta[1, g, i]) * z[h, i-1]

    The caller adds the result to the residual stream (``x + out``), matching the ninfer fused
    ``linear_dynamic_grouped_conv_add`` (which does the residual add in place). Arithmetic in fp32;
    output cast to ``z``'s dtype.
    """
    W, H = z.shape
    z_f = z.float()
    base = base_kernel.float()  # (H, taps, 2)
    base_g = base.reshape(groups, group_size, taps, 2)  # (group, j, tap, side)
    base_out_t0 = base_g[:, :, 0, 1]  # (group, j): base[channel, tap0, side1] (output, current)
    base_out_t1 = base_g[:, :, 1, 1]  # (group, j): base[channel, tap1, side1] (output, previous)
    fd_t0 = finish_delta[0].transpose(0, 1)  # (W, group)
    fd_t1 = finish_delta[1].transpose(0, 1)  # (W, group)
    z_g = z_f.reshape(W, groups, group_size)  # (W, group, j)
    prev = _prev(z_g)  # (W, group, j), [0] = 0
    # base_out_t0[None] (1, group, j) + fd_t0[:, :, None] (W, group, 1) -> (W, group, j)
    cur = (base_out_t0[None] + fd_t0[:, :, None]) * z_g
    prv = (base_out_t1[None] + fd_t1[:, :, None]) * prev
    return (cur + prv).reshape(W, H).to(z.dtype)


def draft_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    scale: float,
    num_q: int,
    n_kv: int,
    window: Optional[int] = None,
) -> torch.Tensor:
    """Non-causal grouped-query attention over context + query-block K/V.

    ``q`` is ``(W, num_q, D)``; ``k``/``v`` are ``(T, n_kv, D)`` with ``T = S_live + W`` where the
    first ``S_live`` keys are the live context rows and the last ``W`` are the query-block rows
    (the caller concatenates ``concat(context, query)`` along the position axis). ``query_positions``
    is ``(W,)`` and ``key_positions`` is ``(T,)`` absolute. ``window=None`` gives full (attend-all)
    attention; an integer ``window`` (the sliding-window scalar ``S``) admits a key iff
    ``abs(p_key - p_query) < window`` (distance ``S-1`` included, ``S`` excluded) -- and every
    query row additionally sees every query-block key (the last ``W`` keys) regardless of window.

    Scores, the stable softmax (exactly over the visible set, additive ``-inf`` mask) and the value
    reduction run in fp32; the output is cast to ``q``'s dtype.
    """
    W, num_q_, D = q.shape
    assert num_q_ == num_q, f"q has {num_q_} heads, expected {num_q}"
    T = k.shape[0]
    n_rep = num_q // n_kv
    k_e = repeat_kv(k, n_rep)  # (T, num_q, D)
    v_e = repeat_kv(v, n_rep)  # (T, num_q, D)
    qf, kf, vf = q.float(), k_e.float(), v_e.float()
    scores = torch.einsum("whd,thd->wht", qf, kf) * scale  # (W, num_q, T) fp32

    if window is None:
        allowed = torch.ones(W, T, dtype=torch.bool, device=q.device)
    else:
        diff = (key_positions[:, None] - query_positions[None, :]).abs()  # (T, W) int
        allowed = (diff < window).t().contiguous()  # (W, T)
        n_ctx = T - W  # the last W keys are the query-block rows: always visible
        allowed[:, n_ctx:] = True

    add = torch.where(allowed, 0.0, float("-inf"))  # (W, T) fp32 additive mask
    scores = scores + add[None, :, :]
    probs = torch.softmax(scores, dim=-1)  # (W, num_q, T) fp32, exactly over the visible set
    out = torch.einsum("wht,thd->whd", probs, vf)  # (W, num_q, D) fp32
    return out.to(q.dtype)


def noncausal_sliding_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    scale: float,
    num_q: int,
    n_kv: int,
    window: int,
) -> torch.Tensor:
    """Symmetric non-causal sliding-window grouped-query attention (local draft layers)."""
    return draft_attention(
        q, k, v, query_positions, key_positions, scale, num_q, n_kv, window=window
    )


def full_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    scale: float,
    num_q: int,
    n_kv: int,
) -> torch.Tensor:
    """Full (attend-all, non-causal) grouped-query attention (full draft layers)."""
    return draft_attention(
        q, k, v, query_positions, key_positions, scale, num_q, n_kv, window=None
    )


# ---------------------------------------------------------------------------
# T4 -- context materializer (target conditioning -> per-layer context K/V)
# ---------------------------------------------------------------------------


def materialize_context_kv(
    features: torch.Tensor,
    positions: torch.Tensor,
    feature_weight: torch.Tensor,
    context_norm_weight: torch.Tensor,
    layer_kv: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    theta: float,
    eps: float,
    n_kv: int,
    head_dim: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Materialize a batch of target residual features into per-layer context K/V (plan T4).

    This is the pure reference form of the DFlash2 fused context materializer
    (``context_kv_materialize.h``). Given the target's per-position residual features ``s_t``
    (the concatenated residual outputs at ``target_layer_ids``, recorded order) and their absolute
    positions, it computes the common normalized context vector and projects it to each draft
    layer's context K/V. Context prefill does NOT run the prompt through the draft residual/MLP
    stack.

    Math (ninfer ``dflash.md`` "Target conditioning and context"):

        c_t = plain_rmsnorm(W_feature @ s_t, context_norm)           # bias-free proj
        per layer l:
            k_raw = W_context_key^l c_t
            k_ctx = rope_1d(plain_head_rmsnorm(k_raw, key_norm^l), position=t)  # full-head 1-D rope
            v_ctx = W_context_value^l c_t

    Args:
        features: ``[n, num_taps * H]`` bf16 -- ``s_t`` for ``n`` positions (row ``i`` is at
            ``positions[i]``).
        positions: ``[n,]`` integer -- the absolute Text cache positions (row order).
        feature_weight: ``[H, num_taps * H]`` -- the bias-free feature projection weight
            (``DFlash2Drafter.feature_projection.weight``).
        context_norm_weight: ``[H]`` -- the context RMSNorm weight.
        layer_kv: a sequence of ``num_layers`` triples ``(context_key_w, context_value_w,
            key_norm_w)`` where ``context_key_w``/``context_value_w`` are ``[n_kv * D, H]`` and
            ``key_norm_w`` is ``[D]``.
        theta: the 1-D split-half RoPE base (``cfg.rope_theta`` = 1e7).
        eps: the plain RMSNorm epsilon (``cfg.rms_norm_eps`` = 1e-6).
        n_kv: the number of KV heads (8).
        head_dim: the head dimension ``D`` (128).

    Returns:
        ``(context_k, context_v)`` -- lists of ``num_layers`` tensors. ``context_k[i]`` is
        ``[n, n_kv, D]`` **BF16** and ``context_v[i]`` is ``[n, n_kv, D]`` **FP16**, where the V is
        the parity-critical double-round ``FP16_RNE(BF16(v_raw))`` (round ``v_raw`` to bf16 first,
        then to fp16 -- never a single fp16 round, and never stored as bf16).
    """
    n = features.shape[0]
    # common normalized context vector (bf16, matching the drafter's LinearReplicated path)
    projected = F.linear(features, feature_weight)  # [n, H] bf16
    c = plain_rmsnorm(projected, context_norm_weight, eps)  # [n, H] bf16
    key_width = n_kv * head_dim
    context_k: List[torch.Tensor] = []
    context_v: List[torch.Tensor] = []
    for ck_w, cv_w, kn_w in layer_kv:
        k_raw = F.linear(c, ck_w).view(n, n_kv, head_dim)  # [n, n_kv, D] bf16
        k_ctx = head_rmsnorm_rope(k_raw, kn_w, positions, theta, eps)  # [n, n_kv, D] bf16
        v_raw = F.linear(c, cv_w).view(n, n_kv, head_dim)  # [n, n_kv, D] bf16
        # double-round: represented BF16(v_raw) then FP16_RNE. v_raw is already bf16 (F.linear of
        # bf16 inputs) so the bf16 cast is a no-op; it is kept explicit so an fp32 v_raw would still
        # be rounded bf16-then-fp16 (the parity matters -- do NOT store bf16 or fp16-of-fp32).
        v_ctx = v_raw.to(torch.bfloat16).to(torch.float16)
        context_k.append(k_ctx.to(torch.bfloat16))
        context_v.append(v_ctx)
    return context_k, context_v


# ---------------------------------------------------------------------------
# Deterministic RNG for the stochastic (temperature > 0) paths (T7/T8)
# ---------------------------------------------------------------------------

# Distinct "purpose" tags for each sampling domain. FreeToken has no ninfer-equivalent counter
# PRNG, so these constants plus the (request_seed, logical_position) key give a *deterministic*
# seed (the same key reproduces the same draw across runs). Bit-parity with ninfer requires porting
# its exact counter PRNG (plan O3) -- greedy (temperature 0) needs no RNG and is exact regardless.
DFLASH2_PROPOSAL_PURPOSE = 0x44463250  # "DF2P": candidate-selector stochastic walk
DFLASH2_ACCEPT_PURPOSE = 0x44463241  # "DF2A": per-draft accept/reject draw
DFLASH2_CORRECTION_PURPOSE = 0x44463243  # "DF2C": first-rejection correction draw
DFLASH2_BONUS_PURPOSE = 0x44463242  # "DF2B": all-accept bonus draw

_MASK64 = (1 << 64) - 1


def _mix64(seed: int, position: int, purpose: int) -> int:
    """Deterministic 64-bit seed from (request_seed, logical_position, purpose) (SplitMix64-style).

    A stable mixing function so the stochastic paths are reproducible for a fixed
    ``(seed, position, purpose)`` key. NOT the ninfer counter PRNG (O3).
    """
    h = (int(seed) ^ int(purpose)) & _MASK64
    h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & _MASK64
    h ^= h >> 31
    h = (h + (int(position) * 0x9E3779B97F4A7C15)) & _MASK64
    h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (h ^ (h >> 31)) & _MASK64


def _make_generator(hidden: torch.Tensor) -> torch.Generator:
    """A torch.Generator on ``hidden``'s device (CUDA-aware); reseeded per draw for reproducibility."""
    dev = hidden.device
    return torch.Generator(device=dev) if dev.type == "cuda" else torch.Generator()


# ---------------------------------------------------------------------------
# T7 -- candidate selector (top-16 shortlist + conditional left-to-right walk)
# ---------------------------------------------------------------------------


def candidate_selector(
    hidden: torch.Tensor,
    proposal_head: Callable[[torch.Tensor], torch.Tensor],
    hidden_projection_w: torch.Tensor,
    predecessor_codebook: torch.Tensor,
    successor_codebook: torch.Tensor,
    anchor: int,
    base_position: int,
    temperature: float,
    seed: int,
    top_k: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DFlash2 candidate selector: per-mask top-16 shortlist + conditional walk (plan T7).

    For each mask hidden ``h_i`` the proposal head gives a stable **top-16** over the raw public-vocab
    logits (NO temperature/penalty/top-k/top-p): unary scores ``u_i [top_k]`` (fp32) and global token
    ids ``C_i [top_k]``. A **left-to-right conditional walk** (predecessor = the actually selected
    token; the anchor for ``i == 0``) then picks the proposal at each position.

    Math (``candidate_selector.h`` / ``dflash.md`` "DFlash2 candidate path"):

        g_i = W_hidden @ h_i                                   # [R]
        pred_token(i) = anchor (i == 0) else the previous step's selected draft
        E_i[c] = u_i[c] + sum_r pred_cb[pred_token, r] * g_i[r] * succ_cb[C_i[c], r]
        temperature <= 0:  j_i = lowest-rank argmax_c E_i[c];  q_i = one_hot(j_i)
        temperature > 0 :  q_i = softmax_c(E_i[c] / temperature); j_i ~ q_i
        d_(i+1) = C_i[j_i]   (becomes the next predecessor)

    Args:
        hidden: ``[K, H]`` bf16 -- the drafter's mask hidden columns ``h[1:]`` (row ``i`` is mask
            column ``i + 1``).
        proposal_head: a callable mapping ``h [K, H]`` bf16 -> raw public-vocab logits ``[K, vocab]``
            (fp32 or bf16; NO softmax/penalty). Agent 4 passes the target's output head
            (``lambda h: F.linear(h, output_head_weight)``).
        hidden_projection_w: ``[R, H]`` bf16 -- the selector ``hidden_projection`` weight.
        predecessor_codebook: ``[vocab, R]`` bf16 -- indexed ``[token, r]`` (ported convention).
        successor_codebook: ``[vocab, R]`` bf16 -- indexed ``[token, r]``.
        anchor: the anchor token id (a public global id; the ``i == 0`` predecessor).
        base_position: the frontier ``F`` (used only for the stochastic RNG key).
        temperature: ``<= 0`` selects greedy (deterministic, no RNG).
        seed: the request seed (stochastic RNG key).
        top_k: the shortlist size (16).

    Returns:
        ``(drafts, proposal_q, candidates)`` where:
        * ``drafts``: ``[K,]`` int64 -- the proposals ``d_1..d_K`` (global token ids).
        * ``proposal_q``: ``[top_k, K]`` fp32 -- the retained proposal distribution ``q_i`` per
            position (one-hot for greedy; the real softmax for temperature > 0). Consumed directly by
            the T8 stochastic path (NOT re-derived from scores).
        * ``candidates``: ``[top_k, K]`` int64 -- the global token ids ``C_i`` (the shortlist). With
            ``proposal_q`` these form the sparse proposal distribution over the public vocab (zero
            outside the shortlist) that T8's correction sampling needs.

    Tie-break (KEEP SEPARATE from T8): the selector argmax tie-break is the **lowest candidate
    rank** ``c`` -- ``torch.argmax`` returns the first (lowest-index) max, which is exactly that. The
    T8 verify argmax tie-break is the **lower token id**; the two differ and must not be conflated.
    Selector arithmetic (the 3-way codebook product and the ``E`` sum) runs in fp32.
    """
    K = hidden.shape[0]
    device = hidden.device
    # (a) raw public-vocab logits; top-16 with lower-token-id tie-break (stable argsort of -logits).
    logits = proposal_head(hidden).float()  # [K, vocab]
    order = torch.argsort(-logits, dim=-1, stable=True)  # [K, vocab]
    candidates = order[:, :top_k].transpose(0, 1).contiguous()  # [top_k, K] global ids
    u = torch.gather(logits, 1, order[:, :top_k]).transpose(0, 1).contiguous()  # [top_k, K] fp32
    # (b) projected hidden g_i (bf16 like the drafter's hidden_projection; upcast for the fp32 walk).
    g = F.linear(hidden, hidden_projection_w)  # [K, R] bf16
    g_f = g.float()  # [K, R]
    pred_cb = predecessor_codebook.float()  # [vocab, R]
    succ_cb = successor_codebook.float()  # [vocab, R]
    # (c) conditional left-to-right walk.
    pred_token = int(anchor)
    drafts: List[int] = []
    qs: List[torch.Tensor] = []
    gen = _make_generator(hidden)
    for i in range(K):
        C_i = candidates[:, i]  # [top_k] global ids
        pred_vec = pred_cb[pred_token]  # [R]
        succ_matrix = succ_cb[C_i]  # [top_k, R]
        edge = u[:, i] + (succ_matrix @ (g_f[i] * pred_vec))  # [top_k] fp32
        if temperature <= 0:
            j = int(torch.argmax(edge).item())  # first-max = lowest candidate rank
            q = torch.zeros(top_k, dtype=torch.float32, device=device)
            q[j] = 1.0
        else:
            q = torch.softmax(edge / temperature, dim=-1)  # [top_k] fp32
            gen.manual_seed(_mix64(seed, base_position + i, DFLASH2_PROPOSAL_PURPOSE))
            j = int(torch.multinomial(q, 1, replacement=False, generator=gen).item())
        d = int(C_i[j].item())
        drafts.append(d)
        qs.append(q)
        pred_token = d
    drafts_t = torch.tensor(drafts, dtype=torch.int64, device=device)  # [K]
    proposal_q = torch.stack(qs, dim=1)  # [top_k, K]
    return drafts_t, proposal_q, candidates


# ---------------------------------------------------------------------------
# T8 -- verify / accept (sparse draft acceptance)
# ---------------------------------------------------------------------------


def accept_sparse_drafts(
    target_dists: torch.Tensor,
    drafts: torch.Tensor,
    candidate_ids: torch.Tensor,
    proposal_q: torch.Tensor,
    *,
    greedy: bool,
    seed: int,
    base_position: int,
    target_argmax: Optional[torch.Tensor] = None,
) -> Tuple[int, torch.Tensor, int]:
    """Verify/accept the draft proposals against the target's per-position distributions (plan T8).

    A PURE function: it consumes the target verify forward's per-position distributions
    (``p_0..p_P``, each predicting position ``F + i + 1``) and the retained proposals. Let ``P =
    len(drafts)`` be the verifiable prefix (``0 <= P <= K``); there are ``P + 1`` live verify columns
    (``p_0`` is the anchor column, ``p_i`` predicts ``F + i + 1``). Draft ``d_(i+1) = drafts[i]`` is
    checked against target column ``p_i``.

    Greedy (temperature 0 -- SHIP THIS; fully deterministic):
        accept the longest target-argmax prefix; ``d_(i+1)`` is accepted iff
        ``drafts[i] == target_argmax[i]``. With ``A`` accepted (``0 <= A <= P``):
        * if ``A < P``: correction = ``target_argmax[A]``;
        * if ``A == P`` (all accepted): bonus = ``target_argmax[P]``.
        committed = ``[drafts[0..A-1], correction_or_bonus]`` (length ``A + 1``).
        The verify argmax tie-break is the **lower token id** (Agent 4 bakes that into
        ``target_argmax``; here it is already a token id).

    Temperature > 0 (documented + implemented; the round driver uses greedy first):
        accept ``d_(i+1)`` with prob ``min(1, p_i[drafts[i]] / q_i[drafts[i]])``; on first rejection
        the correction ~ ``r_i / sum(r_i)`` with ``r_i(v) = max(p_i(v) - q_i(v), 0)``; on all-accept
        the bonus ~ ``p_P``. ``q_i`` is the *sparse* proposal distribution: ``proposal_q[i, rank]``
        on ``candidate_ids[i, rank]``, zero elsewhere (never expanded to a dense distribution and
        never re-derived from scores -- ``proposal_q`` is consumed directly).

    Args:
        target_dists: ``[P + 1, vocab]`` fp32 -- the target's normalized distributions
            ``p_0..p_P`` over the public vocab.
        drafts: ``[P,]`` int -- the proposals ``d_1..d_P`` (the first ``P`` of the selector's ``K``).
        candidate_ids: ``[top_k, P]`` int -- the selector shortlist ``C_i`` for the ``P`` live
            positions (row ``i`` lists the tokens the sparse ``q_i`` is nonzero on).
        proposal_q: ``[top_k, P]`` fp32 -- the retained proposal distribution (T7 output, first
            ``P`` columns).
        greedy: ``True`` for temperature 0 (no RNG).
        seed: the request seed (stochastic RNG key).
        base_position: the frontier ``F`` (stochastic RNG key uses logical position ``F + i``).
        target_argmax: optional ``[P + 1,]`` int -- a precomputed (e.g. penalty-adjusted) lower-token-id
            argmax of each column, overriding the internal ``target_dists.argmax`` (which is already
            lower-token-id). Only the greedy path uses it.

    Returns:
        ``(accepted, committed, continuation_col)`` where:
        * ``accepted``: ``A`` (int) -- the accepted proposal count (``0 <= A <= P``).
        * ``committed``: ``[A + 1,]`` int64 -- the licensed tokens ``y = [d_1..d_A,
          correction_or_bonus]``.
        * ``continuation_col``: the verify column index (``== A``) whose target hidden is the
          continuation anchor feature for the full-license commit (column ``N - 1`` when ``N = A + 1``);
          the round driver gathers ``target_hidden[:, A]``. If it trims the commit to ``N < A + 1``,
          it selects column ``N - 1`` instead.
    """
    P = drafts.numel()
    device = target_dists.device
    # Lower-token-id argmax of each verify column. ``torch.argmax`` is first-max = lowest index =
    # the lower-token-id tie-break the verify requires (distinct from the selector's lowest-rank).
    # Supply a precomputed ``target_argmax`` (e.g. penalty-adjusted) to override.
    if target_argmax is None:
        argmax = target_dists.argmax(dim=-1)  # [P + 1]
    else:
        argmax = target_argmax
    argmax = argmax.to(torch.long)
    if P == 0:
        final = int(argmax[0].item())
        committed = torch.tensor([final], dtype=torch.int64, device=device)
        return 0, committed, 0
    drafts_l = drafts.to(torch.long)  # [P]

    if greedy:
        A = 0
        while A < P and int(drafts_l[A].item()) == int(argmax[A].item()):
            A += 1
        final = int(argmax[A].item())  # A <= P, argmax has P + 1 entries
        committed = torch.cat([drafts_l[:A], torch.tensor([final], dtype=torch.int64, device=device)])
        return A, committed, A

    # temperature > 0: sparse rejection sampling.
    q_rank = None  # lazy: the sparse q over the live shortlist, [P, vocab]
    gen = _make_generator(target_dists)
    A = 0
    rejected = False
    for i in range(P):
        p_i = target_dists[i]  # [vocab]
        d = int(drafts_l[i].item())
        # q_i[d]: the proposal probability of the draft = the selected rank's q (one match in C_i).
        q_i_d = _sparse_q_at(candidate_ids[:, i], proposal_q[:, i], d)
        accept_prob = min(1.0, float(p_i[d].item()) / q_i_d) if q_i_d > 0 else 1.0
        gen.manual_seed(_mix64(seed, base_position + i, DFLASH2_ACCEPT_PURPOSE))
        draw = float(torch.rand(1, device=device, generator=gen).item())
        if draw < accept_prob:
            A += 1
            continue
        # first rejection at column i: correction ~ r_i / sum(r_i), r_i = max(p_i - q_sparse_i, 0).
        if q_rank is None:
            q_rank = _sparse_q(candidate_ids, proposal_q, target_dists.shape[-1], device)  # [P, vocab]
        r = (p_i - q_rank[i]).clamp(min=0.0)
        rs = r.sum()
        final = 0
        if rs > 0:
            gen.manual_seed(_mix64(seed, base_position + A, DFLASH2_CORRECTION_PURPOSE))
            final = int(torch.multinomial(r / rs, 1, replacement=False, generator=gen).item())
        committed = torch.cat([drafts_l[:A], torch.tensor([final], dtype=torch.int64, device=device)])
        rejected = True
        break
    if not rejected:
        # all P accepted: bonus ~ p_P (column P).
        gen.manual_seed(_mix64(seed, base_position + P, DFLASH2_BONUS_PURPOSE))
        final = int(torch.multinomial(target_dists[P], 1, replacement=False, generator=gen).item())
        committed = torch.cat([drafts_l[:A], torch.tensor([final], dtype=torch.int64, device=device)])
        return A, committed, A
    return A, committed, A


def _sparse_q_at(cand_i: torch.Tensor, q_i: torch.Tensor, d: int) -> float:
    """``q_i[d]`` from a sparse proposal distribution: the selected rank's probability (``d`` is
    guaranteed to be a shortlist member). ``cand_i`` ``[top_k]`` ids, ``q_i`` ``[top_k]`` probs."""
    rank = int((cand_i.to(torch.long) == d).nonzero().flatten()[0].item())
    return float(q_i[rank].item())


def _sparse_q(
    candidate_ids: torch.Tensor, proposal_q: torch.Tensor, vocab: int, device: torch.device
) -> torch.Tensor:
    """Expand the sparse proposal distributions to a dense ``[P, vocab]`` (zero outside the
    shortlist). Built once per round, only on the rejection path (stochastic)."""
    P = candidate_ids.shape[1]
    q_dense = torch.zeros(P, vocab, dtype=torch.float32, device=device)
    for i in range(P):
        q_dense[i, candidate_ids[:, i].to(torch.long)] = proposal_q[:, i]
    return q_dense


__all__ = [
    "plain_rmsnorm",
    "rope_1d_split_half",
    "head_rmsnorm_rope",
    "repeat_kv",
    "dynamic_grouped_conv_prepare",
    "dynamic_grouped_conv_finish",
    "draft_attention",
    "noncausal_sliding_attention",
    "full_attention",
    "materialize_context_kv",
    "candidate_selector",
    "accept_sparse_drafts",
    "DFLASH2_PROPOSAL_PURPOSE",
    "DFLASH2_ACCEPT_PURPOSE",
    "DFLASH2_CORRECTION_PURPOSE",
    "DFLASH2_BONUS_PURPOSE",
]
