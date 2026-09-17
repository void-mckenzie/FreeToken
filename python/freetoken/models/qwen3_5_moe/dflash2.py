"""The DFlash2 speculative-decoding drafter (plan T3).

A small masked-block draft model, attached as an OPTIONAL component to the Qwen3.5 target
(plan T15 wires ``self.<attr> = DFlash2Drafter(config.dflash2_args, vocab_size, prefix="dflash2")``
onto ``Qwen3_5ForCausalLM`` so the state-dict keys land under the ``dflash2.`` prefix the weight
reader (T2) emits -- see ``weight.py::_expected_dflash2_tensors`` for the authoritative key set).

Given the target's committed hidden features (materialized by Agent 3 / T4 into a per-layer context
ring) and the unprocessed anchor token ``a = x_F`` (frontier ``F``), the drafter proposes ``K``
tokens in ONE non-causal forward over a ``W = K + 1`` masked block ``[a, MASK, ...]`` at absolute
positions ``[F, ..., F+K]``. The target then verifies the proposals causally and remains the output
authority. This module implements ONLY the draft forward (T3) plus the pure reference ops' host
modules (T5/T6 live in ``dflash2_ops.py``). The context materializer, candidate selector, and
verify/accept (T4/T7/T8/T9) are Agent 3's; they consume the modules registered here (the
``feature_projection`` / ``context_norm`` / per-layer ``context_key`` / ``context_value`` linears and
the ``candidate_selector`` codebooks) but are NOT run inside ``forward``.

Milestone-1 is reference-PyTorch-first (no new CUDA/Triton kernel): every op composes the pure
PyTorch helpers in ``dflash2_ops.py`` and FreeToken's bf16 ``LinearReplicated`` / plain
``RMSNorm``-equivalent. The drafter is served in bf16 and runs TP=1 only (replicated linears,
per-rank buffers; the draft/verify path never touches a TP communicator -- plan R3).
"""

from __future__ import annotations

import math
from typing import Sequence, TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, OPList
from freetoken.utils import nvtx_annotate

from .dflash2_ops import (
    full_attention,
    head_rmsnorm_rope,
    noncausal_sliding_attention,
    plain_rmsnorm,
    dynamic_grouped_conv_finish,
    dynamic_grouped_conv_prepare,
    materialize_context_kv,
    candidate_selector,
)

if TYPE_CHECKING:
    from typing import Callable
    from freetoken.kvcache.dflash2_ring_pool import DFlash2RingPool
    from freetoken.models.config import DFlash2DraftConfig


class PlainRMSNorm(BaseOP):
    """Plain (NOT Gemma ``(1+w)``) RMSNorm: ``w * x / sqrt(mean(x^2) + eps)``.

    DFlash2 uses the plain form for every norm (input_norm, post_attention_norm, context_norm,
    final_norm and the full-head q/k norm). Holds a single ``weight`` param so the state-dict key is
    ``...weight``; the math is the reference-PyTorch :func:`plain_rmsnorm` (fp32 reduce, cast back).
    Works for 2D hidden norms (last dim = H) and 3D per-head norms (last dim = D).
    """

    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return plain_rmsnorm(x, self.weight, self.eps)


class DynamicGroupedConv(BaseOP):
    """A DFlash2 dynamic grouped conv: ``prepare`` (input side) / ``finish`` (output side).

    Holds the ``base_kernel`` as a plain tensor *buffer* (loaded, not a learnable param -- no
    ``.weight`` suffix) in the runtime ``[channel, taps, side]`` view (the reader transposes the
    checkpoint's ``[side, tap, channel]`` at load, T2) and the ``kernel_projection`` as a
    bias-free bf16 ``LinearReplicated``. The reference math is in ``dflash2_ops.py``: ``prepare``
    emits the input-side prepared input AND the output-side ``finish_delta`` (computed ONCE from the
    normalized input); ``finish`` reuses that ``finish_delta`` on the branch output WITHOUT
    re-projecting it (plan R2).
    """

    def __init__(self, cfg: "DFlash2DraftConfig", *, prefix: str = "") -> None:
        self.taps = cfg.conv_kernel_size
        self.groups = cfg.conv_groups
        self.group_size = cfg.conv_group_size
        # runtime [channel, taps, side]; a loaded buffer (not a learnable param)
        self.base_kernel = torch.empty(cfg.hidden_size, cfg.conv_kernel_size, 2)
        self.kernel_projection = LinearReplicated(
            cfg.hidden_size, cfg.conv_projection_rows, has_bias=False,
            quant_config=None, prefix=f"{prefix}.kernel_projection",
        )

    def prepare(self, n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return dynamic_grouped_conv_prepare(
            self.base_kernel, self.kernel_projection.weight, n,
            taps=self.taps, groups=self.groups, group_size=self.group_size,
        )

    def finish(self, z: torch.Tensor, finish_delta: torch.Tensor) -> torch.Tensor:
        return dynamic_grouped_conv_finish(
            self.base_kernel, z, finish_delta,
            taps=self.taps, groups=self.groups, group_size=self.group_size,
        )


class DFlash2Attention(BaseOP):
    """The draft attention projections.

    ``query``/``key``/``value`` and the ``output`` projection are separate bias-free linears (NOT
    fused into a qkv), plus full-head ``query_norm``/``key_norm``. ``context_key``/``context_value``
    are the context-materialization projections (Agent 3 / T4) -- the modules live here so their
    state-dict keys match the contract, but they are NOT applied in the query-block ``forward``.
    """

    def __init__(self, cfg: "DFlash2DraftConfig", *, prefix: str = "") -> None:
        H, D = cfg.hidden_size, cfg.head_dim
        self.head_dim = D
        self.num_q = cfg.num_q_heads
        self.n_kv = cfg.num_kv_heads
        self.q_width = cfg.query_width
        self.k_width = cfg.key_width
        self.theta = cfg.rope_theta
        self.eps = cfg.rms_norm_eps
        self.query = LinearReplicated(H, cfg.query_width, has_bias=False, quant_config=None, prefix=f"{prefix}.query")
        self.key = LinearReplicated(H, cfg.key_width, has_bias=False, quant_config=None, prefix=f"{prefix}.key")
        self.value = LinearReplicated(H, cfg.key_width, has_bias=False, quant_config=None, prefix=f"{prefix}.value")
        # context K/V projections: applied in T4 (context materialization), not in the query-block forward
        self.context_key = LinearReplicated(H, cfg.key_width, has_bias=False, quant_config=None, prefix=f"{prefix}.context_key")
        self.context_value = LinearReplicated(H, cfg.key_width, has_bias=False, quant_config=None, prefix=f"{prefix}.context_value")
        self.query_norm = PlainRMSNorm(D, cfg.rms_norm_eps)
        self.key_norm = PlainRMSNorm(D, cfg.rms_norm_eps)
        self.output = LinearReplicated(cfg.query_width, H, has_bias=False, quant_config=None, prefix=f"{prefix}.output")

    def project(self, prepared: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Separate q/k/v projections then full-head per-head RMSNorm + 1-D split-half RoPE.

        ``prepared`` is ``[W, H]`` (the conv-prepared attention input); returns ``q`` ``[W, num_q, D]``
        and ``k``/``v`` ``[W, n_kv, D]`` (q/k post head-norm+rope, v raw).
        """
        W = prepared.shape[0]
        q = self.query.forward(prepared).view(W, self.num_q, self.head_dim)
        k = self.key.forward(prepared).view(W, self.n_kv, self.head_dim)
        v = self.value.forward(prepared).view(W, self.n_kv, self.head_dim)
        q = head_rmsnorm_rope(q, self.query_norm.weight, positions, self.theta, self.eps)
        k = head_rmsnorm_rope(k, self.key_norm.weight, positions, self.theta, self.eps)
        return q, k, v


class DFlash2GatedMLP(BaseOP):
    """The draft SwiGLU FFN: separate ``gate``/``up``/``down`` linears, ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, cfg: "DFlash2DraftConfig", *, prefix: str = "") -> None:
        H, I = cfg.hidden_size, cfg.intermediate_size
        self.gate = LinearReplicated(H, I, has_bias=False, quant_config=None, prefix=f"{prefix}.gate")
        self.up = LinearReplicated(H, I, has_bias=False, quant_config=None, prefix=f"{prefix}.up")
        self.down = LinearReplicated(I, H, has_bias=False, quant_config=None, prefix=f"{prefix}.down")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down.forward(F.silu(self.gate.forward(x)) * self.up.forward(x))


class DFlash2CandidateSelector(BaseOP):
    """The DFlash2 candidate-selector parameters (consumed by Agent 3 / T7, not in ``forward``).

    ``hidden_projection`` is a bias-free linear ``[R, H]`` and the ``predecessor``/``successor``
    codebooks are plain (non-``.weight``) params ``[vocab, R]`` -- registered here so their
    state-dict keys match the contract.
    """

    def __init__(self, cfg: "DFlash2DraftConfig", vocab_size: int, *, prefix: str = "") -> None:
        R = cfg.selector_rank
        self.hidden_projection = LinearReplicated(
            cfg.hidden_size, R, has_bias=False, quant_config=None, prefix=f"{prefix}.hidden_projection",
        )
        self.predecessor_codebook = torch.empty(vocab_size, R)
        self.successor_codebook = torch.empty(vocab_size, R)


class DFlash2DraftLayer(BaseOP):
    """One DFlash2 draft layer:

        n = plain_rmsnorm(x, input_norm)
        prepared_a, finish_delta_a = attention_conv.prepare(n)
        q, k, v = attention.project(prepared_a)                      # head-norm + 1-D rope on q/k
        a = noncausal_sliding/full_attn(q, concat(K_ctx, k), concat(V_ctx, v))
        y = x + attention_conv.finish(attention.output(a), finish_delta_a)
        m = plain_rmsnorm(y, post_attention_norm)
        prepared_m, finish_delta_m = mlp_conv.prepare(m)
        x = y + mlp_conv.finish(mlp(prepared_m), finish_delta_m)

    The context K/V for this layer arrive pre-projected/normalized/roped (T4); the query block is
    concatenated as the last ``W`` positions and every query row sees every query-block row.
    """

    def __init__(self, cfg: "DFlash2DraftConfig", layer_id: int, *, prefix: str = "") -> None:
        self._layer_id = layer_id
        self._is_local = True  # 27B DFlash2: all draft layers are sliding/local (no full layers)
        self.input_norm = PlainRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_norm = PlainRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attention = DFlash2Attention(cfg, prefix=f"{prefix}.attention")
        self.mlp = DFlash2GatedMLP(cfg, prefix=f"{prefix}.mlp")
        self.attention_conv = DynamicGroupedConv(cfg, prefix=f"{prefix}.attention_conv")
        self.mlp_conv = DynamicGroupedConv(cfg, prefix=f"{prefix}.mlp_conv")
        # execution scalars (not in state_dict)
        self.scale = 1.0 / math.sqrt(cfg.head_dim)
        self.window = cfg.sliding_window
        self.num_q = cfg.num_q_heads
        self.n_kv = cfg.num_kv_heads
        self.q_width = cfg.query_width

    @nvtx_annotate("DFlash2Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        context_k: torch.Tensor,
        context_v: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> torch.Tensor:
        # attention branch
        n = plain_rmsnorm(x, self.input_norm.weight, self.input_norm.eps)
        prepared_a, finish_delta_a = self.attention_conv.prepare(n)
        q, k, v = self.attention.project(prepared_a, positions)
        W = x.shape[0]
        full_k = torch.cat([context_k, k], dim=0)  # (S_live + W, n_kv, D)
        # context_v is FP16 (the ring stores V as FP16_RNE(BF16(v_raw)), plan T4) while the
        # query-block v is BF16; cast the query v to the ring's V dtype so the concat is well-typed
        # and the draft attention reads a uniform FP16 V stream (it upcasts to fp32 internally).
        full_v = torch.cat([context_v, v.to(context_v.dtype)], dim=0)
        key_positions = torch.cat([context_positions, positions])  # (S_live + W,)
        if self._is_local:
            a = noncausal_sliding_attention(
                q, full_k, full_v, positions, key_positions, self.scale, self.num_q, self.n_kv, self.window,
            )
        else:
            a = full_attention(q, full_k, full_v, positions, key_positions, self.scale, self.num_q, self.n_kv)
        y = x + self.attention_conv.finish(self.attention.output.forward(a.reshape(W, self.q_width)), finish_delta_a)

        # mlp branch
        m = plain_rmsnorm(y, self.post_attention_norm.weight, self.post_attention_norm.eps)
        prepared_m, finish_delta_m = self.mlp_conv.prepare(m)
        z_m = self.mlp.forward(prepared_m)
        return y + self.mlp_conv.finish(z_m, finish_delta_m)


class DFlash2Drafter(BaseOP):
    """The full DFlash2 drafter: feature projection + context norm + ``num_layers`` draft layers +
    final norm + candidate-selector parameters.

    ``feature_projection`` / ``context_norm`` are the target-conditioning pieces (T4) and the
    per-layer ``context_key`` / ``context_value`` are the context-materialization projections (T4) --
    all registered here for the state-dict contract but NOT run inside ``forward``. ``forward`` runs
    the masked-block draft backbone over the query block and returns the final per-position hidden
    columns after ``final_norm`` (the mask columns ``1..K`` feed Agent 3's selector).
    """

    def __init__(self, cfg: "DFlash2DraftConfig", vocab_size: int, *, prefix: str = "dflash2") -> None:
        self._cfg = cfg  # frozen dataclass (not a tensor/OP): excluded from state_dict
        self.num_layers = cfg.num_layers
        self.hidden_size = cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        # target conditioning (T4): bias-free feature projection + context norm
        self.feature_projection = LinearReplicated(
            cfg.feature_width, cfg.hidden_size, has_bias=False,
            quant_config=None, prefix=f"{prefix}.feature_projection",
        )
        self.context_norm = PlainRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.final_norm = PlainRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.layers = OPList(
            [DFlash2DraftLayer(cfg, i, prefix=f"{prefix}.layers.{i}") for i in range(cfg.num_layers)]
        )
        self.candidate_selector = DFlash2CandidateSelector(cfg, vocab_size, prefix=f"{prefix}.candidate_selector")

    @nvtx_annotate("DFlash2Draft")
    def forward(
        self,
        x0: torch.Tensor,
        positions: torch.Tensor,
        context_k: Sequence[torch.Tensor],
        context_v: Sequence[torch.Tensor],
        context_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run the masked-block draft backbone.

        Args:
            x0: ``[W, H]`` bf16 -- the TARGET embedding of the query block ``[anchor, MASK, ...]``
                (``W = K + 1``); the drafter reuses the target's ``token_embedding``.
            positions: ``[W,]`` int -- the block's absolute positions ``[F, F+1, ..., F+K]``.
            context_k: sequence of ``num_layers`` tensors, each ``[S_live, n_kv, D]`` bf16 -- the
                per-layer context K (already projected, head-normed and roped by T4's materializer).
            context_v: sequence of ``num_layers`` tensors, each ``[S_live, n_kv, D]`` **fp16** -- the
                per-layer context V (already projected by T4; the ring stores V as
                ``FP16_RNE(BF16(v_raw))`` -- the query-block v is cast to this dtype in each layer).
            context_positions: ``[S_live,]`` int -- the live context rows' absolute positions
                (shared across layers; the rows of ``context_k``/``context_v`` in order).

        Returns:
            ``[W, H]`` bf16 -- the final per-position hidden columns after ``final_norm``; the mask
            columns ``1..K`` are what Agent 3's candidate selector consumes (the anchor column 0 is
            kept here but dropped downstream).
        """
        x = x0
        for i, layer in enumerate(self.layers.op_list):
            x = layer.forward(x, positions, context_k[i], context_v[i], context_positions)
        return plain_rmsnorm(x, self.final_norm.weight, self.eps)

    def materialize_context(
        self,
        ring: "DFlash2RingPool",
        slot: int,
        features: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Materialize target residual features into the context ring and read back the live window.

        This is the drafter-side convenience wrapper for the T4 context materializer. It reuses the
        drafter's own modules (``feature_projection``/``context_norm`` + each layer's
        ``context_key``/``context_value``/``key_norm``), calls the pure ``materialize_context_kv`` op,
        scatters the result into ``ring`` at ``slot``, and returns the ring's live window in the exact
        shapes ``forward`` expects. Incremental-friendly: call it each round with the newly-committed
        positions (a contiguous append), or once with the full prefill ``[0, L)``.

        Args:
            ring: the per-request :class:`~freetoken.kvcache.dflash2_ring_pool.DFlash2RingPool`.
            slot: the request's ring slot (``table_idx``).
            features: ``[n, num_taps * H]`` bf16 -- ``s_t``, the concatenated target residual
                outputs at ``target_layer_ids`` (recorded order) for the ``n`` positions.
            positions: ``[n,]`` int -- the absolute positions, contiguous
                ``[frontier(slot), frontier(slot) + n)``.

        Returns:
            ``(context_k, context_v, context_positions)`` -- the ring's live window:
            ``context_k[i]``/``context_v[i]`` ``[S_live, n_kv, D]`` (K bf16, V fp16) per layer, and
            ``context_positions`` ``[S_live,]`` int. Feed these straight to :meth:`forward`.
        """
        cfg = self._cfg
        layer_kv = [
            (
                layer.attention.context_key.weight,
                layer.attention.context_value.weight,
                layer.attention.key_norm.weight,
            )
            for layer in self.layers.op_list
        ]
        context_k, context_v = materialize_context_kv(
            features,
            positions,
            self.feature_projection.weight,
            self.context_norm.weight,
            layer_kv,
            theta=cfg.rope_theta,
            eps=cfg.rms_norm_eps,
            n_kv=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
        )
        ring.write(slot, positions, context_k, context_v)
        return ring.read_live_window(slot)

    def select_candidates(
        self,
        hidden: torch.Tensor,
        proposal_head: "Callable[[torch.Tensor], torch.Tensor]",
        anchor: int,
        base_position: int,
        temperature: float,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the T7 candidate selector over the drafter's mask hidden columns.

        Convenience wrapper that pulls this drafter's selector parameters (``hidden_projection``,
        ``predecessor_codebook``, ``successor_codebook``) and calls the pure
        :func:`~freetoken.models.qwen3_5_moe.dflash2_ops.candidate_selector`.

        Args:
            hidden: ``[K, H]`` bf16 -- the draft ``forward`` output with the anchor column dropped
                (``h[1:]``).
            proposal_head: callable ``h [K, H]`` bf16 -> raw public-vocab logits ``[K, vocab]``
                (NO softmax/penalty). Agent 4 passes the target's output head.
            anchor: the anchor token id (global).
            base_position: the frontier ``F`` (stochastic RNG key).
            temperature: ``<= 0`` for greedy (deterministic, no RNG).
            seed: the request seed.

        Returns:
            ``(drafts, proposal_q, candidates)`` -- ``drafts`` ``[K,]`` int, ``proposal_q``
            ``[top_k, K]`` fp32, ``candidates`` ``[top_k, K]`` int (see
            :func:`~freetoken.models.qwen3_5_moe.dflash2_ops.candidate_selector`).
        """
        sel = self.candidate_selector
        return candidate_selector(
            hidden,
            proposal_head,
            sel.hidden_projection.weight,
            sel.predecessor_codebook,
            sel.successor_codebook,
            anchor,
            base_position,
            temperature,
            seed,
            top_k=self._cfg.selector_top_k,
        )


__all__ = [
    "DFlash2Drafter",
    "DFlash2DraftLayer",
    "DFlash2Attention",
    "DFlash2GatedMLP",
    "DFlash2CandidateSelector",
    "DynamicGroupedConv",
    "PlainRMSNorm",
]
