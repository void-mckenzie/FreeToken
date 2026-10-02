"""Grouped expert GEMM over native GGUF banks (borrowed ggml MoE kernels).

Ports vLLM/sglang's ``_fused_moe_gguf`` MMVQ path onto FreeToken's offload-cache
interface: the experts are streamed to the GPU as packed block bytes and
dequantized *inside* ``ggml_moe_a8_vec`` -- no bf16 expert copy is materialized. We
use the MMVQ (vector) kernel for both prefill and decode: it consumes ``topk_ids``
directly (no ``moe_align_block_size`` needed) and on small batches it is the right
choice anyway. ``topk_ids`` already index the streamed cache slots (decode) or the
materialized layer positions (prefill).

``fused_experts_gguf`` is the generic form: the caller names the ggml quant type of
each bank (``qt_gate_up`` / ``qt_down``), so a checkpoint that stores gate/up and
down in different types (GLM-5.3-Flash: IQ3_S + IQ4_XS) shares this path, and
clamped-SwiGLU experts get their clamp limit applied before the down projection.
``fused_experts_gguf_q4_0`` stays as the single-type wrapper the q4_0 dispatch uses.
"""

from __future__ import annotations

import torch

from freetoken.layers.activation import (
    gelu_and_mul,
    gelu_tanh_and_mul,
    silu_and_mul,
    swiglu_clamp_and_mul,
)
from freetoken.models.gguf.dequant import GGML_Q4_0

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def _activate(gate_up: torch.Tensor, activation: str, limit: float | None) -> torch.Tensor:
    """[N, 2I] gate|up rows -> [N, I] intermediate. ``swiglu_clamp`` (GLM-5.3 experts)
    clamps gate/up to ``limit`` inside the kernel like layers/moe.py does for the
    dense formats."""
    if activation == "swiglu_clamp":
        assert limit is not None, "swiglu_clamp experts require a clamp limit"
        return swiglu_clamp_and_mul(gate_up, alpha=1.0, limit=limit)
    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")
    return act_fn(gate_up)


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, row_bytes(H, qt_gate_up)] uint8
    down_q: torch.Tensor,  # [num_slots, H, row_bytes(I, qt_down)] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    qt_gate_up: int,
    qt_down: int,
    limit: float | None = None,
) -> torch.Tensor:
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    if activation not in _ACT and activation != "swiglu_clamp":
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]

    # gate_up: [num_tokens*top_k, 2I] -> activation -> [num_tokens*top_k, I]
    gate_up = ggml_moe_a8_vec(hidden_states, gate_up_q, topk_ids, top_k, qt_gate_up, n2, num_tokens)
    inter = _activate(gate_up, activation, limit)
    # down: each of the num_tokens*top_k intermediate rows uses its own expert id.
    out = ggml_moe_a8_vec(inter, down_q, topk_ids, 1, qt_down, h, num_tokens * top_k)
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


def fused_experts_gguf_q4_0(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, H//32*18] uint8
    down_q: torch.Tensor,  # [num_slots, H, I//32*18] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    return fused_experts_gguf(
        hidden_states, gate_up_q, down_q, topk_weights, topk_ids,
        activation, GGML_Q4_0, GGML_Q4_0,
    )


__all__ = ["fused_experts_gguf", "fused_experts_gguf_q4_0"]
