"""GLM-5.3-Flash GGUF adapter: build the FreeToken ``ModelConfig`` from GGUF metadata
and load native-GGUF weights (unsloth ``GLM-5.3-Flash-GGUF``, ``general.architecture``
= ``glm5next``).

The GGUF geometry is identical to the HF checkpoint (hybrid KDA/DSA layers, mHC x4,
288-expert MoE with a shared expert and clamped SwiGLU), so ``parse_gguf_config``
derives a text-config-shaped shim from the GGUF KV metadata + tensor-name census and
reuses ``glm5_next.config.parse_config`` wholesale -- the layer-type arrays that HF's
config carries are recovered from which tensors exist per block (``attn_q_a`` -> DSA,
else KDA; ``indexer.proj`` -> indexer "full"), and the trailing MTP block (the one
carrying ``blk.N.nextn.*``) is dropped exactly like the HF loader drops the
never-read MTP layer.

Quant formats in the served tier (unsloth UD-IQ4_XS): routed experts gate/up IQ3_S,
down IQ4_XS (streamed packed to the ggml MoE kernels via the ``gguf_glm5`` bank
format); attention/indexer/dense-MLP/shared-expert projections and the embedding
Q8_0; the untied head Q6_K; norms, routers and mHC tensors F32. The kv_b pair
(attn_k_b / attn_v_b) is the one fusion dequantized at load: the DSA module consumes
kv_b as a dense per-head bmm operand (``_kv_b``), which packed rows cannot back, so
it is materialized fused-bf16 in the module's (heads, nope+value, kv_lora) view
layout and the module runs unchanged.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.layers import BaseOP
from freetoken.models.config import ModelConfig
from freetoken.models.gguf.dequant import (
    GGML_IQ3_S,
    GGML_IQ4_XS,
    GGML_Q6_K,
    GGML_Q8_0,
    dequantize,
    row_bytes,
)

from .config import parse_config

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim

# Format tag for the routed-expert banks (offload-cache quant_format /
# model_config.expert_quant / moe_weight_format); the (IQ3_S, IQ4_XS) type pair lives
# in moe/offload_cache.GGUF_EXPERT_QTYPES keyed by it.
GGUF_GLM5_FORMAT = "gguf_glm5"

_KDA_LAYER = "linear_attention"
_DSA_LAYER = "deepseek_sparse_attention"

# Per-layer F32/bf16 one:1 tensors (gguf suffix -> (module-relative name, dtype)).
_LAYER_DENSE_MAP = {
    "attn_norm.weight": ("input_layernorm.weight", torch.bfloat16),
    "ffn_norm.weight": ("post_attention_layernorm.weight", torch.bfloat16),
    "hc_attn_fn.weight": ("hc_attn_fn", torch.float32),
    "hc_attn_base.weight": ("hc_attn_base", torch.float32),
    "hc_attn_scale.weight": ("hc_attn_scale", torch.float32),
    "hc_ffn_fn.weight": ("hc_ffn_fn", torch.float32),
    "hc_ffn_base.weight": ("hc_ffn_base", torch.float32),
    "hc_ffn_scale.weight": ("hc_ffn_scale", torch.float32),
    "ffn_gate_inp.weight": ("mlp.gate.weight", torch.bfloat16),
    "exp_probs_b.bias": ("mlp.e_score_correction_bias", torch.float32),
}
# DSA-only dense (F32) tensors.
_DSA_DENSE_MAP = {
    "attn_q_a_norm.weight": ("self_attn.q_a_layernorm.weight", torch.bfloat16),
    "attn_kv_a_norm.weight": ("self_attn.kv_a_layernorm.weight", torch.bfloat16),
    "indexer.proj.weight": ("self_attn.indexer.weights_proj.weight", torch.bfloat16),
    "indexer.k_norm.weight": ("self_attn.indexer.k_norm.weight", torch.bfloat16),
    "indexer.k_norm.bias": ("self_attn.indexer.k_norm.bias", torch.bfloat16),
    "indexer_compressor_gate.weight": ("self_attn.indexer.index_kpool_compress_gate", torch.bfloat16),
    "indexer_compressor_ape.weight": ("self_attn.indexer.index_kpool_compress_ape", torch.float32),
}
# KDA-only dense (F32) tensors.
_KDA_DENSE_MAP = {
    "ssm_a": ("self_attn.A_log", torch.float32),
    "ssm_dt.bias": ("self_attn.dt_bias", torch.float32),
    "ssm_norm.weight": ("self_attn.o_norm.weight", torch.bfloat16),
}
# Q8_0-packed projections per layer kind (gguf suffix -> module attr for GGUFLinear).
_DSA_PACKED_MAP = {
    "attn_q_a.weight": "self_attn.q_a_proj",
    "attn_q_b.weight": "self_attn.q_b_proj",
    "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa",
    "attn_output.weight": "self_attn.o_proj",
    "indexer.attn_q_b.weight": "self_attn.indexer.wq_b",
    "indexer.attn_k.weight": "self_attn.indexer.wk",
}
_KDA_PACKED_MAP = {
    "ssm_f_b.weight": "self_attn.f_b_proj",
    "ssm_g_b.weight": "self_attn.g_b_proj",
    "attn_output.weight": "self_attn.o_proj",
}
_DENSE_MLP_MAP = {
    "ffn_gate.weight": "mlp.gate_proj",
    "ffn_up.weight": "mlp.up_proj",
    "ffn_down.weight": "mlp.down_proj",
}
_SHEXP_MAP = {
    "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj",
    "ffn_up_shexp.weight": "mlp.shared_experts.up_proj",
    "ffn_down_shexp.weight": "mlp.shared_experts.down_proj",
}
# Routed-expert suffixes: streamed as packed banks, skipped by iter_gguf_weights.
_EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
# KDA in_proj fusion order; must match Glm5NextKDA._in_proj_split / weight.py _KDA_IN_PROJ.
_KDA_IN_PROJ = ("attn_q.weight", "attn_k.weight", "attn_v.weight",
                "ssm_beta.weight", "ssm_f_a.weight", "ssm_g_a.weight")
_KDA_CONV = ("ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight")


def _to_dtype(t, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize a GgufTensor (F32/F16/Q*) to a dense tensor of its torch shape."""
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, dtype)
    return flat.reshape(t.shape)


def _require_tp1(what: str) -> None:
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(
            f"glm5_next GGUF {what} currently supports TP=1 only "
            "(GGUF quant layers and expert banks are not tensor-parallel sharded)."
        )


def _block_of(name: str) -> int:
    return int(name.split(".")[1])


def _layer_census(names: set[str]):
    """Recover the per-layer layout the HF config would carry. The MTP block (the only
    block with ``.nextn.`` tensors) is excluded; KDA vs DSA comes from projection
    names, indexer presence from ``indexer.proj.weight``.

    Returns ``(num_layers, layer_types, indexer_types, has_indexer_compressor)``.
    """
    blocks = {_block_of(n) for n in names if n.startswith("blk.")}
    mtp = {_block_of(n) for n in names if ".nextn." in n}
    if len(mtp) != 1:
        raise ValueError(f"expected exactly one nextn (MTP) block, found {sorted(mtp)}")
    decoder = sorted(blocks - mtp)
    if decoder != list(range(len(decoder))):
        raise ValueError(f"GGUF blocks are not a dense 0..N range: {sorted(blocks)}")
    layer_types = []
    indexer_types = []
    for i in decoder:
        dsa = f"blk.{i}.attn_q_a.weight" in names
        layer_types.append(_DSA_LAYER if dsa else _KDA_LAYER)
        indexer_types.append("full" if f"blk.{i}.indexer.proj.weight" in names else "disabled")
    kcompress = any(n.endswith("indexer_compressor_gate.weight") for n in names)
    return len(decoder), layer_types, indexer_types, kcompress


def _kv_heads(shim: "GgufConfigShim"):
    """MLA head widths recovered from GGUF tensor shapes (torch order): q_b rows/heads
    = nope head width; v_b rows = value head width; conv channels give KDA heads."""
    from freetoken.models.gguf.reader import iter_gguf_tensors

    q_b = v_b = conv_q = None
    for t in iter_gguf_tensors(shim.model_path):
        if t.name.endswith("attn_q_b.weight") and "indexer" not in t.name and q_b is None:
            q_b = t
        elif t.name.endswith("attn_v_b.weight") and v_b is None:
            v_b = t
        elif t.name.endswith("ssm_conv1d_q.weight") and conv_q is None:
            conv_q = t
    if q_b is None or v_b is None:
        raise ValueError("GGUF lacks attn_q_b/attn_v_b to size the MLA heads")
    heads = int(shim.metadata["glm5next.attention.head_count"])
    qk_nope = q_b.shape[0] // heads
    v_head = v_b.shape[1]
    kda_heads = (conv_q.shape[0] // 3) // int(shim.metadata["glm5next.kda.head_dim"])
    return heads, qk_nope, v_head, kda_heads


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata

    def g(key: str):
        val = m.get(f"glm5next.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key glm5next.{key}")
        return val

    from freetoken.models.gguf.reader import gguf_tensor_names

    n_layers, layer_types, indexer_types, kcompress = _layer_census(gguf_tensor_names(shim.model_path))
    heads, qk_nope, v_head, kda_heads = _kv_heads(shim)
    first_dense = int(g("leading_dense_block_count"))
    clamp = m.get("glm5next.swiglu_clamp_exp")
    swiglu_limit = float(clamp[0] if isinstance(clamp, list) else clamp)

    text = SimpleNamespace(
        num_hidden_layers=n_layers,
        layer_types=tuple(layer_types),
        mlp_layer_types=("dense",) * first_dense + ("sparse",) * (n_layers - first_dense),
        first_k_dense_replace=first_dense,
        hidden_size=int(g("embedding_length")),
        num_attention_heads=heads,
        vocab_size=int(shim.vocab_size),
        intermediate_size=int(g("feed_forward_length")),
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        max_position_embeddings=int(g("context_length")),
        q_lora_rank=int(g("attention.q_lora_rank")),
        kv_lora_rank=int(g("attention.kv_lora_rank")),
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=0,  # NoPE: rope.dimension_count == 0
        v_head_dim=v_head,
        mla_use_nope=True,
        index_n_heads=int(g("attention.indexer.head_count")),
        index_head_dim=int(g("attention.indexer.key_length")),
        index_topk=int(g("attention.indexer.top_k")),
        indexer_types=tuple(indexer_types),
        indexer_rope_interleave=False,
        index_kpool=int(g("attention.indexer.kpool")),
        index_kpool_compress=bool(kcompress),
        # Not in the GGUF metadata (predates the llama.cpp key); the served
        # zai-org/GLM-5.3-Flash config sets it true, and so does llama.cpp's default.
        index_kpool_always_select_tail=True,
        linear_attn_config={
            "num_heads": kda_heads,
            "head_dim": int(g("kda.head_dim")),
            "short_conv_kernel_size": int(g("ssm.conv_kernel")),
            "gate_lower_bound": float(g("kda.gate_lower_bound")),
        },
        mhc=True,
        hc_mult=int(g("hyper_connection.count")),
        hc_sinkhorn_iters=int(g("hyper_connection.sinkhorn_iterations")),
        hc_eps=float(g("hyper_connection.epsilon")),
        n_routed_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        n_shared_experts=int(g("expert_shared_count")),
        norm_topk_prob=bool(g("expert_weights_norm")),
        routed_scaling_factor=float(g("expert_weights_scale")),
        n_group=int(g("expert_group_count")),
        topk_group=int(g("expert_group_used_count")),
        attention_bias=False,
        tie_word_embeddings=False,
        swiglu_limit=swiglu_limit,
    )
    hf = SimpleNamespace(
        text_config=text,
        model_type="glm5_next",
        architectures=list(shim.architectures),
    )
    cfg = parse_config(hf)
    return dataclasses.replace(
        cfg,
        expert_quant=GGUF_GLM5_FORMAT,
        moe_weight_format=GGUF_GLM5_FORMAT,
        tie_word_embeddings=False,  # the GGUF ships output.weight
        vision_config=None,  # the GGUF ships no vision tower
        architectures=list(shim.architectures),
    )


def is_gguf_model(config: ModelConfig) -> bool:
    return getattr(config, "moe_weight_format", None) == GGUF_GLM5_FORMAT


# --------------------------------------------------------------------------------------
# Weight loading: GGUF tensor names -> FreeToken glm5_next module params.
# --------------------------------------------------------------------------------------


def _fuse_kv_b(kb, vb, config) -> torch.Tensor:
    """attn_k_b (per-head [kv_lora, nope] stored transposed) + attn_v_b (per-head
    [value, kv_lora]) -> fused dense kv_b_proj.weight in the module's
    (heads, nope+value, kv_lora) view layout."""
    kb = _to_dtype(kb, torch.bfloat16)
    vb = _to_dtype(vb, torch.bfloat16)
    if kb.shape[-1] != vb.shape[-1]:  # k stored transposed; align to (heads, rows, lora)
        kb = kb.transpose(1, 2)
    fused = torch.cat([kb, vb], dim=1)  # (heads, nope+value, lora)
    heads = config.glm5_args.num_heads
    assert fused.shape[0] == heads, (fused.shape, heads)
    return fused.reshape(heads * fused.shape[1], fused.shape[2]).contiguous()


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every non-expert, non-MTP glm5_next param.

    Quantized projections (KDA/DSA attention, indexer wq_b/wk, dense MLP, shared
    expert) and the embedding stay packed and are yielded as ``.qweight`` (uint8);
    norms / router / mHC dequantize to bf16/fp32; the kv_b pair dequantizes and fuses
    into the module's dense layout (see module docstring). Routed experts stream
    through the offload cache (``load_gguf_expert_sources``)."""
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    assert not include_moe_experts, (
        "glm5_next GGUF experts serve from the offload cache via "
        "load_gguf_expert_sources(); they are not part of the state dict."
    )
    assert include_non_moe
    _require_tp1("weight loading")

    config = parse_gguf_config(cached_load_hf_config(model_path))
    args = config.glm5_args
    n_layers = config.num_layers
    first_dense = config.first_k_dense_replace
    kda_in_proj = {layer: {} for layer in range(n_layers) if args.is_kda_layer(layer)}
    kda_conv = {layer: {} for layer in range(n_layers) if args.is_kda_layer(layer)}
    kv_b = {}

    def emit_kda_in_proj(layer: int):
        base = f"model.layers.{layer}.self_attn"
        parts = kda_in_proj.pop(layer)
        assert set(parts) == set(_KDA_IN_PROJ), (
            f"KDA in_proj incomplete for layer {layer}: {sorted(parts)}"
        )
        yield f"{base}.in_proj.qweight", torch.cat([parts[s] for s in _KDA_IN_PROJ], dim=0)
        conv = kda_conv.pop(layer)
        assert set(conv) == set(_KDA_CONV), (
            f"KDA conv incomplete for layer {layer}: {sorted(conv)}"
        )
        yield f"{base}.conv1d.weight", torch.cat(
            [_to_dtype(conv[s], torch.bfloat16) for s in _KDA_CONV], dim=0
        )

    def kda_ready(layer: int) -> bool:
        return (
            layer in kda_in_proj
            and set(kda_in_proj[layer]) == set(_KDA_IN_PROJ)
            and set(kda_conv.get(layer, {})) == set(_KDA_CONV)
        )

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()
            continue
        if name == "output.weight":
            yield "lm_head.qweight", t.packed()
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _to_dtype(t, torch.bfloat16)
            continue
        if not name.startswith("blk."):
            continue
        layer = _block_of(name)
        if layer >= n_layers or ".nextn." in name:
            continue  # MTP block: never read (the HF loader drops it the same way)
        if any(name.endswith(s) for s in _EXPERT_SUFFIXES):
            continue  # routed experts -> offload banks
        suffix = name.split(".", 2)[2]
        base = f"model.layers.{layer}"

        # 1:1 dense tensors (all layers).
        if suffix in _LAYER_DENSE_MAP:
            rel, dtype = _LAYER_DENSE_MAP[suffix]
            yield f"{base}.{rel}", _to_dtype(t, dtype)
            continue
        # Dense MLP (leading layers) / shared expert (sparse layers).
        mlp_map = _DENSE_MLP_MAP if layer < first_dense else _SHEXP_MAP
        if suffix in mlp_map:
            yield f"{base}.{mlp_map[suffix]}.qweight", t.packed()
            continue

        if args.is_kda_layer(layer):
            if suffix in _KDA_IN_PROJ:
                kda_in_proj[layer][suffix] = t
                if kda_ready(layer):
                    yield from emit_kda_in_proj(layer)
                continue
            if suffix in _KDA_CONV:
                kda_conv.setdefault(layer, {})[suffix] = t
                if kda_ready(layer):
                    yield from emit_kda_in_proj(layer)
                continue
            if suffix in _KDA_DENSE_MAP:
                rel, dtype = _KDA_DENSE_MAP[suffix]
                val = _to_dtype(t, dtype)
                if suffix == "ssm_a":
                    # llama.cpp's glm5next converter stores this pre-transformed as
                    # -exp(A_log) (conversion/glm.py); FreeToken's KDA kernels expect
                    # the raw A_log and apply the exp inside. Invert at load.
                    val = torch.log(-val)
                yield f"{base}.{rel}", val
                continue
            if suffix in _KDA_PACKED_MAP:
                yield f"{base}.{_KDA_PACKED_MAP[suffix]}.qweight", t.packed()
                continue
            if suffix in _DSA_DENSE_MAP or suffix in _DSA_PACKED_MAP:
                raise ValueError(f"DSA tensor in a KDA layer: {name}")
            raise ValueError(f"unmapped glm5next KDA GGUF tensor: {name}")

        # DSA layer. kv_b fuses when both halves have arrived.
        if suffix in ("attn_k_b.weight", "attn_v_b.weight"):
            slot = "k" if suffix == "attn_k_b.weight" else "v"
            kv_b.setdefault(layer, {})[slot] = t
            if len(kv_b[layer]) == 2:
                kb_vb = kv_b.pop(layer)
                yield (
                    f"{base}.self_attn.kv_b_proj.weight",
                    _fuse_kv_b(kb_vb["k"], kb_vb["v"], config),
                )
            continue
        if suffix in _DSA_DENSE_MAP:
            rel, dtype = _DSA_DENSE_MAP[suffix]
            yield f"{base}.{rel}", _to_dtype(t, dtype)
            continue
        if suffix in _DSA_PACKED_MAP:
            yield f"{base}.{_DSA_PACKED_MAP[suffix]}.qweight", t.packed()
            continue
        if suffix in _KDA_DENSE_MAP or suffix in _KDA_PACKED_MAP or suffix in _KDA_IN_PROJ or suffix in _KDA_CONV:
            raise ValueError(f"KDA tensor in a DSA layer: {name}")
        raise ValueError(f"unmapped glm5next DSA GGUF tensor: {name}")

    # every KDA layer emitted its in_proj/conv (emit pops the entries) and every DSA
    # layer completed its kv_b pair (fusion pops the entries).
    assert not kda_in_proj, f"missing/incomplete KDA in_proj layers: {sorted(kda_in_proj)}"
    assert not kda_conv, f"missing/incomplete KDA conv layers: {sorted(kda_conv)}"
    assert not kv_b, f"incomplete kv_b pairs: {sorted(kv_b)}"


# --------------------------------------------------------------------------------------
# Model swap: dense bf16 Linear/Embedding -> native GGUF-quant ops.
# --------------------------------------------------------------------------------------


class GgufLMHead(BaseOP):
    """Untied LM head over a native Q6_K ``[vocab, row_bytes(hidden)]`` table.

    Mirrors ``ParallelLMHead``'s last-token-of-prefill behavior (logits only needed at
    sequence ends) while keeping the table packed; the GEMV runs through
    ``fused_mul_mat_gguf`` (MMVQ). The ``qweight`` attr matches the loader key
    ``lm_head.qweight`` through the standard BaseOP state-dict naming."""

    def __init__(self, num_embeddings: int, embedding_dim: int, quant_type: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._quant_type = quant_type
        self.qweight = torch.empty(
            num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx
        from freetoken.layers.gguf import fused_mul_mat_gguf

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return fused_mul_mat_gguf(x, self.qweight, self._quant_type)


def _swap_linear(owner, attr: str) -> None:
    """Replace ``owner.<attr>`` (dotted paths allowed) with a native Q8_0 GGUFLinear
    sized from the existing dense Linear."""
    from freetoken.layers.gguf import GGUFLinear

    parts = attr.split(".")
    for p in parts[:-1]:
        owner = getattr(owner, p, None)
        if owner is None:
            return
    lin = getattr(owner, parts[-1], None)
    if lin is None or not hasattr(lin, "weight"):
        return
    out_features, in_features = lin.weight.shape
    setattr(owner, parts[-1], GGUFLinear(in_features, out_features, GGML_Q8_0, has_bias=False))


def convert_glm5_next_to_gguf(model, config: ModelConfig) -> None:
    """In place: replace glm5_next's dense projections + embedding with native GGUF ops.

    Swapped to GGUFLinear (Q8_0): KDA in/f_b/g_b/o, DSA q_a/q_b/kv_a/o + indexer
    wq_b/wk, dense MLP and shared expert; the embedding becomes GGUFEmbedding (Q8_0)
    and the untied head a packed ``GgufLMHead`` (Q6_K). Left dense: conv1d, all
    norms, kv_b's bmm operands (dequantized at load), mHC, the router."""
    from freetoken.layers.gguf import GGUFEmbedding

    inner = model.model
    inner.embed_tokens = GGUFEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        quant_type=GGML_Q8_0,
        embed_scale=None,
    )
    model.lm_head = GgufLMHead(config.vocab_size, config.hidden_size, GGML_Q6_K)

    args = config.glm5_args
    for i, layer in enumerate(inner.layers.op_list):
        attn = layer.self_attn
        if args.is_kda_layer(i):
            for attr in ("in_proj", "f_b_proj", "g_b_proj", "o_proj"):
                _swap_linear(attn, attr)
        else:
            for attr in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
                _swap_linear(attn, attr)
            for attr in ("indexer.wq_b", "indexer.wk"):
                _swap_linear(attn, attr)
        mlp = layer.mlp
        if i >= config.first_k_dense_replace:
            for attr in ("gate_proj", "up_proj", "down_proj"):
                _swap_linear(mlp.shared_experts, attr)
        else:
            for attr in ("gate_proj", "up_proj", "down_proj"):
                _swap_linear(mlp, attr)


# --------------------------------------------------------------------------------------
# Routed-expert host banks (native IQ3_S / IQ4_XS) for the offload cache.
# --------------------------------------------------------------------------------------


def _gguf_expert_specs(config: ModelConfig) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    E = config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    return {
        "gate_up": ((E, 2 * I, row_bytes(H, GGML_IQ3_S)), torch.uint8),
        "down": ((E, H, row_bytes(I, GGML_IQ4_XS)), torch.uint8),
    }


def load_gguf_expert_sources(
    model_path: str, config: ModelConfig, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Per-layer host banks of the routed experts' native IQ3_S/IQ4_XS block bytes.

    ``gate_up`` is ``[E, 2I, H//256*110]`` (gate|up packed rows concatenated on the
    output axis) and ``down`` ``[E, H, I//256*136]`` per sparse layer; the MTP block's
    experts are skipped like the HF loader skips the MTP layer. Bank lifecycle
    matches the gemma4 GGUF path: mmap HostBanks, pin-after-fill (or converter sink),
    ``LayerCompletionTracker`` fires twice per layer (gate_up + down writes).
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    _require_tp1("expert banks")
    first_dense = config.first_k_dense_replace
    sparse_layers = set(range(first_dense, config.num_layers))
    E = config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    hb = alloc_layer_banks(_gguf_expert_specs(config), config.num_layers)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    gu_bytes = row_bytes(H, GGML_IQ3_S)
    dn_bytes = row_bytes(I, GGML_IQ4_XS)
    seen = {"gate": set(), "up": set(), "down": set()}
    gu_parts: dict[int, list[torch.Tensor]] = {}

    def _flush_gate_up(layer: int, tracker) -> None:
        parts = gu_parts.get(layer)
        if parts is not None and len(parts) == 2:
            del gu_parts[layer]
            banks["gate_up"][layer].copy_(torch.cat(parts, dim=1))
            tracker.note(layer)

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None
        for t in iter_gguf_tensors(model_path):
            if not t.name.startswith("blk."):
                continue
            layer = _block_of(t.name)
            if layer not in sparse_layers:
                continue
            if t.name.endswith("ffn_gate_exps.weight"):
                assert t.ggml_type == GGML_IQ3_S, f"{t.name}: expected IQ3_S gate, got {t.ggml_type}"
                gu_parts.setdefault(layer, []).append(t.packed().reshape(E, I, gu_bytes))
                seen["gate"].add(layer)
                if tracker is not None:
                    _flush_gate_up(layer, tracker)
            elif t.name.endswith("ffn_up_exps.weight"):
                assert t.ggml_type == GGML_IQ3_S, f"{t.name}: expected IQ3_S up, got {t.ggml_type}"
                gu_parts.setdefault(layer, []).append(t.packed().reshape(E, I, gu_bytes))
                seen["up"].add(layer)
                if tracker is not None:
                    _flush_gate_up(layer, tracker)
            elif t.name.endswith("ffn_down_exps.weight"):
                assert t.ggml_type == GGML_IQ4_XS, f"{t.name}: expected IQ4_XS down, got {t.ggml_type}"
                banks["down"][layer].copy_(t.packed().reshape(E, H, dn_bytes))
                seen["down"].add(layer)
                if tracker is not None:
                    tracker.note(layer)

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)

    want = sparse_layers
    assert seen["gate"] == want and seen["up"] == want and seen["down"] == want, (
        f"missing gguf expert layers: gate {sorted(want - seen['gate'])}, "
        f"up {sorted(want - seen['up'])}, down {sorted(want - seen['down'])}"
    )
    return banks


def dummy_gguf_expert_sources(config: ModelConfig) -> dict[str, list[torch.Tensor]]:
    """Random IQ3_S/IQ4_XS expert banks shaped like ``load_gguf_expert_sources`` output."""
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    hb = alloc_layer_banks(_gguf_expert_specs(config), config.num_layers)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    for t in banks["gate_up"] + banks["down"]:
        t.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(hb)
    return banks


__all__ = [
    "GGUF_GLM5_FORMAT",
    "parse_gguf_config",
    "iter_gguf_weights",
    "convert_glm5_next_to_gguf",
    "is_gguf_model",
    "load_gguf_expert_sources",
    "dummy_gguf_expert_sources",
]
