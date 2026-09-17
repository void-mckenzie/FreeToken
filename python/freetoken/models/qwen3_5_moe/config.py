from __future__ import annotations

from typing import Any

from freetoken.layers.quantization import QuantConfig, QuantKind
from freetoken.models.config import (
    DFlash2DraftConfig,
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    mrope_layout_from_rope_params,
)
from freetoken.models.qwen3_vl.config import parse_vision_config


def _expert_quant(hf_config: Any, text: Any) -> tuple[str, tuple[int, int] | None]:
    """The routed experts' quant kind as the engine's format tag, with the scale block of block-fp8."""
    if not (getattr(text, "num_experts", 0) or 0):
        return "none", None
    # the engine reads this tag for its MoE strategy decisions; every module takes its own scheme from the QuantConfig when it is built
    scheme = QuantConfig.from_hf(hf_config).scheme_for_name("model.language_model.layers.0.mlp.experts.0.gate_proj")
    if scheme is None:
        return "none", None
    return str(scheme.kind), scheme.weight.group if scheme.kind is QuantKind.FP8_BLOCK else None


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        return list(layer_types)
    # Fall back to full_attention_interval: every Nth layer (1-indexed) is full.
    interval = int(getattr(text, "full_attention_interval", 4))
    n = int(text.num_hidden_layers)
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


# ------------------------------------------------------------------ DFlash2 drafter config
#
# The DFlash2 drafter is an OPTIONAL component of the target. Its geometry (attention dims,
# draft-layer count, FFN width, norm eps, 1-D RoPE theta, the residual taps, the sliding-window
# scalar S, the conv tap/group dims, the selector rank/top-k and the mask token id) is read from
# a drafter config block. ninfer reads it from a separate component whose inner ``dflash_config``
# carries the DFlash2-specific fields (see ninfer src/models/qwen3_5/config.cpp::draft); the exact
# on-box field names of the FreeToken target config.json are open item O4, so the lookup below is
# tolerant of a few plausible embeddings and defaults to None when absent (every existing
# checkpoint then loads unchanged).

# candidate top-level keys that, when present, hold the drafter config block
_DFLASH2_BLOCK_KEYS = ("dflash_config", "dflash2_config", "dflash2", "draft_config")


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a dict OR an attribute-style config object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _find_dflash2_block(hf_config: Any, text: Any) -> Any:
    """Locate the DFlash2 drafter config block, or None when the checkpoint carries no drafter.

    Accepts the block on the top-level config or on ``text_config``; the block is recognised when
    it carries the DFlash2 inner config (``dflash_config`` / ``target_layer_ids``) or the draft
    geometry (``num_hidden_layers``).
    """
    for root in (hf_config, text):
        for key in _DFLASH2_BLOCK_KEYS:
            block = _field(root, key)
            if block is None or isinstance(block, (str, int, float, bool)):
                continue
            if (
                _field(block, "dflash_config") is not None
                or _field(block, "target_layer_ids") is not None
                or _field(block, "num_hidden_layers") is not None
            ):
                return block
    return None


def parse_dflash2_args(hf_config: Any, text: Any, hidden_size: int) -> DFlash2DraftConfig | None:
    """Build a ``DFlash2DraftConfig`` from the checkpoint's drafter block, or ``None`` when absent.

    The hidden width H and the embedding/output head come from the target (the drafter stores
    neither). Attention/FFN/norm/RoPE geometry and the ``dflash_config`` DFlash2 fields are read
    from the block with the official 27B geometry as the default for any field left unspecified.
    """
    block = _find_dflash2_block(hf_config, text)
    if block is None:
        return None
    # the DFlash2-specific fields live in the inner ``dflash_config``; a flat block keeps them
    # alongside the geometry.
    inner = _field(block, "dflash_config")
    if inner is None:
        inner = block

    target_layer_ids = _field(inner, "target_layer_ids")
    if target_layer_ids is None:
        return None  # a block without the DFlash2 taps is not a drafter
    target_layer_ids = tuple(int(t) for t in target_layer_ids)
    if not target_layer_ids:
        return None

    rope_params = _field(block, "rope_parameters")
    rope_theta = _field(rope_params, "rope_theta", _field(block, "rope_theta", 1e7))

    cfg = DFlash2DraftConfig(
        hidden_size=int(hidden_size),
        num_layers=int(_field(block, "num_hidden_layers", 5)),
        intermediate_size=int(_field(block, "intermediate_size", 17408)),
        num_q_heads=int(_field(block, "num_attention_heads", 32)),
        num_kv_heads=int(_field(block, "num_key_value_heads", 8)),
        head_dim=int(_field(block, "head_dim", 128)),
        target_layer_ids=target_layer_ids,
        sliding_window=int(_field(block, "sliding_window", 2048)),
        mask_token_id=int(_field(inner, "mask_token_id", 0)),
        conv_kernel_size=int(_field(inner, "conv_kernel_size", 2)),
        conv_group_size=int(_field(inner, "conv_group_size", 16)),
        selector_rank=int(_field(inner, "selector_rank", 256)),
        selector_top_k=int(_field(inner, "selector_top_k", 16)),
        rms_norm_eps=float(_field(block, "rms_norm_eps", 1e-6)),
        rope_theta=float(rope_theta),
    )
    # mirror ninfer config.cpp::draft sanity checks (loud failure on an inconsistent block)
    if cfg.hidden_size % cfg.conv_group_size:
        raise ValueError(f"DFlash2 conv group size {cfg.conv_group_size} does not divide hidden width {cfg.hidden_size}")
    if cfg.num_q_heads % cfg.num_kv_heads:
        raise ValueError(f"DFlash2 query heads {cfg.num_q_heads} not divisible by KV heads {cfg.num_kv_heads}")
    return cfg


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)

    head_dim = (
        getattr(text, "head_dim", None)
        or text.hidden_size // text.num_attention_heads
    )
    num_kv_heads = getattr(text, "num_key_value_heads", text.num_attention_heads)

    rope_params = getattr(text, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
    partial = (
        rope_params.get("partial_rotary_factor")
        or getattr(text, "partial_rotary_factor", None)
        or 1.0
    )
    rotary_dim = int(head_dim * partial)

    # For text-only with the default rope type, partial NeoX rope needs no scaling dict
    # (the mRoPE params reduce to standard partial rope for text). Avoid carrying the
    # unhashable ``mrope_section`` list into get_rope's cache key.
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
    )

    expert_quant, weight_block_size = _expert_quant(hf_config, text)

    # Dense variants (e.g. Qwen3.6-27B) report num_experts==0: route the decoder MLP through
    # the dense Qwen3_5DenseMLP instead of the MoE block.
    num_experts = getattr(text, "num_experts", 0) or 0
    moe_enabled = num_experts > 0

    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    # 3-axis rope only with vision; text-only serving keeps the 1-D partial rope and the decode-graph layout
    vision_config = parse_vision_config(hf_config)
    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=text.max_position_embeddings,
        base=rope_theta,
        scaling=rope_scaling,
        mrope_section=(
            list(rope_params["mrope_section"])
            if vision_config is not None and "mrope_section" in rope_params
            else None
        ),
        mrope_layout=mrope_layout_from_rope_params(rope_params),
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=text.linear_num_key_heads,
        num_value_heads=text.linear_num_value_heads,
        key_head_dim=text.linear_key_head_dim,
        value_head_dim=text.linear_value_head_dim,
        conv_kernel_dim=text.linear_conv_kernel_dim,
        output_gate="silu",
    )
    # Order groups by their first layer id for deterministic iteration.
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=getattr(text, "intermediate_size", 0),
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=full_rotary,
        num_experts=num_experts,
        num_experts_per_tok=getattr(text, "num_experts_per_tok", 0),
        moe_intermediate_size=getattr(text, "moe_intermediate_size", 0),
        shared_expert_intermediate_size=getattr(text, "shared_expert_intermediate_size", 0),
        norm_topk_prob=True,
        moe_enabled=moe_enabled,
        use_qk_norm=True,
        model_type=getattr(hf_config, "model_type", "qwen3_5_moe"),
        architectures=getattr(hf_config, "architectures", ["Qwen3_5MoeForConditionalGeneration"]),
        vision_config=vision_config,
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        expert_quant=expert_quant,
        weight_block_size=weight_block_size,
        dflash2_args=parse_dflash2_args(hf_config, text, text.hidden_size),
    )


__all__ = ["parse_config", "parse_dflash2_args"]
