"""Qwen3.5 / 3.6 / 3.8 checkpoint reader.

The dense pass reads every Linear module under the scheme the checkpoint's QuantConfig gives it, the same answer the model built its buffers from, so bf16, block-fp8, ModelOpt and llm-compressor exports in any mix all land as the model's state dict. Routed experts are read by the offload cache (``nvfp4_expert_spec`` / ``iter_expert_pieces``); only bf16 stacked experts and resident block-fp8 experts come from here.
"""

from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4
from freetoken.layers.quantization import QuantConfig, QuantKind, QuantScheme, get_quant_config
from freetoken.models.config import VISION_KEY_PREFIXES
from freetoken.models.loader import ShardReader, iter_weight_files, safetensors_weight_map
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec
from freetoken.models.register import ModelSpec, get_model_spec
from freetoken.utils import cached_load_hf_config, download_hf_weight
from tqdm import tqdm

from .config import parse_config
from freetoken.models.qwen3_vl.weight import rename_vl_prefix

# bf16 checkpoints store the routed experts pre-stacked per layer
_STACKED_EXPERT_RE = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$")
# per-expert tensors of a quantized checkpoint: the offload cache's expert reader takes these
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
# the ``model.language_model.`` anchor excludes the MTP head's ``mtp.layers.N.mlp.experts.*``
_EXPERT_KEY_RE = (
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>{kinds})$"
)
# role -> the expert bank reader's canonical (ModelOpt) tensor kind
_BANK_KINDS = {"weight": "weight", "weight_scale": "weight_scale", "weight_global": "weight_scale_2"}

# Gemma-style (1+weight) RMSNorm weights; the GDN gated norm (linear_attn.norm) is a plain weight*x norm
_GEMMA_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)
# leaves the model builds as Linear layers: only their tensors are read under the QuantConfig, the rest passes through as stored
_LINEAR_LEAVES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj", "gate", "shared_expert_gate", "lm_head",
})
# activation scales of modules whose scheme carries no input_scale role
_DROPPED_SUFFIXES = frozenset({"input_scale", "input_global_scale"})
_ELEM_DTYPES = {"e4m3": torch.float8_e4m3fn, "e2m1": torch.uint8}
_QUANT_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8, torch.int8)


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        return None
    # static KV-cache scales of the quantizers; the KV cache runs in the engine's dtype
    if raw_name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
        return None
    return rename_vl_prefix(raw_name)


def _is_gemma_norm(name: str) -> bool:
    return name == "model.norm.weight" or name.endswith(_GEMMA_NORM_SUFFIXES)


def _per_row_scale(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-tensor scalar or per-channel ``[rows, 1]`` fp8 scale -> fp32 ``[rows]``; any other count is refused rather than broadcast onto the wrong rows."""
    flat = scale.reshape(-1).to(torch.float32)
    if flat.numel() == 1:
        return flat.expand(rows).contiguous()
    if flat.numel() != rows:
        raise ValueError(
            f"fp8 weight_scale has {flat.numel()} elements for a weight with {rows} output rows "
            f"(shape {tuple(scale.shape)}); expected 1 or {rows}"
        )
    return flat.contiguous()


def _dequant_nvfp4(weight: torch.Tensor, weight_scale: torch.Tensor, weight_global: torch.Tensor) -> torch.Tensor:
    """Packed NVFP4 -> bf16 on CUDA (the kernel is GPU-only, the converter reads on CPU), returned on the caller's device."""
    device = weight.device
    if device.type != "cuda":
        weight, weight_scale, weight_global = (t.to("cuda") for t in (weight, weight_scale, weight_global))
    slots = torch.zeros(1, dtype=torch.int32, device=weight.device)
    out = dequant_nvfp4(
        weight.unsqueeze(0).contiguous(), weight_scale.unsqueeze(0).contiguous(), weight_global.unsqueeze(0),
        slots, dtype=torch.bfloat16,
    )[0]
    return out.to(device)


def _dequant(scheme: QuantScheme, part: dict[str, torch.Tensor]) -> torch.Tensor:
    """bf16 weight of a module the checkpoint quantized but the family serves unquantized."""
    weight = part["weight"]
    if scheme.kind is QuantKind.FP8_TENSOR:
        return (weight.to(torch.float32) * part["weight_scale"][:, None]).to(torch.bfloat16)
    if scheme.kind is QuantKind.FP8_BLOCK:
        from freetoken.kernel.triton.fp8_block_linear import dequant_block_fp8

        return dequant_block_fp8(weight, part["weight_scale_inv"])
    if scheme.kind is QuantKind.NVFP4:
        return _dequant_nvfp4(weight, part["weight_scale"], part["weight_global"])
    raise NotImplementedError(f"no bf16 dequantization for {scheme}")


class _DenseReader:
    """Routes each Linear tensor to the buffer its module's scheme declares; packed projections are concatenated per role once every part is in."""

    def __init__(self, quant: QuantConfig | None, spec: ModelSpec) -> None:
        self.quant = quant
        self.groups = {fused: parts for fused, parts in spec.packed_modules_mapping if fused != "experts"}
        self.by_part: dict[str, list[tuple[str, int]]] = {}
        for fused, parts in self.groups.items():
            for idx, part in enumerate(parts):
                self.by_part.setdefault(part, []).append((fused, idx))
        # target module -> (part count, {part: {role: tensor}}, {part: the roles its module stores})
        self.pending: dict[str, tuple[int, dict[int, dict[str, torch.Tensor]], dict[int, set[str]], QuantScheme | None]] = {}

    def scheme(self, module: str) -> QuantScheme | None:
        return None if self.quant is None else self.quant.scheme_for(module)

    def stored(self, module: str) -> QuantScheme | None:
        """The scheme the checkpoint stores ``module`` under, before the family's unquantized_modules."""
        if self.quant is None:
            return None
        return self.quant.scheme_for_name(self.quant.name_map.to_checkpoint(module)[0])

    def target(self, module: str) -> tuple[str, int, int]:
        """``(fused module, part index, part count)``; a standalone linear is its own single-part target."""
        parent, _, leaf = module.rpartition(".")
        candidates = self.by_part.get(leaf)
        if not candidates:
            return module, 0, 1
        if len(candidates) > 1:
            # GDN: quantized checkpoints split qkv|z from the bf16 b|a; same test as gdn.py
            split = self.scheme(f"{parent}.in_proj_qkvz") is not None
            keep = {"in_proj_qkvz", "in_proj_ba"} if split else {"in_proj"}
            candidates = [c for c in candidates if c[0] in keep]
        fused, idx = candidates[0]
        return f"{parent}.{fused}", idx, len(self.groups[fused])

    def add(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        """Take one tensor; the emitted ``[(name, tensor)]`` once its target module is complete, ``[]`` before, None if ``name`` is not a Linear's tensor."""
        module, _, suffix = name.rpartition(".")
        if module.rpartition(".")[2] not in _LINEAR_LEAVES:
            return None
        stored = self.stored(module)
        roles = {"weight": "weight"} if stored is None else {e.name: r for r, e in self.quant.storage(stored).items()}
        role = roles.get(suffix)
        if role is None:
            if suffix in _DROPPED_SUFFIXES:
                return []
            raise ValueError(
                f"{name}: the checkpoint's quant config declares {module} {stored or 'unquantized'}, stored as {sorted(roles)}"
            )
        if stored is None and tensor.dtype in _QUANT_DTYPES:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} unquantized")
        if stored is not None and role == "weight" and tensor.dtype is not _ELEM_DTYPES[stored.weight.elem]:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} {stored}")
        target, idx, count = self.target(module)
        _, parts, expected, _ = self.pending.setdefault(target, (count, {}, {}, stored))
        parts.setdefault(idx, {})[role] = tensor
        expected[idx] = set(roles.values())
        if len(parts) < count or any(set(parts[i]) != expected[i] for i in parts):
            return []
        del self.pending[target]
        return self._emit(target, [parts[i] for i in range(count)], stored)

    def missing(self) -> list[str]:
        """One line per incomplete module: the roles its parts still lack."""
        lines = []
        for target, (count, parts, expected, stored) in sorted(self.pending.items()):
            lacking = sorted(set().union(*(expected[i] - set(parts[i]) for i in parts)))
            note = ""
            if lacking == ["input_scale"]:
                fix = "declares W4A16_NVFP4 or sets with_input_scale false" if stored is not None and stored.kind is QuantKind.NVFP4 else "sets with_input_scale false"
                note = f" (an export without activation scales {fix})"
            if len(parts) < count:
                lacking.append(f"{count - len(parts)} of {count} fused parts")
            lines.append(f"{target}: missing {lacking}{note}")
        return lines

    def _emit(self, target: str, parts: list[dict[str, torch.Tensor]], stored: QuantScheme | None):
        if stored is not None:
            parts = [self._check(target, stored, part) for part in parts]
            if self.scheme(target) is None:
                parts = [{"weight": _dequant(stored, part)} for part in parts]
        out = []
        for role in parts[0]:
            tensors = [part[role] for part in parts]
            if role == "input_scale":
                # fused parts read the same activation, so ModelOpt calibrates one range for them: max is exact then and safe if they drift
                value = torch.stack(tensors).max()
            else:
                value = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
            out.append((f"{target}.{role}", value))
        return out

    def _check(self, target: str, scheme: QuantScheme, part: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Validate one part against ``scheme`` and put its scales in the layer's form."""
        part = {
            role: 1.0 / tensor.to(torch.float32) if self.quant.storage(scheme)[role].reciprocal else tensor
            for role, tensor in part.items()
        }
        weight = part["weight"]
        if weight.dtype is not _ELEM_DTYPES[scheme.weight.elem]:
            raise ValueError(f"{target}: weight is {weight.dtype} but the checkpoint's quant config declares {scheme}")
        rows, cols = weight.shape[0], weight.shape[1] * (2 if scheme.weight.elem == "e2m1" else 1)
        block_rows, block_cols = scheme.weight.group or (1, 1)
        scale_role = "weight_scale_inv" if "weight_scale_inv" in part else "weight_scale"
        out = dict(part)
        if block_cols < 0:
            out[scale_role] = _per_row_scale(part[scale_role], rows)
        else:
            if rows % block_rows or cols % block_cols:
                raise ValueError(f"{target}: {rows}x{cols} weight is not a multiple of the {block_rows}x{block_cols} scale block of {scheme}")
            expected = (rows // block_rows, cols // block_cols)
            if tuple(part[scale_role].shape) != expected:
                raise ValueError(f"{target}: {scale_role} is {tuple(part[scale_role].shape)}, expected {expected} for {scheme}")
            if scheme.weight.scale == "e4m3" and part[scale_role].dtype is not torch.float8_e4m3fn:
                raise ValueError(f"{target}: {scale_role} is {part[scale_role].dtype} but {scheme} stores e4m3 scales")
        if "weight_global" in part:
            g = part["weight_global"].reshape(-1).to(torch.float32)
            if g.numel() != 1:
                raise ValueError(f"{target}: weight_global has {g.numel()} elements, expected one per-tensor scale")
            out["weight_global"] = g.to(torch.float16).expand(rows).contiguous()
        if "input_scale" in part:
            out["input_scale"] = part["input_scale"].reshape(()).to(torch.float32)
        return out


# ======================================================================================
# DFlash2 drafter reader (external checkpoint, e.g. z-lab/Qwen3.6-27B-DFlash2).
# ======================================================================================
#
# The DFlash2 drafter is an OPTIONAL component that lives in a SEPARATE HF repo, not in the
# target checkpoint. It reuses the target's ``token_embedding`` / ``output_head``, so those are
# NOT read from the draft repo. Everything else is read here, dequantized/cast to bf16 (the
# reference path serves the drafter in bf16 -- see plan open item O2), and renamed to the model's
# ``dflash2.``-prefixed state-dict keys.
#
# --- Expected DFlash2 drafter safetensors keys (open item O1: confirm on the box) ---
# Derived from ninfer src/models/qwen3_5/load/dflash.cpp + dflash2.cpp. The draft component's
# logical root prefix is ``dflash2``; the exact HF physical prefix (``dflash2/``? ``model.``? no
# prefix? ``/`` vs ``.`` separators?) must be confirmed against the draft repo's
# model.safetensors.index.json on the box. ``_dflash2_normalize`` tolerates any of them. The
# ninfer-derived logical set (per 27B DFlash2) is:
#
#   dflash2/feature_projection                              [H, num_taps*H]   bf16
#   dflash2/context_norm                                     [H]               bf16
#   dflash2/final_norm                                       [H]               bf16
#   dflash2/layers/{i}/input_norm                            [H]               bf16   (i in 0..L-1)
#   dflash2/layers/{i}/post_attention_norm                   [H]               bf16
#   dflash2/layers/{i}/attention/query                       [32*128, H]       bf16
#   dflash2/layers/{i}/attention/key                         [8*128,  H]       bf16
#   dflash2/layers/{i}/attention/value                       [8*128,  H]       bf16
#   dflash2/layers/{i}/attention/context_key                 [8*128,  H]       bf16
#   dflash2/layers/{i}/attention/context_value               [8*128,  H]       bf16
#   dflash2/layers/{i}/attention/query_norm                  [128]             bf16
#   dflash2/layers/{i}/attention/key_norm                    [128]             bf16
#   dflash2/layers/{i}/attention/output                      [H, 32*128]       bf16
#   dflash2/layers/{i}/mlp/gate                              [17408, H]        bf16
#   dflash2/layers/{i}/mlp/up                                [17408, H]        bf16
#   dflash2/layers/{i}/mlp/down                              [H, 17408]        bf16
#   dflash2/layers/{i}/attention_conv/base_kernel            [2, 2, H]         bf16   (checkpoint [side,tap,channel])
#   dflash2/layers/{i}/attention_conv/kernel_projection      [1280, H]         bf16
#   dflash2/layers/{i}/mlp_conv/base_kernel                  [2, 2, H]         bf16
#   dflash2/layers/{i}/mlp_conv/kernel_projection            [1280, H]         bf16
#   dflash2/candidate_selector/hidden_projection             [256, H]          bf16
#   dflash2/candidate_selector/predecessor_codebook          [vocab, 256]      bf16
#   dflash2/candidate_selector/successor_codebook            [vocab, 256]      bf16
#
# ``base_kernel`` is stored as ``[side, tap, channel]`` and transposed at load to the runtime
# ``[channel, tap, side]`` view (permute(2, 1, 0)).

# leading segments stripped when normalizing a draft physical key to its logical tail
_DFLASH2_ROOT_PREFIXES = ("dflash2", "dflash", "draft_model", "draft", "model")
# top-level (non-layer) logical tail -> FreeToken state-dict key
_DFLASH2_TOP_KEYS = {
    "feature_projection": "dflash2.feature_projection.weight",
    "context_norm": "dflash2.context_norm.weight",
    "final_norm": "dflash2.final_norm.weight",
    "candidate_selector.hidden_projection": "dflash2.candidate_selector.hidden_projection.weight",
    "candidate_selector.predecessor_codebook": "dflash2.candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook": "dflash2.candidate_selector.successor_codebook",
}
# per-layer logical tail -> the ``.``-suffixed leaf of the FreeToken key (layer index reinserted)
_DFLASH2_LAYER_KEYS = {
    "input_norm": "input_norm.weight",
    "post_attention_norm": "post_attention_norm.weight",
    "attention.query": "attention.query.weight",
    "attention.key": "attention.key.weight",
    "attention.value": "attention.value.weight",
    "attention.context_key": "attention.context_key.weight",
    "attention.context_value": "attention.context_value.weight",
    "attention.query_norm": "attention.query_norm.weight",
    "attention.key_norm": "attention.key_norm.weight",
    "attention.output": "attention.output.weight",
    "mlp.gate": "mlp.gate.weight",
    "mlp.up": "mlp.up.weight",
    "mlp.down": "mlp.down.weight",
    "attention_conv.base_kernel": "attention_conv.base_kernel",
    "attention_conv.kernel_projection": "attention_conv.kernel_projection.weight",
    "mlp_conv.base_kernel": "mlp_conv.base_kernel",
    "mlp_conv.kernel_projection": "mlp_conv.kernel_projection.weight",
}
_DFLASH2_LAYER_TAIL_RE = re.compile(r"^layers\.(\d+)\.(.*)$")


def _dflash2_normalize(raw: str) -> str:
    """Draft physical key -> separator-normalized logical tail.

    Unifies ``/`` and ``.`` separators and strips a single leading component/root prefix (see O1).
    E.g. ``dflash2/layers/0/attention/query`` and ``model.layers.0.attention.query`` both reduce to
    ``layers.0.attention.query``.
    """
    key = raw.replace("/", ".")
    head, _, rest = key.partition(".")
    if head in _DFLASH2_ROOT_PREFIXES and rest:
        key = rest
    return key


def _dflash2_free_token_key(normalized: str) -> str | None:
    """Normalized logical tail -> FreeToken ``dflash2.``-prefixed state-dict key, or None to skip.

    The ``.weight`` suffix matches the FreeToken Linear/norm idiom (LinearReplicated / RMSNorm
    store a ``weight`` param); ``base_kernel`` (a conv buffer) and the selector codebooks (plain
    parameters) carry no ``.weight``. The target embedding / output head are never mapped (they
    come from the target), so an unexpected key simply returns None.
    """
    if normalized in _DFLASH2_TOP_KEYS:
        return _DFLASH2_TOP_KEYS[normalized]
    m = _DFLASH2_LAYER_TAIL_RE.match(normalized)
    if m is None:
        return None
    index, tail = m.group(1), m.group(2)
    leaf = _DFLASH2_LAYER_KEYS.get(tail)
    if leaf is None:
        return None
    return f"dflash2.layers.{index}.{leaf}"


def _dflash2_to_bf16(raw: str, tensor: torch.Tensor) -> torch.Tensor:
    """Materialize one drafter tensor as bf16 for the reference path.

    Floating-point tensors (bf16/fp16/fp32/fp64) are cast to bf16. A quantized (Q8/NVFP4) draft
    checkpoint is a follow-up (plan open item O2): rather than silently mis-serve it, raise with a
    pointer at the open item.
    """
    if tensor.is_floating_point():
        return tensor.to(torch.bfloat16)
    raise NotImplementedError(
        f"DFlash2 drafter tensor {raw!r} is stored as {tensor.dtype}; the reference path serves the "
        "drafter in bf16. A quantized (Q8/NVFP4) DFlash2 draft checkpoint needs a dequant-at-load "
        "path (plan open item O2). Inspect the draft repo's model.safetensors.index.json on the box."
    )


def _expected_dflash2_tensors(cfg: "DFlash2DraftConfig", vocab_size: int) -> dict[str, tuple]:
    """The full set of FreeToken ``dflash2.`` keys a drafter of ``cfg`` must yield, with shapes."""
    H, taps, q, k, I, R, D = (
        cfg.hidden_size, cfg.num_taps, cfg.query_width, cfg.key_width,
        cfg.intermediate_size, cfg.selector_rank, cfg.head_dim,
    )
    conv_rows = 2 * cfg.conv_kernel_size * cfg.conv_groups
    base = (H, cfg.conv_kernel_size, 2)  # runtime [channel, tap, side] after the transpose
    tensors: dict[str, tuple] = {
        "dflash2.feature_projection.weight": (H, taps * H),
        "dflash2.context_norm.weight": (H,),
        "dflash2.final_norm.weight": (H,),
        "dflash2.candidate_selector.hidden_projection.weight": (R, H),
        "dflash2.candidate_selector.predecessor_codebook": (vocab_size, R),
        "dflash2.candidate_selector.successor_codebook": (vocab_size, R),
    }
    for i in range(cfg.num_layers):
        p = f"dflash2.layers.{i}."
        tensors[p + "input_norm.weight"] = (H,)
        tensors[p + "post_attention_norm.weight"] = (H,)
        tensors[p + "attention.query.weight"] = (q, H)
        tensors[p + "attention.key.weight"] = (k, H)
        tensors[p + "attention.value.weight"] = (k, H)
        tensors[p + "attention.context_key.weight"] = (k, H)
        tensors[p + "attention.context_value.weight"] = (k, H)
        tensors[p + "attention.query_norm.weight"] = (D,)
        tensors[p + "attention.key_norm.weight"] = (D,)
        tensors[p + "attention.output.weight"] = (H, q)
        tensors[p + "mlp.gate.weight"] = (I, H)
        tensors[p + "mlp.up.weight"] = (I, H)
        tensors[p + "mlp.down.weight"] = (H, I)
        tensors[p + "attention_conv.base_kernel"] = base
        tensors[p + "attention_conv.kernel_projection.weight"] = (conv_rows, H)
        tensors[p + "mlp_conv.base_kernel"] = base
        tensors[p + "mlp_conv.kernel_projection.weight"] = (conv_rows, H)
    return tensors


def iter_dflash2_weights(
    model_path: str,
    draft_path: str,
    device: torch.device,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Read the external DFlash2 drafter checkpoint (a separate HF repo/dir) and yield its tensors
    as bf16, renamed to the model's ``dflash2.``-prefixed state-dict keys.

    The drafter reuses the target's embedding/output head, so those are NOT read from ``draft_path``.
    When the target config carries a ``dflash2_args`` block the reader also verifies that every
    expected tensor is present with the right shape (loud failure, since the drafter cannot build
    otherwise); with no drafter config it simply yields whatever maps. See the key list above and
    plan open items O1 (physical key names) and O2 (draft quantization).
    """
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen3_5_moe DFlash2 drafter loading supports TP=1 only")

    # expected tensor set (from the target's drafter geometry) for completeness/shape validation
    target_cfg = parse_config(cached_load_hf_config(model_path))
    dflash_cfg = target_cfg.dflash2_args
    expected = (
        _expected_dflash2_tensors(dflash_cfg, target_cfg.vocab_size)
        if dflash_cfg is not None
        else None
    )

    folder = download_hf_weight(draft_path)
    weight_map = safetensors_weight_map(folder)  # physical key -> shard basename
    # discover the physical-key -> FreeToken mapping from the repo's actual keys (robust to O1 prefix)
    mapped: dict[str, str] = {}
    for raw in weight_map:
        ft_key = _dflash2_free_token_key(_dflash2_normalize(raw))
        if ft_key is not None:
            mapped[raw] = ft_key
    if not mapped:
        raise ValueError(
            f"no DFlash2 drafter tensors recognized under {draft_path}; the physical key names do not "
            "match the ninfer-derived set -- inspect model.safetensors.index.json and update "
            "_dflash2_normalize/_DFLASH2_TOP_KEYS/_DFLASH2_LAYER_KEYS (plan open item O1)"
        )

    reader = ShardReader(draft_path, device)
    yielded: set[str] = set()
    try:
        for raw, ft_key in mapped.items():
            tensor = _dflash2_to_bf16(raw, reader.get_tensor(raw))
            if ft_key.endswith(".base_kernel"):
                # checkpoint [side, tap, channel] -> runtime [channel, tap, side]
                tensor = tensor.permute(2, 1, 0).contiguous()
            if expected is not None:
                want = expected.get(ft_key)
                if want is None:
                    raise ValueError(f"mapped DFlash2 key {ft_key!r} is not in the expected set (config/reader drift)")
                if tuple(tensor.shape) != want:
                    raise ValueError(f"DFlash2 tensor {ft_key!r} has shape {tuple(tensor.shape)}, expected {want}")
            yielded.add(ft_key)
            yield ft_key, tensor
    finally:
        reader.close()

    if expected is not None:
        missing = sorted(set(expected) - yielded)
        if missing:
            shown = "\n  ".join(missing[:8]) + (f"\n  ... {len(missing) - 8} more" if len(missing) > 8 else "")
            raise ValueError(
                f"DFlash2 draft checkpoint {draft_path} is missing {len(missing)} tensors:\n  {shown}"
            )


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
    include_draft: bool = False,
    draft_path: str | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense weights fused to the model's buffers, and the routed experts only where a resident path takes them from here: bf16 stacked experts as stored, block-fp8 experts restacked per layer.

    Per-expert NVFP4 experts always come from the offload cache's expert reader.
    """
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen3_5_moe weight loading supports TP=1 only")
    hf_config = cached_load_hf_config(model_path)
    config = parse_config(hf_config)
    stacked = include_moe_experts and config.is_moe and config.expert_quant == "none"
    if include_non_moe or stacked:
        reader = _DenseReader(get_quant_config(), get_model_spec(hf_config.architectures[0])) if include_non_moe else None
        yield from _iter_shards(model_path, device, reader, stacked=stacked, include_vision=include_vision)
    if include_moe_experts and config.is_moe and config.expert_quant == "fp8_block":
        yield from _resident_fp8_experts(model_path, config)
    # the DFlash2 drafter, when enabled, yields its tensors (``dflash2.``-prefixed) alongside the target
    if include_draft:
        if draft_path is None:
            raise ValueError("include_draft requires draft_path (the DFlash2 drafter weights dir)")
        yield from iter_dflash2_weights(model_path, draft_path, device)


def _iter_shards(model_path: str, device: torch.device, reader: _DenseReader | None, *, stacked: bool, include_vision: bool):
    for file in tqdm(iter_weight_files(model_path), desc="Loading weights", disable=not get_tp_info().is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None or _EXPERT_RE.search(name):
                    continue
                if not include_vision and name.startswith(VISION_KEY_PREFIXES):
                    continue
                if _STACKED_EXPERT_RE.match(name):
                    if stacked:
                        yield name, f.get_tensor(raw_name)
                    continue
                if reader is None:
                    continue
                tensor = f.get_tensor(raw_name)
                emitted = reader.add(name, tensor)
                if emitted is not None:
                    yield from emitted
                elif _is_gemma_norm(name):
                    yield name, tensor + 1.0  # (1 + weight) baked into the stored norm weight
                else:
                    yield name, tensor
    if reader is not None and reader.pending:
        lines = reader.missing()
        shown = "\n  ".join(lines[:8]) + (f"\n  ... {len(lines) - 8} more" if len(lines) > 8 else "")
        raise ValueError(f"checkpoint is missing tensors the quant config declares for {len(lines)} modules:\n  {shown}")


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision tower alone, named as iter_weights names it."""
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is not None and name.startswith(VISION_KEY_PREFIXES):
                    yield name, f.get_tensor(raw_name)


def iter_weights_parallel(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[str, torch.Tensor]]:
    """experts-only parallel read via the common chunked multi-threaded O_DIRECT reader.
    Qwen3.5 stores experts pre-fused/pre-stacked per layer (already ``[E, ...]``), so no
    merge/stack -- just rename and yield; bank builder places by name as the serial path."""
    assert include_moe_experts and not include_non_moe, (
        "qwen3_5_moe parallel reader is experts-only (used by the expert piece reader)"
    )
    from freetoken.models.weight import iter_expert_tensors_parallel

    if get_tp_info().size > 1:
        raise NotImplementedError("qwen3_5_moe weight loading currently supports TP=1 only")

    def _is_expert(raw_name: str) -> bool:
        name = _rename(raw_name)
        return name is not None and _STACKED_EXPERT_RE.match(name) is not None

    for raw_name, tensor in iter_expert_tensors_parallel(
        model_path, _is_expert, workers=workers, chunk=chunk
    ):
        yield _rename(raw_name), tensor


# ======================================================================================
# Block-FP8 routed experts (Qwen3.5-35B-A3B-FP8): offload expert pieces and resident stacks.
# ======================================================================================

# Routed-expert checkpoint key (per-expert, un-fused). ``mtp.layers...`` is excluded by the
# ``model.language_model.`` anchor, so the parallel reader only sees the real experts.
_FP8_EXPERT_KEY_RE = (
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate|up|down)_proj\.(?P<kind>weight|{scale})$"
)


def _resident_fp8_experts(model_path, config):
    from freetoken.kernel.triton.fp8_block_linear import FP8

    B = 128
    L, E, H, I, dense = _moe_dims(config)
    shapes = {
        "gate_up_proj": ((E, 2 * I, H), FP8),
        "gate_up_scale_inv": ((E, 2 * I // B, H // B), torch.bfloat16),
        "down_proj": ((E, H, I), FP8),
        "down_scale_inv": ((E, H // B, I // B), torch.bfloat16),
    }
    layers: dict[int, dict[str, torch.Tensor]] = {}
    placed = [0] * L
    for li, e0, e1, piece in iter_expert_pieces(model_path, config, QuantKind.FP8_BLOCK, parallel=None):
        stack = layers.setdefault(li, {n: torch.empty(shape, dtype=dt) for n, (shape, dt) in shapes.items()})
        stack["gate_up_proj"][e0:e1, :I] = piece["gate"]
        stack["gate_up_proj"][e0:e1, I:] = piece["up"]
        stack["gate_up_scale_inv"][e0:e1, : I // B] = piece["gate_scale"]
        stack["gate_up_scale_inv"][e0:e1, I // B :] = piece["up_scale"]
        stack["down_proj"][e0:e1] = piece["down"]
        stack["down_scale_inv"][e0:e1] = piece["down_scale"]
        placed[li] += e1 - e0
        if placed[li] == E:
            pre = f"model.layers.{dense + li}.mlp.experts"
            for name, tensor in layers.pop(li).items():
                yield f"{pre}.{name}", tensor
    assert not layers, f"incomplete resident fp8 experts for layers {sorted(layers)}"


def _moe_dims(model_config):
    L = model_config.num_moe_layers
    return (
        L, model_config.num_experts, model_config.hidden_size,
        model_config.moe_intermediate_size, model_config.num_layers - L,  # dense prefix
    )


def iter_expert_pieces(model_path, config, kind: QuantKind, *, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20):
    """Block-fp8 routed experts, one piece per expert: ``{gate, up, down}`` fp8 codes and their
    ``_scale`` (block scale) companions, named as the checkpoint's dialect stores them. Other expert kinds use the generic readers."""
    if kind is not QuantKind.FP8_BLOCK:
        return None
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen3_5_moe fp8 expert banks support TP=1 only")
    from freetoken.models.weight import experts_scattered, iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    L, E, H, I, dense = _moe_dims(config)
    scale = get_quant_config().stored_tensors(QuantKind.FP8_BLOCK)["weight_scale_inv"].name
    key_re = re.compile(_FP8_EXPERT_KEY_RE.format(scale=re.escape(scale)))
    suffix = {"weight": "", scale: "_scale"}

    def locate(raw_name: str):
        m = key_re.match(raw_name)
        if m is None:
            return None
        li = int(m["layer"]) - dense
        if not 0 <= li < L:
            raise ValueError(f"unexpected routed-expert layer in {raw_name}")
        return li, int(m["expert"]), m["proj"] + suffix[m["kind"]]

    if parallel is None:
        parallel = experts_scattered(model_path)
    if parallel:
        tensors = iter_expert_tensors_parallel(
            model_path, lambda n: key_re.match(n) is not None, workers=workers, chunk=chunk
        )
        return per_expert_pieces(tensors, locate, tensors_per_expert=6)

    def _serial():
        reader = ShardReader(model_path, torch.device("cpu"))
        try:
            for li in tqdm(range(L), desc="Loading fp8 experts (serial)", disable=not get_tp_info().is_primary()):
                for e in range(E):
                    base = f"model.language_model.layers.{dense + li}.mlp.experts.{e}"
                    for proj in ("gate", "up", "down"):
                        for kind, suf in suffix.items():
                            name = f"{base}.{proj}_proj.{kind}"
                            yield name, reader.get_tensor(name)
        finally:
            reader.close()

    return per_expert_pieces(_serial(), locate, tensors_per_expert=6)


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    """The per-expert NVFP4 layout under the checkpoint's dialect names (ModelOpt or llm-compressor)."""
    quant = get_quant_config()
    stored = quant.stored_tensors(QuantKind.NVFP4)
    kind_map = {stored[role].name: kind for role, kind in _BANK_KINDS.items()}
    return Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(_EXPERT_KEY_RE.format(kinds="|".join(map(re.escape, kind_map)))),
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=lambda layer, config: layer,  # every layer is MoE
        desc=f"Qwen3.5 NVFP4 experts ({quant.dialect})",
        kind_map=kind_map,
        global_reciprocal=stored["weight_global"].reciprocal,
    )


__all__ = [
    "iter_weights",
    "iter_weights_parallel",
    "iter_expert_pieces",
    "nvfp4_expert_spec",
    "iter_dflash2_weights",
]
