from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.blocks import embed_input_ids
from freetoken.models.qwen3_vl.vision import Qwen3VLVisionModel, QwenVLVisionMixin
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .dflash2 import DFlash2Drafter
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                quant_config=config.quant,
                prefix=f"{prefix}.linear_attn",
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = (
            Qwen3_5MoE(config, layer_id, prefix=f"{prefix}.mlp")
            if config.moe_enabled
            else Qwen3_5DenseMLP(config, prefix=f"{prefix}.mlp")
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = embed_input_ids(self.embed_tokens, input_ids, get_global_ctx().batch)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        x, _ = self.norm.forward_add_residual(x, residual)
        return x


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        # Optional DFlash2 speculative-draft drafter (plan T15). Attached ONLY when the checkpoint
        # carries a drafter geometry (``config.dflash2_args``) so every existing no-drafter
        # checkpoint builds the byte-for-byte unchanged model object (no ``dflash2`` attribute,
        # no drafter keys in ``state_dict``). Built on the meta device like every other sub-module
        # (the engine constructs the model under ``torch.device("meta")``). It is a BaseOP sub-module
        # named ``dflash2``: that attribute name is what puts its ``state_dict`` keys under the
        # ``dflash2.`` prefix the T2 weight reader emits (the drafter's internal ``prefix="dflash2"``
        # is the LinearReplicated quant-method prefix and agrees with it), so ``load_state_dict`` and
        # ``finalize_quant`` recurse into it automatically with no engine change.
        if config.dflash2_args is not None:
            self.dflash2 = DFlash2Drafter(config.dflash2_args, config.vocab_size, prefix="dflash2")
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)

    def capture_dflash2_features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the target over the active batch's input tokens and capture the DFlash2 drafter's
        conditioning features for the speculative verify forward (plan T15; consumed by Agent 4b's
        round driver).

        Eager, non-graph-captured: this re-runs the same decoder layer loop as ``forward`` but, after
        each block whose zero-based id is in ``config.dflash2_args.target_layer_ids``, snapshots the
        block's COMPLETE residual output (after BOTH the mixer/attention AND the FFN residual adds)
        and assembles the drafter's conditioning input. The round driver calls it directly (NOT via
        the ``graph_runner``), wrapped in ``ctx.forward_batch(batch)``, exactly like the prefill path.
        To move the whole draft->verify round to CUDA-graph capture in a later milestone, the target
        verify forward and the drafter draft forward would be recorded into a graph keyed on
        (padded block size, batch size) with the per-request ring / pending-feature buffers as static
        graph inputs (mirroring how ``GraphRunner`` records the decode graphs), and the FLA
        ``num_accepted_tokens`` seam would steer each GDN layer's final state to the accepted column.
        This milestone keeps it eager for correctness-first.

        The attention / GatedDeltaNet sub-modules read their absolute positions and KV state from the
        active batch (``get_global_ctx().batch``), exactly as ``forward`` does, so the round driver
        arranges the batch BEFORE calling and wraps the call in ``ctx.forward_batch(batch)`` (with the
        attention metadata prepared), just as a normal decode/prefill step. Required batch state:

        * ``batch.input_ids``  -- the verify-block tokens ``[a, d_1, ..., d_P]`` (``W = P + 1`` columns)
          laid out flat over the batch.
        * ``batch.positions``  -- the absolute positions ``[F, F+1, ..., F+P]`` of those columns.
        * ``batch.out_loc``    -- the target-KV pages the W verify tokens write into (the verify K/V are
          committed at the verify positions), plus the GDN state routing (``linear_table_idx`` /
          ``fla_metadata``) the hybrid layers use.
        * the target KV cache committed up to the frontier ``F`` (all context positions ``< F``
          resident); the verify forward appends the W columns on top.

        Args:
            None -- reads the active ``get_global_ctx().batch`` (mirrors ``forward``).

        Returns:
            ``(s_t, final_hidden, logits)`` where:
            * ``s_t``: ``[n, num_taps * H]`` bf16 -- the concatenated complete residual outputs at
              ``target_layer_ids`` in their recorded (config) order, ``s_t = concat(r_t^l for l in
              target_layer_ids)``; the drafter's conditioning input (see plan T4).
            * ``final_hidden``: ``[n, H]`` bf16 -- the final-norm hidden per column (the continuation
              anchor source; column ``N - 1`` of a ``N``-commit feeds the next round).
            * ``logits``: ``[n, vocab]`` -- the raw output-head result per column (for the verify
              distributions ``p_i``; Agent 4b applies penalties/temperature for T8 acceptance).
        """
        dflash2 = getattr(self, "dflash2", None)
        if dflash2 is None:
            raise RuntimeError(
                "capture_dflash2_features needs a DFlash2 drafter; this checkpoint carries no "
                "dflash2_args. Build from a drafter checkpoint to enable speculative decoding."
            )
        target_layer_ids = tuple(dflash2._cfg.target_layer_ids)
        num_layers = len(self.model.layers.op_list)
        if not target_layer_ids:
            raise ValueError("DFlash2 drafter has no target_layer_ids (empty conditioning taps)")
        if any(l < 0 or l >= num_layers for l in target_layer_ids):
            raise ValueError(
                f"DFlash2 target_layer_ids {target_layer_ids} out of range for the "
                f"{num_layers}-block target"
            )

        batch = get_global_ctx().batch
        x = embed_input_ids(self.model.embed_tokens, batch.input_ids, batch)
        residual: torch.Tensor | None = None
        target_set = set(target_layer_ids)
        taps_by_layer: dict[int, torch.Tensor] = {}
        for layer_id, layer in enumerate(self.model.layers.op_list):
            x, residual = layer.forward(x, residual)
            if layer_id in target_set:
                # The complete post-block residual r_t^l = running residual stream (the
                # mixer/attention add already folded in by the fused add-norm) PLUS this block's FFN
                # branch output. ``Qwen3_5DecoderLayer.forward`` returns (branch_out, residual_stream),
                # so this is exactly r_t^l AFTER both residual adds -- NOT the pre-norm / mid-block
                # value. ``+`` materializes a fresh bf16 tensor, so it is a snapshot safe from the
                # next block's in-place residual add (GemmaRMSNorm.forward_add_residual mutates the
                # residual buffer in place).
                taps_by_layer[layer_id] = residual + x
        final_hidden, _ = self.model.norm.forward_add_residual(x, residual)
        logits = self.lm_head.forward(final_hidden)
        # concat in the target_layer_ids (recorded) order so s_t matches the drafter's
        # feature_projection layout (num_taps * H), regardless of the taps' sort order.
        s_t = torch.cat([taps_by_layer[l] for l in target_layer_ids], dim=-1)
        return s_t.to(torch.bfloat16), final_hidden.to(torch.bfloat16), logits


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    """The MoE releases share the dense code path: the decoder picks the routed or dense MLP from config.num_experts."""


class Qwen3_5ForConditionalGeneration(QwenVLVisionMixin, Qwen3_5ForCausalLM):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            assert not config.vision_config.deepstack_visual_indexes, "Qwen3.5 consumes no DeepStack features"
            self.visual = Qwen3VLVisionModel(config.vision_config, quant_config=config.quant, prefix="visual")


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """The MoE releases with the vision tower; see Qwen3_5MoeForCausalLM."""


__all__ = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
]
