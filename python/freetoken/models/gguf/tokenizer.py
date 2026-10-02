"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF
config, which the gemma4 strict dataclass rejects (per-layer ``num_key_value_heads``
array). So we call the GGUF->fast tokenizer converter directly on the
``tokenizer.ggml.*`` metadata, bypassing config entirely.

transformers keys its GGUF converter registry by model architecture, so an
architecture its version does not know (``glm5next``) needs an explicit map onto a
registered converter of the same tokenization family -- ``qwen2``'s is a byte-level
BPE, which is what the GLM GGUFs carry (``tokenizer.ggml.model = gpt2`` +
``tokenizer.ggml.pre = glm4`` + a merges list). Candidates are probed against the
installed registry so a transformers version that gains a native converter for an
architecture is used automatically.
"""

from __future__ import annotations

from typing import Any

from .reader import gguf_architecture, load_gguf_metadata

# GGUF architecture -> preferred transformers GGUF tokenizer-converter key.
# glm5next: unsloth GLM-5.3 GGUFs are byte-level BPE (same family as qwen2's
# registered GGML converter); verified the tier round-trips through it.
_TOKENIZER_ARCH = {"gemma4": "gemma4_text", "glm5next": "qwen2"}


def _converter_arch(meta: dict[str, Any], arch: str) -> str:
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS as REG

    pre = meta.get("tokenizer.ggml.pre")
    candidates = [_TOKENIZER_ARCH.get(arch), pre, arch]
    for cand in candidates:
        if cand is not None and cand in REG:
            return cand
    raise ValueError(
        f"no GGUF tokenizer converter for architecture {arch!r}; transformers knows "
        f"{sorted(REG)} -- add a _TOKENIZER_ARCH entry for a compatible family"
    )


def _eot_token(meta: dict[str, Any]) -> str | None:
    """The GGUF's end-of-turn token string, if the metadata names one. llama.cpp's
    ``eot_token_id`` marks the end of a chat turn (GLM: the next turn's start marker,
    which is exactly where generation must halt)."""
    tokens = meta["tokenizer.ggml.tokens"]
    eot = meta.get("tokenizer.ggml.eot_token_id")
    if eot is not None and int(eot) < len(tokens):
        return tokens[int(eot)]
    return None


def load_gguf_tokenizer(model_path: str):
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.ggml import convert_gguf_tokenizer

    meta = load_gguf_metadata(model_path)
    arch = gguf_architecture(model_path)
    conv_arch = _converter_arch(meta, arch)
    tok_dict: dict[str, Any] = {
        k[len("tokenizer.ggml.") :]: v
        for k, v in meta.items()
        if k.startswith("tokenizer.ggml.")
    }
    fast, _extra = convert_gguf_tokenizer(conv_arch, tok_dict)

    tokens = tok_dict["tokens"]

    def tok_for(id_key: str, default: str) -> str:
        tid = meta.get(f"tokenizer.ggml.{id_key}")
        return tokens[int(tid)] if tid is not None and int(tid) < len(tokens) else default

    # Chat turns end with a turn-end marker; prefer it as eos so chat generation
    # halts (the formal stop token is also a stop id, see gguf_eos_token_ids). The
    # metadata's own eot_token_id wins; gemma4's GGUF lacks it, so keep its literal.
    turn_end = _eot_token(meta)
    if turn_end is None and arch == "gemma4":
        turn_tok = "<" + "turn" + "|>"  # name-style special, not chatml-shaped
        turn_end = turn_tok if turn_tok in tokens else None
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast,
        bos_token=tok_for("bos_token_id", "<bos>"),
        eos_token=turn_end or tok_for("eos_token_id", "<eos>"),
        unk_token=tok_for("unknown_token_id", "<unk>"),
        pad_token=tok_for("padding_token_id", "<pad>"),
    )
    chat_template = meta.get("tokenizer.chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


def gguf_eos_token_ids(model_path: str, tokenizer) -> set[int]:
    """Stop ids for GGUF generation: the formal stop token plus the chat turn-end
    (``eot_token_id``) and end-of-message (``eom_token_id``) markers when present."""
    meta = load_gguf_metadata(model_path)
    tokens = meta["tokenizer.ggml.tokens"]
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for key in ("eos_token_id", "eot_token_id", "eom_token_id"):
        tid = meta.get(f"tokenizer.ggml.{key}")
        if tid is not None and int(tid) < len(tokens):
            ids.add(int(tid))
    # gemma4's GGUF names its turn end without an eot id; look it up in the vocab
    # directly (convert_tokens_to_ids would map an absent name to <unk>, wrongly
    # adding it as a stop id).
    if gguf_architecture(model_path) == "gemma4":
        for name in ("<" + "eos" + ">", "<" + "turn" + "|>"):
            try:
                ids.add(tokens.index(name))
            except ValueError:
                pass
    return ids


__all__ = ["load_gguf_tokenizer", "gguf_eos_token_ids"]
