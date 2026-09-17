# Porting the DFlash2 Speculative-Draft Drafter from ninfer into FreeToken

Status: **READ-ONLY scoping / planning document.** No source files were modified.
This plan drives a chain of sequential, single-focus implementation agents. Every file path
and cited pattern is load-bearing — the agents rely on it and cannot re-scope.

**Constraint (read this first).** This machine has no GPU and is not the build box. Nothing can be
compiled or run here. The port must be *correct-by-construction*: it must match FreeToken's existing
idioms precisely so it compiles and runs on the user's real box (RTX 5090, sm_120a, Windows) without
local testing. The strategy below is therefore **reference-PyTorch-first**: the first milestone
implements the entire DFlash2 drafter + draft→verify loop in pure PyTorch on top of FreeToken's
existing, already-proven primitives (bf16 `LinearReplicated`, plain `RMSNorm`, Triton RoPE, the
target model itself). **No new CUDA or Triton kernel is required for the first milestone.** Hot ops
are converted to Triton/AOT CUDA only in later, self-contained milestones.

---

## 0. What "DFlash2" is (the source-of-truth math)

From ninfer `docs/maintainer/dflash.md` (authoritative), `src/models/qwen3_5/config.cpp`,
`src/models/qwen3_5/execution/draft.cpp`, `src/models/qwen3_5/program/storage/draft_context.{h,cpp}`,
and the op contracts in `include/ninfer/ops/`.

The drafter is an **optional component** of the Qwen3.5 target. It is a small masked-block
model that, given the target's committed hidden features and the unprocessed anchor token `a=x_F`
(frontier `F`), proposes `K` tokens in ONE non-causal forward over a `W=K+1` masked block
`[a, MASK, …, MASK]` at absolute positions `[F, F+1, …, F+K]`. The target then verifies the
proposals causally and remains the output authority.

### 0.1 Geometry (official 27B DFlash2 instance)

| Quantity | value |
|---|---|
| Hidden width H | 5120 (target hidden; the draft has no own width) |
| Draft layers L | 5 (all **sliding/local**, no full-attention layers) |
| Intermediate width | 17408 |
| Q heads / KV heads / head dim D | 32 / 8 / 128 |
| Target block IDs | `[5,19,33,47,61]` (5 residual taps) |
| Sliding-window scalar S (ring capacity) | 2048 |
| Norm eps / RoPE theta | 1e-6 / 1e7 |
| Dynamic convolution | 2 taps, group width 16 (G = H/16 = 320) |
| Candidate selector | top-16 (selector_top_k), rank R=256 |
| Mask token id | from config (`mask_token_id`) |

Runtime knobs (not weights): draft count `K ∈ 1..15`, concurrency `B ∈ 1..8`, `W=K+1`.
DFlash2 **always** computes the full `W`-column proposal even when only a prefix of length `P`
can be verified this round; the target verify runs on `Q=P+1` live columns.

### 0.2 The full math (translate this, not the C++)

**Target conditioning / context.** For processed position `t`, let `r_t^l` be the target block-l
residual output (after mixer+FFN residual add). Let `l ∈ target_layer_ids`.

```
s_t = concat(r_t^l for l in target_layer_ids)          # [num_taps * H]
c_t = plain_rmsnorm(W_feature @ s_t, context_norm)      # bias-free proj -> [H]
```

Per draft layer `l` the common `c_t` is projected to that layer's context K/V:
```
k_raw = W_context_key^l c_t
k_ctx = rope_1d(plain_head_rmsnorm(k_raw, key_norm^l), pos=t)   # per-head norm + 1-D split-half rope
v_ctx = W_context_value^l c_t
```
DFlash2's fused context materializer stores **K as BF16** and **V as `FP16_RNE(BF16(v_raw))`**.
Context prefill does NOT run the prompt through the draft residual/MLP stack. RoPE here is full-head
1-D split-half (NOT the target's 3-axis MRoPE).

**Query block + attention masks.** Draft input = target embedding of
`[a, MASK, …, MASK]` at positions `[F, …, F+K]`. Attention is **non-causal** over context + query
block K/V. Local (sliding) layer: `allowed(p_q,p_k) = |p_k - p_q| < S` (distance S−1 included, S
excluded), symmetric, plus every query sees every query-block row. (No full layers in 27B DFlash2.)

**Backbone (each of the 5 layers), DFlash2 form with dynamic grouped conv:**
```
n = plain_rmsnorm(x, input_norm)
(prepared_a, finish_delta_a) = attn_conv.prepare(n)         # rmsnorm+conv fused
q,k,v = query_key_value(prepared_a)                          # single fused QKV proj
q,k  = plain_head_rmsnorm_and_rope(q, k, positions)          # per-head norm + 1-D rope, in place
a    = noncausal_attn(q, concat(K_ctx,k), concat(V_ctx,v), scale=1/sqrt(D), window=S)
y    = x + attn_conv.finish(W_output @ a, finish_delta_a)    # W_out + conv + residual fused

m = plain_rmsnorm(y, post_attention_norm)
(prepared_m, finish_delta_m) = mlp_conv.prepare(m)
z = W_down ( SiLU(W_gate prepared_m) * (W_up prepared_m) )   # SwiGLU
x_next = y + mlp_conv.finish(z, finish_delta_m)
```
After all layers: `h = plain_rmsnorm(x, final_norm)`, then take the **mask columns only** (drop the
anchor column 0) → `hidden [H, K]` per request.

**Dynamic grouped conv** (2 sides {input,output}, 2 taps, group width 16). For normalized input
`X[b,i,g,j]`, `g∈[0,G)`, `j∈[0,16)`:
```
d[s,t,g,i,b] = sum_h kernel_projection[s,t,g, :H] @ n[:,i,b]        # [2,2,G,H] -> [2,2,G,i,b]
input-side (prepare):
  prepared[h,i,b] = (base[h,0,0]+d[0,0,g,i,b]) * n[h,i,b]
                 + (i>0) * (base[h,1,0]+d[0,1,g,i,b]) * n[h,i-1,b]
  finish_delta[g,0:2,i,b] = d[1,:,g,i,b]      # the output-side delta, kept from prepare
output-side (finish, applied to z = W @ branch_out):
  residual[h,i,b] += (base[h,0,1]+finish_delta[g,0,i,b]) * z[h,i,b]
                    + (i>0) * (base[h,1,1]+finish_delta[g,1,i,b]) * z[h,i-1,b]
```
`base_kernel` is checkpoint-stored `[side=2, tap=2, H]` (BF16); the runtime views it as
`[H, tap, side]`. `kernel_projection` is `[2*2*G, H] = [1280, H]` (BF16 in the reference; the
ninfer production kernel uses Q8 but that is a private internal detail — the oracle is FP64 over
the represented inputs). Position 0 has **no** preceding tap. No cross-request/round history.

**Candidate selector.** For each mask hidden `h_i`, the head gives stable **top-16** unary scores
`u_i [16]` and global token ids `C_i [16]` (FP32 scores; codebooks BF16). Then a **left-to-right
conditional walk** (predecessor = the actually selected token, anchor for i=0):
```
g_i = W_hidden @ h_i                                             # [R,H] -> [R]
pred(i,p) = anchor (i=0) else C_(i-1)[p]
E_i[p,c] = u_i[c] + sum_r W_pred[pred(i,p), r] * g_i[r] * W_succ[C_i[c], r]
temperature<=0:  j_i = lowest-rank argmax_c E_i[j_(i-1), c];  q_i = onehot(j_i)
temperature>0 :  q_i = softmax_c(E_i[j_(i-1), c] / temperature); j_i ~ q_i  (RNG below)
d_(i+1) = C_i[j_i]   (this becomes the next predecessor)
```
Retain the actual FP32 `q_i [16,K]` and global ids for target correction.
RNG counter key = `(request_seed, F+i, purpose=DFlash2Proposal)`. Ties: **lowest rank** wins.
Codebooks are `[vocab, R]`; `W_hidden` is `[R, H]`.

**Target verify / acceptance.** Verify block = `[a, d_1, …, d_P]` at positions `[F, …, F+P]`
(`P = current_extents[b]`, the verifiable prefix, `0<=P<=K`). The target runs causally over it,
retaining GDN records + target features; `p_i` = target distribution at column i.
Greedy: accept the longest target-argmax prefix; emit correction at first mismatch, or the bonus
from `p_P` on all-accept. Temperature>0: accept `d_(i+1)` w.p. `min(1, p_i[d]/q_i[d])`; on first
rejection sample `~ max(p_i-q_i,0)` normalized; on all-accept sample bonus `~ p_P`.

**Committed prefix / publication.** If `A` accepted, licensed output
`y = [d_1..d_A, correction_or_bonus]`, length `L=A+1`. Frontend picks final prefix `N ∈ [0,L]`
(EOS/stop/budget). Commit `N` input rows `[a, d_1..d_(N-1)]`, publish `N` outputs; continuation
hidden comes from verify column `N-1`; `y_N` becomes the next anchor. First `N` columns of captured
target features become **pending context work** (context may lag target by `N` until materialized).

### 0.3 Drafter weights (from `weights.h` + `load/dflash.cpp` + `load/dflash2.cpp`)

Physical checkpoint names live under the `dflash2/` component prefix (see §4.1 for the exact HF
naming caveat). The logical set (per 27B DFlash2):

| logical name | shape | kind |
|---|---|---|
| `feature_projection` | `[H, num_taps*H] = [5120, 25600]` | param (bf16) |
| `context_norm`, `final_norm` | `[H]` | direct bf16 |
| per layer i: `input_norm`, `post_attention_norm` | `[H]` | direct bf16 |
| `attention/query` | `[32*128, H] = [4096, 5120]` | param |
| `attention/key`, `attention/value` | `[8*128, H] = [1024, 5120]` | param |
| `attention/context_key`, `attention/context_value` | `[1024, 5120]` | param |
| `attention/query_norm`, `attention/key_norm` | `[128]` | direct bf16 |
| `attention/output` | `[H, 4096] = [5120, 4096]` | param |
| `mlp/gate`, `mlp/up` | `[17408, H]` | param |
| `mlp/down` | `[H, 17408]` | param |
| `attention_conv/base_kernel`, `mlp_conv/base_kernel` | `[2, 2, H]` | direct bf16 |
| `attention_conv/kernel_projection`, `mlp_conv/kernel_projection` | `[1280, H]` | param |
| `candidate_selector/hidden_projection` | `[256, H]` | param |
| `candidate_selector/predecessor_codebook`, `successor_codebook` | `[vocab, 256]` | direct bf16 |

The drafter reuses the **target's** `token_embedding` and `output_head` (they are not stored in the
drafter checkpoint; `weights.token_embedding = weights.output_head` in the target).

### 0.4 Storage

Five local draft layers keep K/V in **cyclic rings**, capacity `S=2048`; absolute position `p`
maps to slot `p mod S`; live interval `[max(0, F_ctx - S), F_ctx)`. K BF16, V FP16 (2 bytes each).
Per-state-image payload ≈ `5 * (2+2) * 8 * 128 * 2048 = 40 MiB`. There is **no full-attention pool**
in 27B DFlash2. Query-block K/V, candidate paths, `q`, verify results are round scratch. Pending
features survive commit until context catch-up.

---

## 1. Component map (DFlash2 piece → FreeToken file → idiom to copy)

Paths are absolute under `C:/Users/MUKKES~1/AppData/Local/Temp/freetoken-repo/`.

| # | DFlash2 piece | FreeToken file (create/modify) | FreeToken pattern to copy |
|---|---|---|---|
| 1 | Drafter config parse | **modify** `python/freetoken/models/qwen3_5_moe/config.py`; **modify** `python/freetoken/models/config.py` (new `ModelConfig` field) | `parse_config` in `qwen3_5_moe/config.py` (reads HF `text_config`); opaque-payload pattern `ModelConfig.dsv4_args/glm5_args/m3_args` in `models/config.py` |
| 2 | Drafter model + layers (feature proj, 5 draft blocks, final norm, selector) | **create** `python/freetoken/models/qwen3_5_moe/dflash2.py` | `Qwen3_5DecoderLayer`/`Qwen3_5Model` in `qwen3_5_moe/model.py` (BaseOP, `OPList`, residual-stream norms); `GatedMLP` in `models/blocks.py` |
| 3 | Drafter linear layers | inside `dflash2.py` | `LinearReplicated` in `layers/linear.py`; bf16 default `layers/quantization/linear/unquantized.py` (`F.linear`) |
| 4 | Drafter plain RMSNorm | inside `dflash2.py` | `RMSNorm` (plain) in `layers/norm.py` (NOT Gemma); kernel `kernel/triton/norm.py::rmsnorm` |
| 5 | Drafter RoPE (1-D split-half) + per-head norm | inside `dflash2.py` | `get_rope`/`Qwen3_5Attention._project` in `qwen3_5_moe/attention.py`; `kernel/triton/rope.py` |
| 6 | Drafter weight reader (external `z-lab/…-DFlash2` safetensors) | **modify** `python/freetoken/models/qwen3_5_moe/weight.py` (new `iter_dflash2_weights`, wire into `iter_weights`) | `_DenseReader` / `_iter_shards` in `weight.py`; `ShardReader` + `iter_weight_files` in `models/loader.py` |
| 7 | Context materializer (feature→c→per-layer ring K/V) | **create** `python/freetoken/models/qwen3_5_moe/dflash2_ops.py` (ref) | math from ninfer `context_kv_materialize.h`; compose `LinearReplicated` + `RMSNorm` + RoPE + a ring-scatter (PyTorch) |
| 8 | Dynamic grouped conv (prepare/finish) | **create** `dflash2_ops.py` (ref) | math from `dynamic_grouped_conv.h`; PyTorch `F.linear` + `unfold`/`einsum` |
| 9 | Non-causal sliding-window + full attention over ring+block | **create** `dflash2_ops.py` (ref) | math from `sliding_window_attention.h` / `softmax_attention.h`; PyTorch masked `F.scaled_dot_product_attention` or manual matmul; (later Triton, cf. `kernel/triton/attention.py`) |
| 10 | Candidate selector (top-16 + conditional walk) | **create** `dflash2_ops.py` (ref) | math from `candidate_selector.h`; PyTorch `torch.topk` + `einsum` + `torch.argmax`/`softmax` + `torch.rand` |
| 11 | Verify/accept (sparse draft acceptance) | **create** `dflash2_ops.py` (ref) | math from `speculative_round.h::speculative_accept_sparse_drafts`; PyTorch |
| 12 | Ring/paged KV storage for drafter | **create** `python/freetoken/kvcache/dflash2_ring_pool.py` (or a small buffer in `dflash2.py`) | ring/slot idiom of `kvcache/linear_state_pool.py` and `kvcache/swa_radix_cache.py` (per-request slot, `p mod S`) |
| 13 | draft→verify round driver | **create** `python/freetoken/models/qwen3_5_moe/dflash2_loop.py` | execution flow from `execution/draft.cpp::dflash_decode_batch_body`; target forward reuse via `model.forward()` + `Batch` |
| 14 | Engine/scheduler integration (enable + drive the loop) | **modify** `engine/engine.py`, `scheduler/scheduler.py` (decode step), `engine/config.py` (flags) | `Engine.forward_batch` + `Scheduler._process_last_data`; FLA spec-decode seam `kernel/fla/fused_recurrent.py` (`num_accepted_tokens`) for GDN writeback |
| 15 | Model wiring (attach drafter to `Qwen3_5ForCausalLM`) | **modify** `qwen3_5_moe/model.py` (build `self.draft` when `config.dflash2_args` present) | `Qwen3_5ForCausalLM.__init__` in `model.py` |

---

## 2. Ordered work breakdown (dependency-ordered, one agent each)

Each task is self-contained and testable-on-the-box. **[RPF] = reference-PyTorch-first (no new
kernel); [TRITON-LATER] / [AOT-LATER] = performance conversion, self-contained, later.**

**T1 — Drafter config.** [RPF]
- Modify `python/freetoken/models/config.py`: add frozen dataclass `DFlash2DraftConfig`
  (hidden width `H` from target, `num_layers`, `intermediate_size`, `num_q_heads=32`,
  `num_kv_heads=8`, `head_dim=128`, `target_layer_ids`, `sliding_window` S, `mask_token_id`,
  `conv_kernel_size=2`, `conv_group_size=16`, `selector_rank=256`, `selector_top_k=16`,
  `rms_norm_eps=1e-6`, `rope_theta=1e7`) and a `ModelConfig.dflash2_args: Any|None = None` field
  (copy the `dsv4_args`/`glm5_args` opaque-payload idiom, lines ~320-339).
- Modify `qwen3_5_moe/config.py::parse_config`: read the HF draft config block (the DFlash2
  component config; see §4.1 naming caveat) and populate `dflash2_args`. Default `None` when
  absent so every existing checkpoint loads unchanged.
- ninfer source: `src/models/qwen3_5/config.cpp::draft()` + `DFlash2Config`.
- **Done:** `parse_config` returns a `ModelConfig` whose `dflash2_args` matches the 27B geometry
  in §0.1 when the draft config is present; `None` otherwise. No model built yet.

**T2 — Drafter weight reader.** [RPF]
- Modify `qwen3_5_moe/weight.py`: add `iter_dflash2_weights(model_path, draft_path, device)`
  reading the external DFlash2 safetensors (a separate HF repo/dir, e.g. `z-lab/Qwen3.6-27B-DFlash2`),
  renaming to FreeToken state-dict keys **prefixed `dflash2.`** (e.g. `dflash2.feature_projection`,
  `dflash2.layers.0.attention.query`, `dflash2.layers.0.attention_conv.base_kernel`,
  `dflash2.candidate_selector.predecessor_codebook`). Transpose `base_kernel` `[2,2,H]→[H,2,2]`
  (checkpoint `[side,tap,channel]` → runtime `[channel,tap,side]`). Wire into `iter_weights` so the
  drafter tensors are yielded alongside the target when a draft path is configured.
- Pattern: `_DenseReader`/`_iter_shards` in the same file; `ShardReader`/`iter_weight_files` in
  `models/loader.py`. Reuse `_rename`-style key mapping; skip target-embedding/output-head (they
  come from the target).
- ninfer source: `weights.h` + `load/dflash.cpp` + `load/dflash2.cpp`.
- **Done:** `iter_weights(..., include_draft=True)` yields every tensor in the §0.3 table with the
  `dflash2.` prefix, correct shapes/dtypes, `base_kernel` transposed. (Requires the actual DFlash2
  checkpoint; see §5 open question O1 for key-name confirmation.)

**T3 — Drafter model + layers (forward, no KV, no selector).** [RPF]
- Create `qwen3_5_moe/dflash2.py`: `DFlash2DraftModel(BaseOP)` holding `feature_projection`
  (`LinearReplicated(H, num_taps*H)`), `context_norm`/`final_norm` (`RMSNorm`), and an
  `OPList` of `DFlash2Layer`. Each `DFlash2Layer` holds `input_norm`/`post_attention_norm`
  (`RMSNorm`), `qkv` (`LinearReplicated(H, 4096+1024+1024)`), `context_key`/`context_value`
  (`LinearReplicated(H,1024)` each), `q_norm`/`k_norm` (`RMSNorm(128)` or plain per-head),
  `o_proj` (`LinearReplicated(4096,H)`), `gate`/`up`/`down` (`LinearReplicated` SwiGLU, cf.
  `GatedMLP` in `models/blocks.py`), and `attn_conv`/`mlp_conv` (`DynamicGroupedConv` from T5).
- `forward(x, positions)` runs the §0.2 backbone (no context yet; query-block attention only,
  window over the block). Use plain `RMSNorm` (NOT Gemma), 1-D split-half RoPE (`get_rope`,
  theta=1e7).
- Pattern: `Qwen3_5DecoderLayer`/`Qwen3_5Model` in `model.py` (residual-stream, `OPList`,
  `nvtx_annotate`); `LinearReplicated` in `layers/linear.py`.
- **Done:** `DFlash2DraftModel.forward` runs on dummy bf16 weights for a `W=K+1` block and returns
  the mask hidden columns `[H, K]`; `state_dict()` keys match the `dflash2.` names from T2.

**T4 — Context materializer + ring storage.** [RPF]
- Create `python/freetoken/kvcache/dflash2_ring_pool.py`: per-request ring `K,V` slots, capacity
  S, slot = `p mod S`, K bf16 / V fp16; per-layer planes (5). Copy the per-request slot idiom from
  `kvcache/linear_state_pool.py` (or keep a simple preallocated buffer tensor inside `dflash2.py`
  for the first pass — acceptable, 40 MiB/image).
- Add `DFlash2DraftModel.materialize_context(features, positions)`: compute `c = rmsnorm(feature_proj(concat r^l), context_norm)`; per layer `k_raw=context_key(c)`,
  `k=rmsnorm_head(k_raw,key_norm)`, `k=rope1d(k,pos)`, `v=context_value(c)`, scatter into the ring
  at `pos mod S`. **K bf16, V `fp16`** (round `v_raw` to bf16 then to fp16, per §0.2).
- ninfer source: `context_kv_materialize.h` (exact K/V storage + rope) + `draft.cpp::append_context_impl`
  (feature projection + `rmsnorm` + per-layer proj).
- **Done:** after materializing a context of length L, the ring holds the correct K/V at
  `p mod S` for all `p<L`; a unit check against a manual PyTorch reference matches in fp32.

**T5 — Dynamic grouped conv.** [RPF]
- Create `DynamicGroupedConv` in `dflash2_ops.py` (or `dflash2.py`): `prepare(n)->(prepared, finish_delta)`
  and `finish(z, finish_delta)->out` implementing §0.2 exactly. `kernel_projection [1280,H]`
  (view rows as `[side,tap,group]`), `base_kernel [H,2,2]` (channel,tap,side). Use `F.linear`
  for `d` and `einsum`/broadcast for the conv; **position 0 has no preceding tap**; `finish_delta
  = d[1,:,:]` from prepare (do NOT re-project).
- ninfer source: `dynamic_grouped_conv.h` (both ops' exact math + shapes).
- **Done:** `prepare`/`finish` match a naive FP64 reference on random bf16 inputs (max-abs error
  within bf16 rounding) for `W=2..16, B=1..8`.

**T6 — Non-causal sliding-window + full attention.** [RPF]
- Create `dflash2_ops.py::draft_attention(q, k, v, positions, ring, window=S, scale)`: build the
  visible key set = ring context `p` with `|p - p_q| < S` (symmetric; include the query-block rows)
  + query-block K/V; compute `softmax(scale * q·k)·v`. Reference: PyTorch — gather the visible
  ring rows per query position into a dense `[D, Hkv, Tvis, B]`, concatenate the query K/V, mask
  non-visible with `-inf`, then `F.scaled_dot_product_attention` (or manual `matmul`+`softmax` for
  exactness during validation). Full (non-causal) = all populated context + block.
- ninfer source: `sliding_window_attention.h` + `softmax_attention.h` (visibility + oracle).
- **Done:** for a random ring+block, output matches a naive FP64 reference (stable softmax over
  exactly the visible set) within bf16 tolerance.

**T7 — Candidate selector.** [RPF]
- Create `dflash2_ops.py::candidate_selector(hidden, output_head, W_hidden, pred_cb, succ_cb,
  anchors, frontiers, seed)`: (a) `scores, C = topk(softmax? NO — raw logits over public vocab)`.
  The unary top-16 is over the **raw** proposal-head logits (no temperature/penalty/top-p):
  `topk` → `u [16,K]` (FP32 scores) and `C [16,K]` (global ids). (b) `g = W_hidden @ h [R,K]`.
  (c) conditional walk per §0.2: for i=0..K-1, `E[p,c] = u[c] + (pred_cb[pred] ⊙ g) · (succ_cb[C[c]])`
  via `einsum`; greedy = **lowest-rank** argmax; temperature>0 = `softmax(E/tau)` + `torch.rand`
  draw with a **deterministic per-(seed, F+i) generator** (`torch.Generator` seeded by
  `hash((seed, F+i, purpose))`). Return `drafts [K]`, `proposal_q [16,K]` (FP32).
- ninfer source: `candidate_selector.h` (exact E formula, tie-break "lowest rank", RNG counter key).
- **Done:** greedy walk matches a manual reference exactly (integer ids); stochastic path uses the
  documented RNG seed and reproduces across runs with the same seed.

**T8 — Verify/accept (sparse).** [RPF]
- Create `dflash2_ops.py::accept_sparse_drafts(target_logits, target_tokens, drafts, C, proposal_q,
  extents, frontiers, anchors, sampling)`: implement §0.2 acceptance. Greedy: longest
  penalty-adjusted-argmax prefix + correction/bonus. Temperature>0: accept w.p.
  `min(1, p[d]/q[d])`; rejection → sample `~ max(p-q,0)` normalized; all-accept → bonus `~ p_P`.
  Return `licensed_tokens [W]`, `licensed_counts`, `accepted [B]`, new anchor, delta length.
- ninfer source: `speculative_round.h::speculative_accept_sparse_drafts` + `dflash.md` "Target
  verification and committed prefix".
- **Done:** greedy acceptance matches a manual reference; stochastic acceptance conserves the
  processed-target distribution (sum of accepted-prob + correction-prob == 1) on random inputs.

**T9 — draft→verify round driver.** [RPF]
- Create `qwen3_5_moe/dflash2_loop.py::DFlash2Round`: given per-request anchors/frontiers/
  pending-target-features, run: (1) `materialize_context`; (2) draft `forward` over the `W` block;
  (3) `candidate_selector`; (4) build the verify block `[a,d_1..d_P]`; (5) **run the target model
  forward over the verify block** (a `Batch` of `W` tokens/req, varlen/cu_seqlens) capturing target
  logits+argmax and the per-column continuation hidden + GDN records; (6) `accept_sparse_drafts`;
  (7) commit `N` accepted rows to the target KV (via `Batch.out_loc`/page table) and set GDN final
  state to the accepted slot (the FLA `num_accepted_tokens` seam, `kernel/fla/fused_recurrent.py`);
  (8) capture continuation hidden + pending features for next round; return licensed tokens per
  request.
- ninfer source: `execution/draft.cpp::dflash_decode_batch_body` (the exact ordering) +
  `program/speculative/target_verification.cpp::target_verify_accept`.
- **Done:** one round advances the target frontier by `N∈[0,K+1]`, publishes the correct licensed
  tokens, and leaves target KV + GDN state consistent (verified by a state-diff vs a greedy
  non-speculative run over the same prompt).

**T10 — Engine/scheduler integration + flags.** [RPF]
- Modify `engine/config.py` (or `scheduler/config.py`): add `--spec dflash2`, `--dflash2-draft-tokens K`,
  `--dflash2-path` (the DFlash2 weights dir). Modify `engine/engine.py`: when enabled, build the
  drafter (already part of `self.model` via T15) and route decode steps through `DFlash2Round`
  instead of the single-token `forward_batch`+sample. Modify `scheduler/scheduler.py`:
  `_process_last_data` must consume **multiple** licensed tokens per request (loop over
  `licensed_counts`), honoring EOS/stop/budget to pick `N`; update `req` output + `token_pool`.
  Reuse the FLA `num_accepted_tokens` writeback so GDN state lands on the accepted column.
- **Done:** end-to-end generation with `--spec dflash2` runs; acceptance statistics are reported;
  output for a temperature-0 prompt is **identical** to greedy non-speculative decoding (the key
  correctness gate, since the target remains the authority).

**T11 — Drafter attention → Triton (perf).** [TRITON-LATER, self-contained]
- Replace the T6 PyTorch `draft_attention` with a Triton kernel over the small ring+block
  (T≤16, window≤4096, GQA 32/8). Copy the grid/stride/launch idiom from `kernel/triton/rope.py`
  and the attention structure from `kernel/triton/attention.py` (`_extend_attention_kernel`).
  Non-causal + symmetric window mask. Keep the PyTorch path as `FREETOKEN_DFLASH2_REF_ATTENTION=1`.
- **Done:** Triton attention matches the PyTorch reference on random ring+block within bf16 tol;
  speedup measured vs the reference on the box.

**T12 — Dynamic conv → Triton (perf).** [TRITON-LATER]
- Replace T5 with a Triton fused `rmsnorm+conv-prepare` and `linear+conv+add-finish`, copying the
  `kernel/triton/norm.py` (rmsnorm) + a `F.linear`-equivalent GEMM. Optional; only if T5 is a
  bottleneck (H=5120, W≤16 is small, so likely NOT a bottleneck — mark optional).
- **Done:** matches T5 reference; no regression.

**T13 — rmsnorm_rope → Triton fused (perf).** [TRITON-LATER]
- Fuse per-head RMSNorm + 1-D split-half RoPE into one kernel (cf. `kernel/triton/rope.py` +
  `kernel/triton/norm.py`). ninfer's `rmsnorm_rope.h` is the math. Optional.
- **Done:** matches the composed `rmsnorm`+`apply_rope` reference within bf16 tol.

**T14 — AOT CUDA hot path (optional, last).** [AOT-LATER]
- Only if Triton is insufficient: add `kernel/csrc/src/dflash2_*.cu` + a `KernelSpec` in
  `kernel/aot.py` (cf. `_store_spec`/`_radix_spec`), JIT via `load_aot`/`load_jit` in
  `kernel/utils.py`. This is the LAST resort; the reference + Triton milestones are expected to
  be sufficient for a 5090 single-request drafter.

**T15 — Model wiring.** [RPF]
- Modify `qwen3_5_moe/model.py`: in `Qwen3_5ForCausalLM.__init__`, if
  `config.dflash2_args is not None`, build `self.draft = DFlash2DraftModel(config, prefix="dflash2")`
  and expose `self.draft` for the round driver. Ensure `state_dict()`/`load_state_dict()` (BaseOP,
  `layers/base.py`) pick up the `dflash2.`-prefixed drafter keys from T2/T3.
- **Done:** `create_model` + `load_state_dict` loads a full DFlash2 checkpoint (target + drafter)
  without missing/unexpected-key errors.

---

## 3. Implementation strategy per op

| op | Milestone-1 choice | Why | Later option |
|---|---|---|---|
| Feature projection, QKV, output, gate/up/down, selector hidden proj | **bf16 `LinearReplicated`** (`F.linear`) | `QuantKind.NONE` default is `F.linear` (unquantized.py) — zero compile, matches target idiom, correct-by-construction. The drafter is small (W≤16) so bf16 GEMM is fine. | Quantized linear if the drafter checkpoint ships quantized (see O2). |
| Plain RMSNorm (input/post/context/final) | **existing `RMSNorm`** (`kernel/triton/norm.py::rmsnorm`) | Already present, proven, bf16. Drafter uses PLAIN rms (NOT Gemma `(1+w)`). | — |
| Per-head RMSNorm + 1-D RoPE | **compose `rmsnorm` + `apply_rope_with_cos_sin_cache_inplace`** | Both exist; avoids a new fused kernel. theta=1e7, full-head split-half. | T13 fused Triton. |
| Context materializer (proj+norm+rope+ring scatter) | **PyTorch composition** | Thin wrapper over the above; the ring scatter is a `index_put_`/gather. | AOT `context_kv_materialize` if prefill-bound. |
| Dynamic grouped conv | **PyTorch** (`F.linear` + `einsum`, position-0 no-tap) | Small (H=5120, W≤16, G=320, 2 taps); trivially correct in PyTorch. T5 is the correctness anchor. | T12 Triton (optional). |
| Non-causal sliding-window + full attention | **PyTorch masked SDPA** (T6) | T≤16 queries, ≤ S+W visible keys; PyTorch is simple + exact for validation. This is the one op most likely to need a kernel for speed. | T11 Triton (primary perf win). |
| Candidate selector (top-16 + conditional walk) | **PyTorch** (`topk`+`einsum`+`argmax`/`softmax`+`Generator`) | Sequential walk over K≤15 steps of 16 candidates — tiny; must be exactly reproducible (deterministic `Generator`). | AOT `candidate_selector` only if RNG parity with a reference is required (see O3). |
| Verify/accept | **PyTorch** | Elementwise over P≤K columns; must reproduce rejection sampling exactly. | AOT `speculative_accept_sparse_drafts` for RNG parity. |
| Ring/paged KV storage | **preallocated bf16/fp16 buffer + `p mod S` index** | 40 MiB/image; a plain tensor buffer is simplest and correct. | A pool class if many concurrent requests (B≤8) need slot management. |
| Verify-step target forward | **reuse `model.forward()`** over a verify `Batch` | The target model + attention backend + GDN already handle varlen blocks; the FLA `num_accepted_tokens` seam already writes GDN state to the accepted slot. No new target kernels. | — |

**Rationale for reference-PyTorch-first overall:** (a) no compile step on a box we can't reach;
(b) every op composes primitives FreeToken already ships and already validates, so the risk of a
silent kernel bug is minimized; (c) the drafter's working set (W≤16, H=5120, 5 layers) is small
enough that PyTorch is acceptable for a first correct milestone; (d) the performance-critical
non-causal attention is isolated in T6→T11 so it converts cleanly later without touching the
correctness path.

---

## 4. Integration points (where the loop plugs in; hooks to reuse)

1. **Config/flags.** `engine/config.py` / `scheduler/config.py`: add `--spec dflash2`,
   `--dflash2-draft-tokens K` (1..15), `--dflash2-path DIR`. Gated so the default path is unchanged.
2. **Model build.** `engine/engine.py::__init__` builds `self.model = create_model(...)` then
   `self.model.load_state_dict(...)`. The drafter is a sub-`BaseOP` of `self.model` (T15), so it is
   built + loaded with no engine change beyond the config. `finalize_quant(self.model)` (T3's
   `LinearReplicated`s) is already walked by the existing `finalize_quant` (`layers/quantization/
   method.py`), so no new finalize path is needed.
3. **Decode-step driver.** Today `scheduler/scheduler.py::_forward` → `engine.forward_batch(batch,
   args)` → one `model.forward()` + one sample → `_process_last_data` consumes ONE token/req.
   For DFlash2, when enabled, the decode branch instead calls `DFlash2Round.step(decode_batch)`
   (T9) which returns per-request **licensed tokens** (`[licensed_counts[b]]`) + counts.
   `_process_last_data` then loops over those tokens (EOS/stop/budget → choose `N`), appending them
   to `req` and updating `token_pool`. This is the single behavioral change to the scheduler.
4. **Target verify.** The round reuses `self.model.forward()` over a **verify Batch**
   (`phase` = a small multi-token decode/prefill with `cu_seqlens`/`out_loc` pointing the `W`
   verify tokens into the target KV pages). The attention backend (paged) and GDN (varlen)
   already process such blocks. Reuse the **FLA `num_accepted_tokens`** seam
   (`kernel/fla/fused_recurrent.py::IS_SPEC_DECODING`) so each GDN layer's final state is written
   to the accepted slot (`num_accepted_tokens-1` via `ssm_state_indices`) rather than the last
   verify column — this is exactly the "Fold uses records from this physical verify block" rule in
   `dflash.md`.
5. **Continuation hidden + pending features.** The round captures the verify column `N-1` hidden
   (continuation anchor feature) and the first `N` columns' target residual features (pending
   context) and stores them per-request for the next round (cf. `DFlashPersistentState`
   `pending_features` in `draft_context.h`). A per-request host/GPU scratch buffer suffices for the
   first milestone.
6. **Drafter KV lifetime.** The ring is per-request (slot) and must be reset when a request is
   admitted/finished (hook into the same places the GDN `LinearStatePool` slots are assigned/freed
   in `engine.engine` + `scheduler.cache.py`).

---

## 5. Risks & open questions

**R1 — Numerical precision (highest risk, unverifiable here).**
- **BF16/FP16 casts:** context K is BF16, V is `FP16_RNE(BF16(v_raw))` — the double-round
  (bf16 then fp16) must be reproduced, not a single fp16 round. The draft attention reads FP16 V
  from the ring. Get the storage dtype wrong and every proposal drifts. Mitigation: T4/T6 validate
  against an FP64 oracle with the *exact* storage roundings on the box.
- **RNG tie-breaking:** the greedy selector tie-break is **lowest candidate rank** (not lowest
  token id); the verify-accept argmax tie-break is **lower token id** (per `speculative_round.h`).
  These differ and both must be exact.
- **RNG seeding for the candidate selector:** ninfer keys on `(request_seed, F+i, purpose=
  DFlash2Proposal)`; FreeToken has no identical PRNG. For a first milestone, **greedy (temperature
  0) is fully deterministic and needs no RNG parity** — ship greedy first. Stochastic parity
  requires porting ninfer's counter-based PRNG (O3).
- **Proposal-head top-16 is over RAW logits** (no softmax/temperature/penalty), while the verify
  target distributions DO apply penalties/temperature/filters. Do not conflate the two.

**R2 — Dynamic grouped conv.** The fused `prepare` emits the *input-side* prepared input AND the
*output-side* `finish_delta` (which is `d[1,:,:]`, the output-side delta, computed ONCE from the
normalized input and reused on the branch output — it is **not** recomputed from the branch
output). A common porting bug is re-projecting the delta. Position 0 must skip the tap-1 term.
Base-kernel axis order (`[side,tap,channel]` in checkpoint → `[channel,tap,side]` runtime) must be
transposed at load (T2). Validate T5 in FP64 before anything else.

**R3 — TP interaction.** `qwen3_5_moe` weight loading already raises `NotImplementedError` for
`tp_size>1` (`weight.py`), and the default `DistributedInfo` is `(0,1)`. The DFlash2 drafter
therefore runs **TP=1 only**: use `LinearReplicated` (replicated weights) and per-rank ring
buffers; the draft/verify path **does not touch the TP communicator**. Do not introduce
`DistributedCommunicator` calls in the drafter. This matches the drafter being small and
single-GPU on the 5090.

**R4 — Cannot be verified without a reference model / GPU (blockers here).**
- No GPU → nothing compiles or runs locally; all "done" checks are deferred to the box.
- The **target-verify acceptance** can only be validated by diffing DFlash2-temperature-0 output
  against **greedy non-speculative decoding** of the same prompt (they MUST match, since the target
  is the authority). This requires a working non-spec baseline on the box.
- **Candidate-selector / stochastic RNG parity** with ninfer cannot be confirmed without a
  ninfer reference trace; ship greedy first.
- **Weight key names + dtype of the external DFlash2 checkpoint** are unknown until the actual
  `z-lab/…-DFlash2` safetensors are inspected (O1/O2).

**O1 — Drafter checkpoint naming/format (must confirm on the box).** The DFlash2 weights are a
**separate HF repo** (`z-lab/Qwen3.6-*-DFlash2`), not in the target checkpoint. The logical tensor
set is fixed (§0.3) but the exact safetensors **key names** (does the repo use a `dflash2/` prefix?
`model.` root? nested `layers.N.`?) and whether the projection weights are **bf16 or quantized**
(Q8/NVFP4 in the ninfer artifact) must be confirmed by reading the repo's
`model.safetensors.index.json` / a shard header on the box before finalizing T2. The reader (T2)
should be written against the *actual* keys; if quantized, dequant to bf16 at load (cf.
`_dequant` in `weight.py`) for the reference path.

**O2 — Drafter quantization.** The ninfer artifact stores many drafter projections as
Q4/Q5/Q8/NVFP4 (see the artifact-manifest `formats`). The **reference path should serve the
drafter in bf16** (dequant at load) to keep it correct-by-construction; quantized drafter linears
are a later optimization only.

**O3 — PRNG parity.** Whether stochastic sampling must bit-match ninfer (for reproducibility
claims) or only be statistically correct. Default: statistically correct with a documented,
seeded `torch.Generator`; greedy (temp 0) is exact and needs no parity.

**O4 — Draft-config source.** The HF config block for the drafter (target_layer_ids, S, conv dims,
selector rank/top-k, mask_token_id, rope theta) — confirm the exact field names in the DFlash2
component's `config.json` on the box (ninfer reads them from a `dflash_config` sub-object,
`config.cpp::draft`). T1 is parameterized on these names.

**O5 — Attention memory/speed of the PyTorch path.** T6 materializes up to `S+W` visible keys ×
32 heads per query row. For S=2048/4096 and W≤16 this is fine in bf16 but is the prime candidate
for the T11 Triton conversion; keep a memory ceiling (clamp the visible window) in the reference.

---

## 6. Build + smoke-test checklist for the real box (RTX 5090, Windows)

**6.1 Environment (once).**
```
cd <freetoken-repo>
uv venv && activate                          # Windows: .venv\Scripts\activate
uv pip install -e ".[accel]"
python setup.py build_ext --inplace          # builds only pinned_tensor/cpu_moe (NOT the AOT CUDA kernels)
```
- AOT CUDA kernels (only relevant for T11+; none for the RPF milestone) JIT-compile on first use
  via tvm-ffi and need a CUDA 13 toolkit with `nvcc` on `PATH` (`kernel/utils.py::check_nvcc_matches_torch`).
  Triton kernels JIT on first use. The RPF milestone (T1–T10, T15) adds **no** compiled extension.

**6.2 Enabling the DFlash2 drafter.**
```
# entry point is the `ft` console script (freetoken.cli:main); `serve` starts the API server.
# target model + separate DFlash2 drafter weight dir.
ft serve --model <Qwen3.6-27B> \
         --spec dflash2 \
         --dflash2-draft-tokens 7 \
         --dflash2-path <z-lab/Qwen3.6-27B-DFlash2>
```
- `--model` (alias `--model-path`) is the target checkpoint; `--dflash2-path` is the SEPARATE
  drafter weight dir (the drafter reuses the target's embedding/output head, so those are not in
  the drafter dir). The drafter builds only when the target checkpoint config carries a DFlash2
  drafter geometry (`dflash2_args`) **and** `--spec dflash2` is passed (see `EngineConfig.dflash2_active`).
- Passing `--spec dflash2` to a drafter-less checkpoint is a safe no-op (the `dflash2` sub-module
  is never built and the model loads byte-for-byte unchanged).
- First run will load target + drafter weights (T2/T15); confirm no missing/unexpected-key errors
  in `load_state_dict` — the 91-key drafter contract (`_expected_dflash2_tensors`) must load clean.

**6.3 Minimal end-to-end smoke test.**
1. **Load check:** start the server with the flags above; it must reach "ready" (drafter built +
   loaded, ring allocated ~40 MiB/image).
2. **Greedy parity (the correctness gate):** with `temperature=0`, generate a fixed prompt
   (e.g. 128 tokens) **with** `--spec dflash2` and **without** (plain greedy). The token streams
   must be **identical** — the target is the output authority, so DFlash2 only changes speed,
   never the greedy output. This single check validates T4–T10 end-to-end.
3. **Acceptance stats:** log accepted/`K` per step; expect a stable mean (sanity: >1 accepted/step
   on a coherent prompt; a mean of ~0 indicates the drafter is broken or the ring is misaligned).
4. **Unit oracles (run on the box before the full e2e):**
   - T5 conv vs FP64 reference (random bf16, W=2..16, B=1..8).
   - T6 attention vs FP64 reference (random ring+block, window S).
   - T7 selector greedy walk vs manual reference (integer-exact).
   - T8 accept greedy vs manual reference.
   - T4 ring: materialize context length L, read back K/V at `p mod S`, compare to a manual PyTorch
     reference in fp32 (K bf16, V fp16 roundings exact).
5. **Diff target.** The reference to diff against is **FreeToken's own non-speculative greedy
   decode** of the same model+prompt (not ninfer). If a ninfer reference trace is available, use it
   to also check the *proposal distributions* (draft tokens + `proposal_q`) — but the *output*
   must match FreeToken greedy regardless.

**6.4 Regression guard.** Run the existing suite with the drafter **disabled** (default) to confirm
the port is inert: `uv run pytest tests/ -m "not slow"`. Add a new test under `tests/` mirroring
`models/qwen3_5_moe/` (per AGENTS.md) for the config parse (T1) and the conv/attention/selector
oracles (T5–T7).
