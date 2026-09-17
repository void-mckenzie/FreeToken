# DFlash2 port — verification audit (vs. ninfer ground truth)

Status: **6 of 6 components audited.** 1 **blocker** + 3 **minor** logical bugs found and
documented; the drafter math, selector, verify/accept, ring/materializer, and feature-capture
components otherwise **MATCH** the ninfer spec line-for-line.

Method: six sequential read-only audit agents, one component at a time, each comparing the
FreeToken port against the ninfer source line-by-line (with file:line on both sides) and
adversarially hunting for a computed-value / ordering / tie-break / dtype / geometry divergence.
The FreeToken port is intentionally **reference PyTorch** (no new CUDA/Triton kernel this
milestone), so kernel-implementation differences are expected and out of scope; only semantic /
math parity is in scope. Every blocker finding below was independently re-confirmed against the
FreeToken scheduler / weight sources (not just the agent's report).

Repo roots: ninfer = `.../Temp/ninfer` (ground truth), FreeToken = `.../Temp/freetoken-repo`
(package under `python/freetoken`).

## TL;DR

| # | Component | Verdict | Notes |
|---|-----------|---------|-------|
| V1 | Geometry + 91-key weight contract | **MATCH** | All 23 logical types / shapes align; `.weight` split correct |
| V2 | Drafter math core (RoPE / grouped conv / sliding attn) | **MATCH** | base_kernel side-orientation re-confirmed (no bug) |
| V3 | Context ring + K/V materializer | **MATCH** (math) + **minor bug** | V double-round, ctx-K/V derivation all correct; **bug M1** (slot-0) |
| V4 | Candidate selector | **MATCH** | codebook orientation, walk, lowest-rank tie-break all correct |
| V5 | Verify/accept + round protocol | **MATCH** | off-by-one, GDN-rewind column, frontier all correct |
| V6 | Target feature capture + scheduler integration | **BUG** (capture MATCH) | **bug B1** (blocker) + minor M2, M3 |

---

## B1 — BLOCKER — round-1 anchor reads the stale CPU stream under default overlap scheduling

**FreeToken:** `models/qwen3_5_moe/dflash2_loop.py:174-177` (`admit`), read against
`scheduler/scheduler.py:262,271,279,367-421,996-998,1007,1011`.
**ninfer:** `docs/maintainer/dflash.md:42-44` (the round anchor `a = x_F` is the committed
unprocessed token from committed target state, materialized *after* the commit exists).

`admit` computes the round-1 anchor from the **CPU** request stream:

```python
F = req.cached_len                                   # committed prompt length (== P)
anchor = int(req.input_ids[F].item()) if F < req.input_ids.numel() else 0   # <-- BUG
```

Under FreeToken's **default overlap scheduling** (`DISABLE_OVERLAP_SCHEDULING` default False),
one scheduler iteration runs, in order:

1. `scheduler.py:262` `_schedule_next_batch()`
2. `scheduler.py:271` `_forward(forward_input)`  ← **the dflash2 `admit` fires here**
   (`scheduler.py:996-998`: `if req.dflash2_state is None: self.dflash2_driver.admit(req)`)
3. `scheduler.py:279` `_process_last_data(last_data)`  ← **the previous batch's sampled token is
   appended to the CPU stream here** (`scheduler.py:420-421`:
   `next_token = next_tokens_cpu[i]; req.append_host(next_token.unsqueeze(0))`)

For a freshly-admitted request the "previous batch" is its **prefill**. So in the iteration right
after the prefill, the first dflash2 decode round's `admit` (step 2) runs **before** the prefill's
first output token `t0` is drained into `req.input_ids` (step 3). At `admit` time
`req.input_ids` still holds only the `P` prompt tokens, so `numel() == F == P`, `F < numel` is
False, and **`anchor = 0`** (which is also `mask_token_id`) instead of `t0`.

**Consequences** (fires by default for any request with `output_len >= 2` in the common
single-request streaming case):
- The round-1 verify block is `[0, d_1..d_K]` instead of `[t0, d_1..d_K]` → wrong anchor column →
  wrong per-column target distributions `p_i` → wrong acceptance. This **breaks the greedy-parity
  correctness gate** (DFlash2 temp-0 output must equal non-speculative greedy).
- The target KV at position `F` stores token-0's K/V.
- The driver's `_commit` (`dflash2_loop.py:389-391`) appends the published tokens at CPU index `F`;
  the late prefill drain then appends `t0` after them, corrupting the committed stream order
  (`[prompt, pub…, t0]`) so `input_ids.numel() != device_len`.

**Why the normal path is immune:** the normal decode path sources its input token from the GPU
`token_pool` written at `scheduler.py:1011` and read at `scheduler.py:1007`
(`batch.input_ids = self.token_pool[input_mapping]`) — not from the CPU stream. The dflash2
`admit` introduced a *new* dependency on `req.input_ids` that the scheduler never guarantees at
`admit` time.

**Fix (recommended):** source the round-1 anchor from the **committed GPU state**, not the CPU
stream — i.e. read the token at position `F` from `token_pool[table_idx, F]` (where the prefill
wrote `t0` at `scheduler.py:1011`), mirroring how the normal decode path reads its input. Do not
fall back to `req.input_ids[F]` (stale under overlap) nor to `0`. Equivalently, defer the dflash2
`admit` to the `_process_last_data` drain point (step 3) where the CPU stream is current. The
round-2+ anchor (`dflash2_loop.py:398`, `st.anchor = committed[N-1]`) is taken from the driver's
own committed tokens and is unaffected.

---

## M1 — minor — DFlash2 ring reserves slot 0, but the driver indexes by `req.table_idx`

**FreeToken:** `kvcache/dflash2_ring_pool.py:88-91` (slot 0 = reserved padding sink; free-list
spans `1..N-1`) + `models/qwen3_5_moe/dflash2_loop.py:172` (`slot = req.table_idx`).
**Convention evidence:** `scheduler/table.py:7` (`_free_slots = list(range(max_running_reqs))` →
`allocate()` **can return 0**); `engine/engine.py:484` (the **dummy** row is `table_idx =
max_running_req`, the *last* index); `kernel/fla/fused_recurrent.py:128-135` (a real request can
key state by raw `table_idx == 0`; only negative ids are invalid).

The ring was modeled on `LinearStatePool` (`kvcache/linear_state_pool.py:91-96`), which *does*
reserve slot 0 — but that pool allocates live slots through its own `alloc()` (never returns 0)
and its slot space is **independent** of `table_idx`. The dflash2 ring does **not** use its
`alloc()`; the driver uses `req.table_idx` **directly** as the slot, and `table_idx == 0` *is* a
valid live request (the dummy is at the *last* index). So a live request with `table_idx == 0`
lands in the ring's "reserved" slot 0, and the ring does not reserve the actual dummy slot
(`max_running_req`).

**Impact:** no data corruption in the current path (the data path is slot-uniform and in-bounds —
the ring is sized `max_running_req + 1`, covering `table_idx 0..max_running_req`), and the
free-list / `padding_slot` / `num_free_slots` are dead code (the driver never calls `alloc()`).
But the reservation is **wrong** (it should reserve the dummy slot, not slot 0) and the
free-list accounting is misleading — a latent hazard if a future path starts using `alloc()`.

**Fix (recommended):** make the ring a plain per-`table_idx` store (drop the slot-0 reservation /
free-list, or reserve the actual dummy slot `max_running_req`), consistent with
`TableManager`'s `0..max_running_req-1` live + `max_running_req` dummy convention.

---

## M2 — minor — cold-start re-prefill omits the multimodal embedding merge

**FreeToken:** `models/qwen3_5_moe/dflash2_loop.py:563-582` (`_cold_start` builds a fresh
`Batch` with no `mm_embeds` and re-prefills `[0, F)` through `capture_dflash2_features`).
**ninfer:** `docs/maintainer/dflash.md:57-58` ("Multimodal inputs use the same target residual
capture **after Vision embedding replacement**").

For a multimodal prompt the captured `s_t` is wrong at vision positions, and the re-prefill
overwrites the already-committed target KV/GDN state at `[0, F)` with text-only values
(`out_loc` is the existing page-table row, `dflash2_loop.py:571`). **Text-only prompts** (the
27B reference scope; the parity gate is text-only greedy) are idempotent and unaffected.

**Fix (recommended, or defer):** thread the request's `mm_embeds` into the `_cold_start` batch so
the vision embedding replacement is applied before capture; otherwise document DFlash2 as
text-only for this milestone and reject multimodal requests when `--spec dflash2` is set.

---

## M3 — minor — stop-string trimming happens after commit, not in the N-preview

**FreeToken:** `models/qwen3_5_moe/dflash2_loop.py:323-342` (`_choose_n` trims N by budget +
first-EOS only) + `scheduler/scheduler.py:381-385` (stop-string check post-commit in
`_process_last_data`).
**ninfer:** `docs/maintainer/dflash.md:258-260` ("Frontend preview chooses the final prefix length
N, `0 <= N <= L`, accounting for EOS, **stop**, budget and cancellation" — i.e. *before* commit).

Client-visible output is still trimmed by `matched_stop` and the request finishes, so the only
effect is committing up to `K+1` extra tokens past the stop boundary into a finishing request's
KV/state (freed on finish). Already acknowledged in-driver
(`dflash2_loop.py:327-331` `TODO(ON-BOX)`).

**Fix (recommended):** move the stop-string / budget / cancellation preview into `_choose_n`
so N is finalized before the commit (matching ninfer's "frontend preview chooses N").

---

## At-risk items that were verified CORRECT (no bug)

These were the classic subtle bugs; each was explicitly checked against ninfer and confirmed
correct:

- **base_kernel side orientation** (V2): checkpoint `[side, tap, channel]` → loader
  `permute(2,1,0)` → `[channel, tap, side]` **preserves the side index** (`weight.py:448-450`);
  FreeToken prepare reads side 0 (input), finish reads side 1 (output)
  (`dflash2_ops.py:174-175,213-214`), matching ninfer (`dynamic_grouped_conv.h:35-36,87-88`).
  Input/output conv kernels are **not** swapped.
- **GDN rewind / continuation column** (V5): rewind is to verify column `N-1` (= `A`), and the
  vendored FLA kernel writes the intermediate state buffer **after** the recurrent-state update
  (`kernel/fla/fused_sigmoid_gating_recurrent.py:202→220`), so `spec_buf[li,req,N-1]` is the
  correct post-column continuation state.
- **verify off-by-one** (V5): `drafts[i]` is checked against `argmax[i]` (same index), with exactly
  `P+1 = K+1` verify columns — matching ninfer (`speculative_round.cuh:127`).
- **two distinct argmax tie-breaks** (V4/V5): the selector's greedy winner is the **lowest
  candidate rank** (`dflash2_ops.py:505`, verified against ninfer's dedicated tie-case test
  `test_candidate_selector.cpp:363-369`); the verify argmax is the **lower token id**
  (`dflash2_ops.py:592`). Correctly kept separate.
- **codebook orientation** (V4): `[vocab, R]`, indexed `[token, r]`, loaded with **no** transpose
  (`weight.py:447-456`); `pred_cb[pred_token]` and `succ_cb[C_i]` index rows = token ids.
- **conditional walk** (V4): threads the **actually-selected** token forward (`dflash2_ops.py:512-515`),
  anchor for `i == 0`.
- **V double-round** (V3): `v_raw.to(bfloat16).to(float16)` = `FP16_RNE(BF16(v_raw))`, stored fp16
  and upcast to fp32 in attention — the exact parity-critical rounding.
- **feature-capture tap point** (V6): `taps_by_layer[layer_id] = residual + x` is the **complete
  post-block residual** after both the attention and FFN residual adds (`model.py:194-204`), in
  `target_layer_ids` **recorded** order, snapshotted with a fresh `+` (safe from the next block's
  in-place `forward_add_residual`).
- **no-drafter regression guard** (V6): every dflash2 branch is gated on `dflash2_driver is not
  None` / `batch.is_dflash2` / `req.dflash2_state`; the `dflash2` sub-module attribute exists only
  when `config.dflash2_args` is set; the prefill path is untouched — the port is inert for
  drafter-less checkpoints.

## Assumptions / on-box residuals (not resolvable without the GPU + the real checkpoint)

- **Physical checkpoint key naming (O1):** the weight reader expects the *bare* logical
  names (`dflash2/layers/0/attention/query`, …). If the real draft repo stores `.weight`-suffixed
  or double-prefixed keys, `load_state_dict` fails **loudly** (`weight.py:436-441`), not silently.
- **Draft quantization (O2):** `_dflash2_to_bf16` raises `NotImplementedError` on a quantized
  (Q8/NVFP4) draft checkpoint; a dequant-at-load path is needed if the 27B draft ships quantized.
- **Stochastic RNG draw parity (O3):** temp>0 uses a SplitMix64-style seed, not ninfer's counter
  PRNG — the *math* matches, the per-draw value may differ. Greedy (temp 0) is fully deterministic
  and needs no RNG.
- **Penalty-free verify:** the shipped temp-0 path uses the raw softmax argmax with no
  presence/frequency penalties, top-k/top-p, or temperature (matches ninfer's raw-greedy fast
  path; lossless). If the engine configures penalties, the verify would need to apply them
  (documented deferral, plan R1).
- **`TODO(ON-BOX)` seams** (`dflash2_loop.py:215,359,461,501`): varlen verify-batch attention/GDN
  metadata, GDN spec-verify + rewind write-back, and KV-scratch page-table accounting — the
  column indexing is correct per source but the runtime metadata still requires box validation.
- **Multimodal** (M2) and **greedy parity for a quantized non-tied lm_head** are deferred; the
  27B reference milestone is text-only with a tied head.
