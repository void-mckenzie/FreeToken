"""The DFlash2 draft -> verify round driver (plan T9, Agent 4b).

Given the target, the (Agent 1-3 / T3) :class:`DFlash2Drafter`, the per-request context ring
(:class:`DFlash2RingPool`) and FreeToken's KV / page-table / GDN state machine, this module
implements ONE speculative round per request, driven from the scheduler's decode step (T10). It is
the single home for all DFlash2-specific logic so it can be inspected and validated on the box in
isolation; the scheduler only feeds it the running decode requests and publishes the tokens it
returns.

The round (ninfer ``dflash.md`` "Execution flow"; plan §0.2 "Committed prefix / publication"), for a
request at committed frontier ``F`` (positions ``[0, F)`` resident in the target KV), anchor
``a = x_F`` (the unprocessed last-published token) and ring ``slot``::

    1. materialize the previous round's pending target features into the ring so the context
       reaches F.  First round (cold start): a re-prefill over the committed prompt [0, F) captures
       the prompt's ``s_t`` and fills the ring (clipped to the live window [F-S, F)).
    2. read the ring live window [max(0, F-S), F).
    3. build the query block x0 = embed([a, MASK, ..., MASK]) at positions [F, ..., F+K] (W=K+1).
    4. draft forward  -> h [W, H].
    5. select         -> drafts [K], proposal_q [top_k, K], candidates [top_k, K].
    6. target verify  -> ONE batched varlen "prefill" forward over [a, d_1..d_P] at [F, ..., F+P],
       via ``model.capture_dflash2_features()`` -> (s_t, final_hidden, logits).
    7. accept         -> accept_sparse_drafts (greedy ship-first; temp>0 is implemented but OFF).
    8. choose N       -> 0 <= N <= A+1 licensed prefix, trimmed by budget (+ first EOS); the target
       remains the output authority.
    9. commit N       -> advance the committed length by N and publish committed[:N]; fold the GDN
       recurrent state to column (N-1) (the ``num_accepted_tokens`` / rewind seam); the verify wrote
       W KV columns at [F, F+W) but only [F, F+N) is committed -- [F+N, F+W) is scratch the next
       round overwrites (the page table / committed frontier advance only to F+N).

GREEDY (temperature 0) is the default: fully deterministic (no RNG), and its output MUST equal
FreeToken's own non-speculative greedy decode of the same prompt (the correctness gate, plan §6.3).
The temperature>0 path reuses the same ops (already stochastic-capable) but is OFF by default; it
requires the box-validated RNG-seed wiring noted below.

THIS MODULE IS NOT TESTABLE HERE (no GPU / not the build box). Every seam coupled to FreeToken's
scheduler / KV / GDN state machine is marked ``TODO(ON-BOX)`` and must be validated first on the
RTX 5090 -- in particular the varlen verify-batch metadata, the GDN state rewind, the KV-scratch
overwrite, and the greedy parity check. No new global machinery is introduced; the driver reuses the
engine's stream / ctx / page table / GDN pool / attention backend and the existing FLA / GDN kernel
params.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn.functional as F

from freetoken.core import Batch, Req
from freetoken.models.qwen3_5_moe.dflash2_ops import accept_sparse_drafts

if TYPE_CHECKING:
    from freetoken.engine.engine import Engine
    from freetoken.kvcache.dflash2_ring_pool import DFlash2RingPool
    from freetoken.models.qwen3_5_moe.dflash2 import DFlash2Drafter
    from freetoken.models.qwen3_5_moe.model import Qwen3_5ForCausalLM
    from freetoken.scheduler.cache import CacheManager


# --------------------------------------------------------------------------- per-request state


@dataclass
class DFlash2ReqState:
    """Per-request DFlash2 round state (carried on ``Req.dflash2_state``; T9/T10).

    * ``slot``          -- the :class:`DFlash2RingPool` slot (mirrors ``Req.table_idx``).
    * ``frontier``      -- the committed KV length ``F`` (positions ``[0, F)`` resident); the next
      round's verify positions are ``[F, F+W)`` and its anchor is ``x_F``.
    * ``anchor``        -- ``a = x_F``: the unprocessed last-published token (the previous round's
      ``committed[N-1]``, or the prefill's first output on the first round).
    * ``alloc_frontier``-- the position to which the page-table row is ALREADY allocated (the KV
      scratch carried across rounds so a round allocates only the fresh committed growth; see the
      KV-scratch invariant in :meth:`DFlash2RoundDriver._commit`).
    * ``pending_features`` -- ``s_t`` ``[n, num_taps*H]`` bf16 for positions ``[frontier-n, F)``
      (the previous round's committed columns) to materialize into the ring this round. ``None``
      until the first commit (cold start handles the prompt).
    * ``pending_positions`` -- ``[n]`` int32 absolute positions of ``pending_features``.
    * ``seed``          -- the request seed (stochastic RNG key; unused for greedy).
    * ``initialized``   -- the ring holds its committed-prompt context (cold start done).
    """

    slot: int
    frontier: int
    anchor: int
    alloc_frontier: int
    pending_features: Optional[torch.Tensor] = None
    pending_positions: Optional[torch.Tensor] = None
    seed: int = 0
    initialized: bool = False


@dataclass
class DFlash2StepResult:
    """One request's round outcome (aligned to ``batch.reqs`` on a DFlash2 decode batch).

    ``published`` is a CPU int32 ``[N]`` (N == ``n_published``); the scheduler publishes one
    ``DetokenizeMsg`` per element and reads finish state from the request (already advanced by the
    driver's commit).
    """

    published: torch.Tensor
    n_published: int
    n_accepted: int  # A: accepted draft count (acceptance stats)


# --------------------------------------------------------------------------- round driver


class DFlash2RoundDriver:
    """Drives one DFlash2 speculative round over a batch of running decode requests.

    Constructed by the scheduler when DFlash2 is enabled for the model. The scheduler calls
    :meth:`admit` / :meth:`release` at the request's first decode step / free (mirroring the GDN
    slot lifecycle) and :meth:`step` for each decode batch (on the engine stream, inside
    ``_forward``). :meth:`step` returns ``(results, copy_done)``; the scheduler stores ``results``
    on the decode batch (``batch.dflash2_results``) and publishes them in ``_process_last_data``.
    """

    def __init__(
        self,
        engine: "Engine",
        model: "Qwen3_5ForCausalLM",
        drafter: "DFlash2Drafter",
        ring: "DFlash2RingPool",
        cache_manager: "CacheManager",
        token_pool: torch.Tensor,
        *,
        draft_count: int,
        mask_token_id: int,
        eos_token_ids: frozenset,
    ) -> None:
        self.engine = engine
        self.model = model
        self.drafter = drafter
        self.ring = ring
        self.cache_manager = cache_manager
        self.token_pool = token_pool
        self.device = engine.device
        self.K = int(draft_count)  # draft tokens proposed per round
        self.W = self.K + 1  # masked query-block width
        self.P = self.K  # verifiable prefix P (== K for the reference milestone; no budget trim)
        self.mask_token_id = int(mask_token_id)
        self.eos_token_ids = frozenset(eos_token_ids)
        # The target's raw output-head weight [vocab, H] (bf16, TP=1): shared by the selector's
        # top-16 and the verify's per-column target distributions so both match the target authority.
        self._head_weight = engine.dflash2_output_head_weight()
        # GDN geometry for the spec-verify intermediate-state buffer + rewind (None for a pure
        # full-attention model, where there is no GDN state to rewind).
        linear_group = engine.config.model_config.linear_attention_group()
        self._gdn_layer_ids: Tuple[int, ...] = tuple(linear_group.layer_ids) if linear_group else ()
        self._gdn_hv_k_v: Optional[Tuple[int, int, int]] = (
            (linear_group.num_value_heads, linear_group.key_head_dim, linear_group.value_head_dim)
            if linear_group
            else None
        )
        self._num_gdn_layers = len(self._gdn_layer_ids)

    # ------------------------------------------------------------------ slot lifecycle

    def admit(self, req: Req) -> DFlash2ReqState:
        """Allocate the request's ring slot and initialize its round state (first decode step).

        The ring slot mirrors ``Req.table_idx`` (range 0..max_running_req), so the scheduler's
        existing table_idx admit/free covers the ring slot's lifetime (plan §4.6). The committed
        prompt ``[0, F)`` is the cold-start context (materialized lazily on the first :meth:`step`);
        the anchor is the prefill's first output ``x_F``.
        """
        slot = req.table_idx
        self.ring.reset(slot)
        F = req.cached_len  # committed prompt length after the prefill
        if F < 1:
            F = max(1, req.input_ids.numel() - 1)
        anchor = int(req.input_ids[F].item()) if F < req.input_ids.numel() else 0
        state = DFlash2ReqState(
            slot=slot,
            frontier=F,
            anchor=anchor,
            alloc_frontier=F,  # page-table row already valid to the committed prompt length
            seed=int(req.uid),  # TODO(ON-BOX): the stochastic path needs a proper per-request RNG
            # seed; greedy (temp 0) is deterministic and needs none.
            initialized=False,
        )
        req.dflash2_state = state
        return state

    def release(self, req: Req) -> None:
        """Free the request's ring slot (finish / abort). Idempotent."""
        state = getattr(req, "dflash2_state", None)
        if state is not None:
            self.ring.free(state.slot)
            req.dflash2_state = None

    # ------------------------------------------------------------------ the round

    def step(self, reqs: List[Req]) -> Tuple[List[DFlash2StepResult], torch.cuda.Event]:
        """Run one DFlash2 round over a batch of decode requests (on the engine stream).

        Returns ``(results, copy_done)`` where ``results`` is aligned to ``reqs``. The driver mutates
        each ``req.dflash2_state`` (frontier, anchor, pending features) and the request's committed
        stream (``append_host`` + ``device_len`` + ``cached_len``) as part of the commit. ``copy_done``
        is recorded on the engine stream after the GPU work so the scheduler's ``_process_last_data``
        can ``synchronize()`` before reading ``results[*].published``.
        """
        device = self.device
        B = len(reqs)
        if B == 0:
            raise ValueError("DFlash2 step called with an empty request list")

        # ---- steps 1-5: per-request draft (the draft context is the per-request ring; the
        # drafter's dynamic grouped conv breaks across requests, so the draft forward is a loop). ----
        # TODO(ON-BOX): the per-request draft loop (B small draft forwards per round); the verify
        # below is the single batched target forward. Batching the draft across requests would need
        # a varlen drafter and is out of scope for the reference milestone.
        verify_input_ids: List[torch.Tensor] = []
        verify_positions: List[torch.Tensor] = []
        drafts_all: List[torch.Tensor] = []
        proposal_q_all: List[torch.Tensor] = []
        candidates_all: List[torch.Tensor] = []
        for req in reqs:
            st = req.dflash2_state
            assert st is not None, "DFlash2 step before admit"
            F = st.frontier

            # step 1: materialize the committed prefix into the ring so the context reaches F.
            if st.initialized and st.pending_features is not None:
                # incremental: the previous round's committed columns [F_prev, F) are the pending.
                # materialize_context writes them into the ring and returns the live window.
                context_k, context_v, context_positions = self.drafter.materialize_context(
                    self.ring, st.slot, st.pending_features, st.pending_positions
                )
            else:
                # first round: cold-start re-prefill over [0, F) fills the ring (ON-BOX VALIDATION).
                self._cold_start(req, st)
                context_k, context_v, context_positions = self.ring.read_live_window(st.slot)

            # step 3: query block x0 = embed([a, MASK, ...]) at positions [F, ..., F+K].
            token_ids = torch.full((self.W,), self.mask_token_id, dtype=torch.int32, device=device)
            token_ids[0] = st.anchor
            x0 = self.engine.dflash2_embed(token_ids)  # [W, H] bf16
            positions = torch.arange(F, F + self.W, dtype=torch.int32, device=device)

            # step 4: draft forward -> [W, H].
            h = self.drafter.forward(x0, positions, context_k, context_v, context_positions)

            # step 5: select (greedy / temp 0 by default; the ops support temp>0 but it is OFF).
            drafts, proposal_q, candidates = self.drafter.select_candidates(
                h[1:],
                self.engine.dflash2_proposal_head,
                anchor=st.anchor,
                base_position=F,
                temperature=0.0,  # GREEDY default (deterministic). temp>0 is off by default.
                seed=st.seed,
            )
            drafts_all.append(drafts)  # [K]
            proposal_q_all.append(proposal_q)  # [top_k, K]
            candidates_all.append(candidates)  # [top_k, K]
            # the verify block is [a, d_1..d_P] at [F, ..., F+P] (P == K for the reference milestone)
            verify_input_ids.append(
                torch.cat([token_ids[:1], drafts[: self.P]])  # [W] (anchor + K drafts)
            )
            verify_positions.append(positions)  # [W]

        # ---- step 6: target verify (ONE batched varlen forward over all reqs, W cols each). ----
        s_t, final_hidden, spec_buf, gdn_slots = self._run_verify(
            reqs, verify_input_ids, verify_positions
        )

        # ---- steps 7-9: per-request accept, choose N, commit. ----
        results: List[DFlash2StepResult] = []
        for i, req in enumerate(reqs):
            st = req.dflash2_state
            F = st.frontier
            target_dists = self._per_column_target_dists(final_hidden, i)  # [W, vocab] fp32
            target_argmax = target_dists.argmax(dim=-1)  # [W] lower-token-id argmax

            # step 7: greedy acceptance (the target remains the authority).
            A, committed, _cont_col = accept_sparse_drafts(
                target_dists,
                drafts_all[i][: self.P],
                candidates_all[i][:, : self.P],
                proposal_q_all[i][:, : self.P],
                greedy=True,
                seed=st.seed,
                base_position=F,
                target_argmax=target_argmax,
            )

            # step 8: choose the licensed prefix length N (budget + first-EOS trim).
            N = self._choose_n(req, st, committed)

            # step 9: commit N (advances the req + folds the GDN state + retains the pending).
            published = committed[:N]  # [N] int (GPU)
            self._commit(req, st, i, F, N, published, s_t, spec_buf, gdn_slots)
            results.append(
                DFlash2StepResult(
                    published=published.to("cpu", dtype=torch.int32).reshape(-1),
                    n_published=N,
                    n_accepted=A,
                )
            )

        copy_done = torch.cuda.Event()
        copy_done.record(self.engine.stream)
        return results, copy_done

    # ------------------------------------------------------------------ helpers

    def _per_column_target_dists(self, final_hidden: torch.Tensor, i: int) -> torch.Tensor:
        """The target's per-column verify distributions ``p_0..p_{W-1}`` (softmax of the raw
        output-head logits), computed from ``final_hidden`` DIRECTLY -- NOT from the
        ``capture_dflash2_features`` logits, which ``lm_head.forward`` collapses to the last column
        for a prefill batch. Greedy: NO temperature/penalty -- the raw argmax is the target
        authority (temp>0 penalties/temperature would be applied here; plan R1)."""
        W = self.W
        final_hidden_i = final_hidden[i * W : (i + 1) * W]  # [W, H] bf16 (all columns)
        logits = F.linear(final_hidden_i, self._head_weight).float()  # [W, vocab] raw
        return F.softmax(logits, dim=-1)  # [W, vocab] fp32

    def _choose_n(self, req: Req, st: DFlash2ReqState, committed: torch.Tensor) -> int:
        """Licensed prefix length N in [0, len(committed)] = [0, A+1], trimmed by the output
        budget and (for greedy) the first EOS in the licensed prefix.

        TODO(ON-BOX): the full licensed N = A+1 is committed to the target KV (the target is the
        authority); the frontend may publish a shorter prefix on EOS/stop/budget. The stop STRING
        check lives in the scheduler's ``_process_last_data`` (which publishes the N tokens and
        detects a stop substring); here we trim by budget and first-EOS so the commit never exceeds
        the request's output budget.
        """
        n_lic = int(committed.numel())  # A + 1
        n = min(n_lic, max(0, req.remain_len))  # output budget
        if n == 0:
            return 0
        if not req.sampling_params.ignore_eos and self.eos_token_ids:
            for j in range(n):
                if int(committed[j].item()) in self.eos_token_ids:
                    n = j + 1  # stop at (and including) the first EOS
                    break
        return n

    def _commit(
        self,
        req: Req,
        st: DFlash2ReqState,
        i: int,
        F: int,
        N: int,
        published: torch.Tensor,
        s_t: torch.Tensor,
        spec_buf: torch.Tensor | None,
        gdn_slots: torch.Tensor,
    ) -> None:
        """Commit the N licensed tokens: advance the committed length, publish, fold the GDN state
        to column (N-1), and retain the pending features for the next round's materialization.

        KV-scratch invariant (TODO(ON-BOX)): the verify wrote W KV columns at ``[F, F+W)`` but only
        ``[F, F+N)`` is committed. ``alloc_frontier`` (the allocated page-table extent) is set to
        ``F+W`` so the NEXT round reuses the scratch ``[F+N, F+W)`` and allocates only the fresh
        committed growth -- the scratch is simply overwritten by the next verify, and the COMMITTED
        frontier (``req.cached_len``) advances only to ``F+N``. The page-table row for
        ``[F, F+W)`` stays allocated (not freed) so the scratch is reusable; on request finish the
        cache manager frees the tail past the committed prefix (``cache_req``).
        """
        device = self.device
        pub = published.to(torch.int32)  # [N] int (GPU)
        pub_cpu = pub.to("cpu", non_blocking=True).reshape(-1)  # [N] int32 CPU
        # TODO(ON-BOX): the per-request sync D2H (B small copies per round) could be batched into
        # one gather+copy for large B; small N (<= K+1) keeps it cheap for the reference milestone.

        if N <= 0:
            # Nothing committed this round (budget exhausted / no licensed token). The frontier does
            # not advance; the request is at its output budget and will finish by length.
            st.alloc_frontier = max(st.alloc_frontier, F + self.W)  # keep the scratch allocated
            return

        # (a) GDN rewind: fold each layer's recurrent state to column (N-1).
        self._gdn_rewind(i, N, st, spec_buf, gdn_slots)

        # (b) pending features for the NEXT round: s_t[:, :N] at positions [F, F+N).
        st.pending_features = s_t[i * self.W : i * self.W + N].detach().clone()  # [N, num_taps*H]
        st.pending_positions = torch.arange(F, F + N, dtype=torch.int32, device=device)

        # (c) advance the request's committed stream by N. ``append_host`` rebinds input_ids to
        # [x_0..x_{F+N}] (length F+1+N) but does NOT touch device_len, so set it explicitly to keep
        # the invariant input_ids.numel() == device_len, cached_len == device_len - 1.
        req.append_host(pub_cpu)
        req.device_len = F + 1 + N
        req.cached_len = F + N  # positions [0, F+N) committed; unprocessed = x_{F+N} (next anchor)
        # keep the GPU token_pool in sync (the DFlash2 path builds verify input_ids directly, but
        # the prefix cache / abort bookkeeping may read the request stream).
        self.token_pool[req.table_idx, F + 1 : F + 1 + N] = pub

        # (d) advance the round state.
        st.frontier = F + N
        st.anchor = int(pub_cpu[-1].item())  # last published = x_{F+N} = next round's anchor
        st.alloc_frontier = F + self.W  # KV scratch carried to the next round

    def _gdn_rewind(
        self,
        i: int,
        N: int,
        st: DFlash2ReqState,
        spec_buf: torch.Tensor | None,
        gdn_slots: torch.Tensor,
    ) -> None:
        """Fold each GDN layer's recurrent state to the accepted column (N-1) (the ``num_accepted``
        / GDN-rewind seam, plan §4.4).

        The verify's GDN kernel cached every column's recurrent state into
        ``spec_intermediate_buffer[li, req, col]`` (layout ``[cache_entry, step, i_hv, o_k, o_v]``)
        and skipped the live-slot write-back. Because the committed prefix is a PREFIX of the verify
        block, the state after the committed N columns is exactly the cached column-(N-1) state --
        copy it back to the request's live GDN slot.
        """
        pool = self.engine.linear_state_pool
        if pool is None or self._num_gdn_layers == 0 or spec_buf is None:
            return  # no GDN layers to rewind (pure full-attention model)
        slot = gdn_slots[i : i + 1]  # [1] int32
        for li in range(self._num_gdn_layers):
            rec = pool.recurrent_states[li]  # [num_slots, HV, K, V] fp32
            # spec_buf[li, i, N-1] is [HV, K, V] (cache_entry=i, step=N-1).
            rec.index_copy_(0, slot, spec_buf[li, i, N - 1].unsqueeze(0).to(rec.dtype))

    def _alloc_verify_kv(self, req: Req, first: int, last: int) -> None:
        """Allocate target-KV pages for ``[first, last)`` (the not-yet-allocated verify columns)
        and write them into the page-table row. Reuses the scratch ``[first, ...)`` already
        allocated by a prior round, so a round allocates only the fresh committed growth
        (the KV-scratch invariant). TODO(ON-BOX): page-table / committed-frontier accounting under
        the scratch (the row for [F, F+W) stays allocated; only [F, F+N) is committed)."""
        if last <= first:
            return
        cm = self.cache_manager
        page_size = self.engine.config.page_size
        first_page = (first + page_size - 1) // page_size
        last_page = (last + page_size - 1) // page_size
        if last_page <= first_page:
            return
        needed = last_page - first_page
        allocated = cm._page_to_token(cm._allocate(needed))
        from freetoken.scheduler.cache import _write_page_table

        _write_page_table(
            self.engine.page_table, allocated, [(req.table_idx, first_page, last_page)], page_size
        )

    # ------------------------------------------------------------------ target verify

    def _run_verify(
        self,
        reqs: List[Req],
        verify_input_ids: List[torch.Tensor],
        verify_positions: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Arrange + run the target verify forward (a varlen "prefill" of W columns/req) via
        ``model.capture_dflash2_features()``, wrapped in ``ctx.forward_batch`` + prepared attention
        metadata -- exactly like the prefill path (plan T15).

        Returns ``(s_t, final_hidden, spec_buf, gdn_slots)``. TODO(ON-BOX): the varlen verify-batch
        metadata (cu_seqlens / cache_indices / has_initial_state / out_loc / positions) and the GDN
        spec-verify metadata must be checked first on the box.
        """
        device = self.device
        B = len(reqs)
        W = self.W

        # KV allocation for the W verify columns [F, F+W) per req (reusing the scratch).
        for req in reqs:
            st = req.dflash2_state
            F = st.frontier
            last = F + W
            if last > st.alloc_frontier:
                self._alloc_verify_kv(req, st.alloc_frontier, last)
                st.alloc_frontier = last

        # flat verify tensors
        input_ids = torch.cat(verify_input_ids).to(device, dtype=torch.int32)  # [B*W]
        positions = torch.cat(verify_positions).to(device, dtype=torch.int32)  # [B*W]
        out_loc = torch.cat(
            [
                self.engine.page_table[
                    r.table_idx, r.dflash2_state.frontier : r.dflash2_state.frontier + W
                ]
                for r in reqs
            ]
        )  # [B*W]

        # GDN metadata: varlen cu_seqlens [0, W, 2W, ...] (int32), per-req live-slot cache_indices,
        # and (for GDN models) the spec-verify intermediate-state buffer [num_gdn_layers, B, W, HV, K, V].
        cu_seqlens = torch.arange(0, (B + 1) * W, W, dtype=torch.int32, device=device)  # [B+1]
        gdn_slots = torch.tensor(
            [self._gdn_slot(req) for req in reqs], dtype=torch.int32, device=device
        )  # [B]
        has_initial = torch.ones(B, dtype=torch.bool, device=device)
        spec_buf = None
        spec_idx = None
        if self._num_gdn_layers > 0 and self._gdn_hv_k_v is not None:
            HV, Kd, Vd = self._gdn_hv_k_v
            # TODO(ON-BOX, memory): this transient fp32 scratch is num_gdn_layers x B x W x HV x K x V
            # (~0.75 GiB for 48 GDN layers / B=1 / W=6); a bf16 buffer halves it if it OOMs on the box.
            # It is allocated per verify round and released once the rewind has folded it back.
            spec_buf = torch.empty(
                (self._num_gdn_layers, B, W, HV, Kd, Vd), dtype=torch.float32, device=device
            )
            spec_idx = torch.arange(B, dtype=torch.int32, device=device)

        # The attention backend derives per-request seqlens from Req.device_len / cached_len and
        # reads page_table[table_idx, :device_len]; for the verify we want seqlens_k = F+W and
        # seqlens_q = W, so temporarily lift device_len to F+W (restored before returning). The
        # attention then uses the varlen extend path (max_q_len = W > 1, prefix_lens = F).
        saved = [(r, r.device_len) for r in reqs]
        for r in reqs:
            r.device_len = r.cached_len + W  # F + W
        batch = Batch(reqs=list(reqs), phase="prefill")
        batch.padded_reqs = list(reqs)
        batch.input_ids = input_ids
        batch.positions = positions
        batch.out_loc = out_loc
        if self._num_gdn_layers > 0:
            from freetoken.attention.linear import FLAMetadata

            batch.fla_metadata = FLAMetadata(
                cu_seqlens=cu_seqlens,
                cache_indices=gdn_slots,
                has_initial_state=has_initial,
                spec_verify=True,
                spec_intermediate_buffer=spec_buf,
                spec_intermediate_indices=spec_idx,
                spec_num_draft=W,
            )
        try:
            self.engine.attn_backend.prepare_metadata(batch)
            with self.engine.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, False):
                s_t, final_hidden, _logits = self.model.capture_dflash2_features()
        finally:
            for r, old in saved:
                r.device_len = old
        return s_t, final_hidden, spec_buf, gdn_slots

    def _cold_start(self, req: Req, st: DFlash2ReqState) -> None:
        """First round: fill the ring with the committed-prompt context by a re-prefill over
        ``[0, F)`` (the committed prompt; the ring clips it to the live window ``[F-S, F)``).

        The committed prompt ``[0, F)`` is already resident in the target KV, so this is a
        re-read that (a) captures ``s_t`` for ``[0, F)`` (materialized into the ring) and (b)
        re-establishes the GDN state ``S_F`` (redundant with the prefill, but correct). No new KV
        pages are allocated (the prompt is committed); the out_loc is the existing page-table row.

        TODO(ON-BOX, cost): this is a one-time per-request re-prefill of the whole committed prompt
        (bounded in practice by the prompt length; the ring keeps only the last S). A cheaper
        bounded variant would re-prefill only ``[max(0, F-S), F)`` and seek the ring frontier to
        ``max(0, F-S)`` -- that needs a ring-frontier seam and is deferred (the reference milestone
        ships the simple full-prompt re-prefill, which is correct by construction).
        """
        device = self.device
        F = st.frontier
        if F <= 0:
            st.initialized = True
            return

        # varlen re-prefill over [0, F): a FRESH prefill (cached_len = 0, seqlens_q = F, seqlens_k = F)
        # so the whole prompt is the query and the prompt's s_t is captured.
        saved_device_len, saved_cached_len = req.device_len, req.cached_len
        req.device_len = F
        req.cached_len = 0
        try:
            input_ids = req.input_ids[0:F].to(device, dtype=torch.int32).contiguous()  # [F]
            positions = torch.arange(0, F, dtype=torch.int32, device=device)
            out_loc = self.engine.page_table[req.table_idx, 0:F]
            batch = Batch(reqs=[req], phase="prefill")
            batch.padded_reqs = [req]
            batch.input_ids = input_ids
            batch.positions = positions
            batch.out_loc = out_loc
            from freetoken.attention.linear import build_fla_metadata

            batch.fla_metadata = build_fla_metadata(batch, device)  # fresh prefill GDN metadata
            self.engine.attn_backend.prepare_metadata(batch)
            with self.engine.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, False):
                s_t, _final_hidden, _logits = self.model.capture_dflash2_features()
        finally:
            req.device_len, req.cached_len = saved_device_len, saved_cached_len

        # materialize [0, F) into the ring (the ring.write clips to [F-S, F)); ring frontier -> F.
        self.drafter.materialize_context(self.ring, st.slot, s_t, positions)
        st.initialized = True
        # no pending yet: the next round's draft reads the live window [max(0, F-S), F).
        st.pending_features = None
        st.pending_positions = None

    @staticmethod
    def _gdn_slot(req: Req) -> int:
        """The request's live GDN-state slot (mirrors ``build_fla_metadata``'s gdn_slot)."""
        return req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx


__all__ = ["DFlash2RoundDriver", "DFlash2ReqState", "DFlash2StepResult"]
