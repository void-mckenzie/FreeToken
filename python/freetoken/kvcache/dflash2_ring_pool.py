"""Per-request cyclic context K/V ring for the DFlash2 speculative-draft drafter (plan T4).

The five *local* (sliding) draft layers store their context K/V in **cyclic rings** of capacity
``S = DFlash2DraftConfig.sliding_window`` (2048 for the 27B instance). There is **no full-attention
pool** in 27B DFlash2 (all five draft layers are local). This is the drafter's analogue of
FreeToken's ``LinearStatePool``: one fixed slot per running request, indexed the same way the page
table / GDN state index (``Req.table_idx``), so the scheduler's existing admit/free of ``table_idx``
covers the ring's lifetime (plan §4.6).

Ring geometry (ninfer ``cyclic_kv_cache.h`` / ``dflash.md`` "Backend storage and lifecycle"):

* absolute position ``p`` maps to physical slot ``p % capacity``;
* for context frontier ``F_ctx`` the live interval is ``[max(0, F_ctx - S), F_ctx)``; bytes outside
  the live interval are unreachable (a new request ``reset``-s its slot to zero + frontier 0).

Storage dtype (the parity-critical part, plan R1):

* **K is stored as BF16** -- ``[num_layers, num_slots, S, n_kv, D]``;
* **V is stored as FP16** -- the materializer writes ``FP16_RNE(BF16(v_raw))`` (the double-round:
  bf16 then fp16, NOT a single fp16 round and NOT bf16). The draft attention reads the FP16 V
  straight out of the ring.

Per state image (27B: 5 local layers, ``n_kv=8``, ``D=128``, ``S=2048``)::

    5 layers * (2-byte K + 2-byte V) * 8 heads * 128 dim * 2048 slots = 40 MiB

Reference-PyTorch only: the ring is a plain preallocated tensor with ``p % S`` scatter/gather. No
CUDA/Triton kernel. The write/read below are the *only* device ops; everything else is metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Union

import torch

if TYPE_CHECKING:
    from freetoken.models.config import DFlash2DraftConfig


def dflash2_ring_bytes_per_req(cfg: "DFlash2DraftConfig") -> int:
    """DFlash2 context ring bytes for ONE request (all local layers). Mirrors the plan's
    ``5 * (2 + 2) * n_kv * D * S`` image; TP=1 only (the drafter never shards, plan R3)."""
    return cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * cfg.sliding_window * (2 + 2)


class DFlash2RingPool:
    """Per-request cyclic K/V storage for the DFlash2 local draft layers.

    One slot per running request (slot id = the request's ``table_idx``). Per slot, per layer, K is
    BF16 ``[S, n_kv, D]`` and V is FP16 ``[S, n_kv, D]``. The pool tracks the per-slot **context
    frontier** ``F_ctx`` (the count of context positions materialized so far = the next absolute
    position to write, since context always starts at 0) so ``read_live_window`` can reconstruct the
    live interval. Slot 0 is a reserved padding sink (FreeToken ``MambaPool`` convention); live slots
    are drawn from ``alloc``.

    Invariant (the caller / round driver owns it): ``write`` appends a **contiguous** block of
    absolute positions ``[frontier(slot), frontier(slot) + n)`` in increasing row order. Both the
    incremental decode path (newly-committed positions each round) and the full-context prefill path
    (``[0, L)`` in one call) satisfy this.
    """

    def __init__(
        self,
        cfg: "DFlash2DraftConfig",
        num_slots: int,
        device: Union[torch.device, str],
    ) -> None:
        self._cfg = cfg
        self.capacity = cfg.sliding_window
        self.num_layers = cfg.num_layers
        self.n_kv = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.num_slots = num_slots
        self.device = torch.device(device)
        # K BF16 / V FP16, one plane per (layer, slot); zero-filled (dead rows are never read).
        self.k = torch.zeros(
            (self.num_layers, num_slots, self.capacity, self.n_kv, self.head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.v = torch.zeros(
            (self.num_layers, num_slots, self.capacity, self.n_kv, self.head_dim),
            dtype=torch.float16,
            device=self.device,
        )
        # per-slot context frontier (host int): count of context positions materialized so far.
        self._frontier: List[int] = [0] * num_slots
        # Free-list allocator over slots 1..num_slots-1 (slot 0 = reserved padding sink).
        self.padding_slot = 0
        self._free_slots: List[int] = list(range(1, num_slots))

    # ---------------------------------------------------------------- slot lifecycle

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def alloc(self, n: int = 1) -> List[int]:
        """Pop ``n`` free slot ids (LIFO). Raises if the pool is exhausted."""
        if n > len(self._free_slots):
            raise RuntimeError(
                f"DFlash2RingPool exhausted: need {n}, have {len(self._free_slots)}"
            )
        return [self._free_slots.pop() for _ in range(n)]

    def free(self, slots: Union[int, Sequence[int], torch.Tensor]) -> None:
        """Return slot ids to the free-list. Accepts an int, list, or 1-D tensor."""
        if isinstance(slots, torch.Tensor):
            slots = slots.flatten().tolist()
        elif isinstance(slots, int):
            slots = [slots]
        self._free_slots.extend(int(s) for s in slots)

    def reset(self, slot: int) -> None:
        """Zero a slot's K/V and rewind its frontier (fresh sequence / request admit)."""
        self.k[:, slot].zero_()
        self.v[:, slot].zero_()
        self._frontier[slot] = 0

    def frontier(self, slot: int) -> int:
        """The slot's context frontier ``F_ctx`` (the next absolute position to write)."""
        return self._frontier[slot]

    # ---------------------------------------------------------------- write / read

    def write(
        self,
        slot: int,
        positions: torch.Tensor,
        k_per_layer: Sequence[torch.Tensor],
        v_per_layer: Sequence[torch.Tensor],
    ) -> None:
        """Scatter a contiguous block of positions' context K/V into the slot's ring.

        Args:
            slot: the request's ring slot.
            positions: ``[n]`` integer absolute positions, contiguous
                ``[frontier(slot), frontier(slot) + n)`` in increasing row order.
            k_per_layer: ``num_layers`` tensors, each ``[n, n_kv, D]`` **BF16** -- the materialized
                context K (already projected, head-normed and roped by the materializer).
            v_per_layer: ``num_layers`` tensors, each ``[n, n_kv, D]`` **FP16** -- the materialized
                context V (``FP16_RNE(BF16(v_raw))``).

        Only the rows still inside the new live window are scattered (rows that wrapped out are
        dead and left as-is). The slot's frontier advances by ``n``.
        """
        n = positions.numel()
        if n == 0:
            return
        old = self._frontier[slot]
        new = old + n
        pos = positions.to(torch.long).view(-1)  # [n]
        # Clip to the live window: skip rows that have wrapped out (only reachable when a single
        # write exceeds S, e.g. an oversized full-context prefill). For the standard geometry a
        # prefill is <= S so this is a no-op.
        live_start = max(0, new - self.capacity)
        skip = max(0, live_start - old)
        if skip:
            pos = pos[skip:]
            k_per_layer = [x[skip:] for x in k_per_layer]
            v_per_layer = [x[skip:] for x in v_per_layer]
        slots_idx = pos % self.capacity  # physical slot per absolute position
        for l in range(self.num_layers):
            self.k[l, slot].index_copy_(0, slots_idx, k_per_layer[l])
            self.v[l, slot].index_copy_(0, slots_idx, v_per_layer[l])
        self._frontier[slot] = new

    def read_live_window(
        self, slot: int
    ) -> tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Read the live context window back for a draft forward.

        Returns ``(context_k, context_v, context_positions)`` where:

        * ``context_k[i]`` is ``[S_live, n_kv, D]`` **BF16** (layer ``i``);
        * ``context_v[i]`` is ``[S_live, n_kv, D]`` **FP16** (layer ``i``);
        * ``context_positions`` is ``[S_live]`` **int32** -- the live rows' absolute positions in
          row order (``[max(0, F-S), F)``).

        ``S_live = min(F, S)``. These are exactly the shapes ``DFlash2Drafter.forward`` expects
        (``context_k``/``context_v`` sequences + a shared ``context_positions``).
        """
        f = self._frontier[slot]
        start = max(0, f - self.capacity)
        positions = torch.arange(start, f, dtype=torch.int32, device=self.device)  # [S_live]
        slots_idx = positions.to(torch.long) % self.capacity  # [S_live]
        context_k = [self.k[l, slot].index_select(0, slots_idx) for l in range(self.num_layers)]
        context_v = [self.v[l, slot].index_select(0, slots_idx) for l in range(self.num_layers)]
        return context_k, context_v, positions

    # ---------------------------------------------------------------- accounting

    @property
    def bytes_per_slot(self) -> int:
        """Total ring bytes for one request (all local layers)."""
        per_k = self.k[:, 0].numel() * self.k.element_size()
        per_v = self.v[:, 0].numel() * self.v.element_size()
        return int(per_k + per_v)

    def __len__(self) -> int:
        return self.num_slots


__all__ = ["DFlash2RingPool", "dflash2_ring_bytes_per_req"]
