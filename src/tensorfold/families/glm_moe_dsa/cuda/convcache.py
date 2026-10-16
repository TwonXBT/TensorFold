"""Kept conversations beside the live one (TF_GLM_CONV_CACHE, single-stream serving with TF_GLM_KEEP_SLOTS).

The single-stream engine keeps one prompt warm (``Engine.kept``): a prompt that extends it resumes after the shared
prefix. An agent that switches between conversations (a sub-agent beside its parent, a title or summary call, two
sessions) lost the other conversation's rows on every switch and re-filled it from scratch: 150K tokens are ~4 minutes
of prefill on TP4. This module parks a conversation's prompt rows in the unused top of the cache planes the window
already holds (the live conversation grows from row 0; parked ones sit below ``capacity`` in the rows it has not
reached) and copies them back to row 0 when a later prompt resumes further into a parked conversation than into the
live one. No memory is added for the target's rows: a parked conversation gives way (is dropped, never copied) as soon
as the live one's prompt or reply needs its rows (``State.ceiling`` / ``State.spill``, checked by
``forward.check_room`` before every forward).

Bits: the parked rows are the prompt rows' own bits (a copy, both ways), and the resume after a restore is the same
``resume_cut`` as for the live prompt, so a restored conversation's reply is its reply after a fresh prefill
(TF_GLM_PROMPT_INVARIANT) or after the same chunk-aligned resume. A DSpark drafter's context ring is copied with it
(``Drafter.kc`` / ``vc`` and its position, a few hundred MB at most), so its drafts continue as if never switched; a
failed ring copy (out of memory on any rank) parks nothing (the ranks agree first).

TF_GLM_CONV_CACHE: the most parked conversations (default 4; 0: off). TF_GLM_CONV_CACHE_MIN: the fewest prompt rows a
conversation must hold to be parked (default 2,048). Every decision is a function of the prompts, the positions and
the sizes, the same on every rank.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch

MAX = int(os.environ.get("TF_GLM_CONV_CACHE") or 4)
MIN_ROWS = int(os.environ.get("TF_GLM_CONV_CACHE_MIN") or 2048)
MARGIN = 256          # rows kept free between the live conversation's next write and the lowest parked conversation
SWITCH = 256          # a parked conversation must resume this many rows further than the live one to be restored


def planes(st) -> list[torch.Tensor]:
    """Every row-indexed tensor of a State's cache (row p: position p): latent codes / scales (or bf16 latents),
    rope keys, indexer keys and the MTP head's planes."""

    out: list[torch.Tensor] = []

    def add(p) -> None:
        if p is None:
            return
        if hasattr(p, "codes"):
            out.append(p.codes)
            out.append(p.scales)
        else:
            out.append(p)

    for p in st.kc:
        add(p)
    for p in st.pc:
        add(p)
    for p in st.vc:
        add(p)
    for p in st.index or []:
        add(p)
    if hasattr(st, "mtp_kc"):
        add(st.mtp_kc)
        add(st.mtp_pc)
        add(getattr(st, "mtp_vc", None))
    return out


def plane_row_bytes(st) -> int:
    return sum(t[0].numel() * t.element_size() for t in planes(st))


@dataclass
class Parked:
    tokens: list[int]
    start: int                       # first cache row of its copy
    used: int                        # LRU stamp
    ring: list[torch.Tensor] | None = None     # the drafter's K/V ring (kc + vc), when one drafts
    ring_end: int = 0
    hits: int = field(default=0)

    @property
    def rows(self) -> int:
        return len(self.tokens)

    @property
    def end(self) -> int:
        return self.start + self.rows


class ConvCache:
    """Parked conversations in the top rows of one State's planes (see the module docstring)."""

    def __init__(self, st, comm=None, world: int = 1, rank: int = 0, most: int | None = None,
                 min_rows: int | None = None) -> None:
        self.st = st
        self.comm, self.world, self.rank = comm, world, rank
        self.most = int(MAX if most is None else most)
        self.min_rows = int(MIN_ROWS if min_rows is None else min_rows)
        self.items: list[Parked] = []
        self.stamp = 0
        self.capacity = st.capacity
        self.planes = planes(st)
        self.row_bytes = sum(t[0].numel() * t.element_size() for t in self.planes)
        st.ceiling = self.capacity
        st.spill = self.spill
        self.note = ""

    # -- room ------------------------------------------------------------------------------------------------------
    def _ceiling(self) -> None:
        self.st.ceiling = min([p.start for p in self.items], default=self.capacity)

    def spill(self, need: int) -> None:
        """The live conversation will write rows below ``need``: drop every parked one that starts below it (plus the
        margin). Called by forward.check_room (the same positions on every rank)."""

        keep = [p for p in self.items if p.start >= need + MARGIN]
        if len(keep) != len(self.items):
            self.items = keep
            self._ceiling()

    def _place(self, n: int, floor: int, pinned: Parked | None) -> int | None:
        """The highest start for n rows at or above ``floor`` clear of every parked conversation, dropping the least
        recently used ones (never ``pinned``) until it fits; None when n rows cannot fit above floor."""

        if floor + n > self.capacity:
            return None
        while True:
            taken = sorted((p.start, p.end) for p in self.items)
            hi = self.capacity
            for lo_, hi_ in reversed(taken):
                if hi - n >= max(hi_, floor):
                    return hi - n
                hi = min(hi, lo_)
            if hi - n >= floor:
                return hi - n
            victims = [p for p in self.items if p is not pinned]
            if not victims:
                return None
            self.items.remove(min(victims, key=lambda p: p.used))

    def _agree(self, ok: bool) -> bool:
        """Every rank's ``ok`` (a ring copy that may fail with out of memory on one rank only)."""

        if self.comm is None or self.world <= 1:
            return ok
        dev = self.planes[0].device
        send = torch.tensor([1 if ok else 0], dtype=torch.int32, device=dev)
        recv = torch.zeros((self.world,), dtype=torch.int32, device=dev)
        self.comm.all_gather(send, recv)
        return bool(int(recv.min()) == 1)

    # -- moves -----------------------------------------------------------------------------------------------------
    def _copy(self, src: int, dst: int, n: int) -> None:
        for t in self.planes:
            t[dst:dst + n].copy_(t[src:src + n])

    def _park(self, tokens: list[int], floor: int, drafter, pinned: Parked | None) -> Parked | None:
        n = len(tokens)
        # an older state of the same conversation (a prefix of it, give or take a reply) is worth nothing beside it
        from .multi import common_prefix

        self.items = [p for p in self.items if common_prefix(p.tokens, tokens) < p.rows - MIN_ROWS or p is pinned]
        start = self._place(n, max(floor, n), pinned)
        if start is None:
            return None
        ring = None
        if drafter is not None:
            ok = True
            try:
                ring = [t.clone() for t in list(drafter.kc) + list(drafter.vc)]
            except torch.cuda.OutOfMemoryError:
                ring, ok = None, False
            if not self._agree(ok):
                return None
        self._copy(0, start, n)
        self.stamp += 1
        p = Parked(list(tokens), start, self.stamp, ring, drafter.context_end if drafter is not None else 0)
        self.items.append(p)
        if len(self.items) > self.most:
            self.items.remove(min((q for q in self.items if q is not p), key=lambda q: q.used))
        self._ceiling()
        return p

    def _restore(self, p: Parked, drafter) -> None:
        self._copy(p.start, 0, p.rows)
        if drafter is not None:
            if p.ring is not None:
                for dst, src in zip(list(drafter.kc) + list(drafter.vc), p.ring):
                    dst.copy_(src)
                drafter.context_end = p.ring_end
                drafter.pos_dev.fill_(p.ring_end)
            else:
                drafter.reset()
        self.items.remove(p)
        self._ceiling()

    # -- the prefill's hook ----------------------------------------------------------------------------------------
    def arrange(self, e, prompt, drafter, step: int, short: int) -> None:
        """Before a prompt fills: restore the parked conversation it resumes furthest into (when that beats the live
        one by SWITCH rows), parking the live one first when it holds MIN_ROWS rows the prompt would overwrite; then
        clear the rows the prompt fills. Leaves ``e.kept`` as the conversation now at row 0."""

        from .decode import resume_cut

        t0 = time.perf_counter()
        live = getattr(e, "kept", None)
        cut_live = resume_cut(live, prompt, step, short)
        best, cut_best = None, cut_live + SWITCH
        for p in self.items:
            k = resume_cut(p.tokens, prompt, step, short)
            if k > cut_best:
                best, cut_best = p, k
        floor = len(prompt) + MARGIN
        # parked conversations under the prompt's rows give way (the one being restored is copied out first)
        self.items = [p for p in self.items if p.start >= floor or p is best]
        notes = []
        lost = len(live) - (0 if best is not None else cut_live) if live else 0
        if live and len(live) >= self.min_rows and lost >= self.min_rows:
            # the restore writes rows [0, best.rows): the live copy must sit above them too
            low = max(floor, best.rows if best is not None else 0)
            if self._park(live, low, drafter, best) is not None:
                notes.append(f"parked {len(live)}")
            elif self.items and best is None:
                notes.append(f"no room to park {len(live)}")
        if best is not None:
            self._restore(best, drafter)
            e.kept = best.tokens
            notes.append(f"restored {best.rows} (resume {cut_best} vs {cut_live})")
        self.items = [p for p in self.items if p.start >= floor]
        self._ceiling()
        if notes:
            torch.cuda.synchronize()
            self.note = (f"conversation cache: {', '.join(notes)} in {1e3 * (time.perf_counter() - t0):.0f} ms; "
                         f"{len(self.items)} parked ({sum(p.rows for p in self.items)} rows)")
        else:
            self.note = ""
