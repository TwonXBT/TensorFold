# Origin: MiaAI-Lab GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold engine, patches 0007/0032/0140; copied verbatim.
# License: Apache-2.0.
"""Prompt-lookup ("copy") drafts: when the context's last tokens occurred before, propose what followed them.

A reply that quotes or edits its prompt (or repeats itself) is cheap to draft: its last ``match`` tokens (the
pending token included) are searched in the prompt and the reply so far, and the tokens after the latest earlier
occurrence become the drafts. They only propose; verification keeps the serial sample, so replies stay exact.

Three refinements, off by default (they only change what is proposed, never what is kept):
- ``reply_match`` (TF_GLM_COPY_REPLY_MATCH): an occurrence that starts inside the reply must match the context's last
  ``reply_match`` tokens, not ``match``. Code replies repeat short boilerplate (``    def __init__(self``, ``}); ``)
  whose continuations differ: over 39 live code replies (2026-09-30) reply-sourced copies kept 23% of their drafts,
  prompt-sourced ones 74%, and a missed copy round costs a wide window and the DFlash2 round it displaced.
- ``miss_most`` (TF_GLM_COPY_MISS_MAX): after a copied round whose drafts were not all kept, the next copied round
  proposes at most this many (a narrow window), until a copied round keeps all of its drafts again.
- ``pad`` (the engine's TF_GLM_WIDE_GRAPHS widths): a proposal whose window (pending token + drafts) falls between 8
  rows and a captured width is padded to that width by repeating its last draft, so the window replays a CUDA graph
  instead of running eagerly. Padded drafts are drafts like any other: kept only where they equal the sample.

Hybrid copies (``hybrid``, TF_GLM_COPY_HYBRID=1, with a DSpark drafter): a copy no longer replaces the drafter's
block. The block runs, and each copy continuation (from up to ``sources`` earlier occurrences, latest first) is walked
through DSpark's slots with the Markov bias conditioned on the copy's own previous token: the copy agrees at a slot
while its token is among DSpark's ``rank`` best picks there (the same score, keyed noise and target filter as
``dspark.Drafter.chain``). The best-agreeing continuation wins (ties: the latest occurrence); see ``hybrid`` for
how its agreed prefix, DSpark's own chain and the copy's tail past the block's slots make the proposal. A copy that
agreed at every slot and was kept whole may draft its next round without the block (``skip``): the drafter's taps
still join every round, so its context stays the committed reply's.

Both ranks hold the same prompt and sample the same tokens, so each computes the same proposals with no exchange.
Pure numpy (no torch), so it is testable anywhere."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

MATCH = 8        # the context's last this many tokens must have occurred before (TF_GLM_COPY_MATCH)
MOST = 5         # drafts a round (TF_GLM_COPY_MAX, up to 15): graphs cover windows of 8 rows, wider ones run eagerly


RANK = 1          # a copy token agrees with DSpark at a slot when it is among its best this many picks there
SOURCES = 4       # earlier occurrences whose continuations are walked (distinct ones, latest first)
SCAN = 64         # occurrences looked at for them, latest first
SHORT = 4        # the hybrid's index match (TF_GLM_COPY_SHORT)
TRUST_REPLY = 24  # a reply occurrence matching this many tokens is copied unvetted (TF_GLM_COPY_TRUST_REPLY)
GRAPHED = 8      # verify windows up to this many rows always replay CUDA graphs (engine.GRAPH_ROWS)


@dataclass(frozen=True)
class CopySettings:
    match: int = MATCH
    most: int = MOST
    reply_match: int = 0             # 0: occurrences in the reply need ``match`` tokens like the prompt's
    pad: tuple[int, ...] = ()        # window widths past GRAPHED rows that replay CUDA graphs (TF_GLM_WIDE_GRAPHS)
    miss_most: int = 0               # 0: off; else drafts a copied round after one that missed
    hybrid: HybridSettings | None = None     # TF_GLM_COPY_HYBRID: copies vetted by DSpark's block

    @classmethod
    def from_env(cls, max_drafts: int, env=None, pad: Sequence[int] = ()) -> CopySettings | None:
        """TF_GLM_COPY_DRAFTS=1 turns copy drafts on (off by default); TF_GLM_COPY_MATCH, TF_GLM_COPY_MAX and
        TF_GLM_COPY_REPLY_MATCH (0 or more than the match, up to 64) and TF_GLM_COPY_MISS_MAX (0: off, or 1 up to
        TF_GLM_COPY_MAX) tune; ``pad``: the engine's wide graph widths."""

        env = os.environ if env is None else env
        on = env.get("TF_GLM_COPY_DRAFTS", "0").strip()
        if on in ("", "0"):
            return None
        if on != "1":
            raise ValueError(f"TF_GLM_COPY_DRAFTS: 0 or 1, not {on!r}")

        def number(name: str, default: int, low: int, high: int) -> int:
            text = env.get(name, "").strip()
            if text == "":
                return default
            if not text.isdecimal() or not low <= int(text) <= high:
                raise ValueError(f"{name}: {low} to {high}, not {text!r}")
            return int(text)

        match = number("TF_GLM_COPY_MATCH", MATCH, 2, 64)
        reply = number("TF_GLM_COPY_REPLY_MATCH", 0, 0, 64)
        if reply and reply <= match:
            raise ValueError(f"TF_GLM_COPY_REPLY_MATCH: 0 (off) or more than TF_GLM_COPY_MATCH ({match}), not {reply}")
        widths = tuple(sorted(int(r) for r in pad if GRAPHED < int(r) <= max_drafts + 1))
        most = number("TF_GLM_COPY_MAX", min(MOST, max_drafts), 1, max_drafts)
        return cls(match, most, reply, widths, number("TF_GLM_COPY_MISS_MAX", 0, 0, most), HybridSettings.from_env(env))

    def code(self) -> list[int]:
        """The settings as ints both ranks compare at startup (zeros: off; the pad widths are the engine's
        TF_GLM_WIDE_GRAPHS, compared on their own)."""

        h = self.hybrid.code() if self.hybrid is not None else [0] * len(HybridSettings().code())
        return [self.match, self.most, self.reply_match, self.miss_most] + h

    def drafts(self, context: Sequence[int], hybrid: bool = True) -> CopyDrafts:
        """A request's index: ``context`` is its prompt and pending token (the reply's first token); ``hybrid``: the
        request drafts with DSpark (``dflash_decode``), so TF_GLM_COPY_HYBRID applies (its short match)."""

        h = self.hybrid if hybrid else None
        return CopyDrafts(context, h.match if h is not None else self.match, self.most,
                          prompt=max(len(context) - 1, 0), reply_match=self.reply_match, pad=self.pad,
                          miss_most=self.miss_most, hybrid=h)


class CopyDrafts:
    """One sequence's context (prompt, then committed reply tokens and the pending one) and its copy proposals."""

    def __init__(self, context: Sequence[int], match: int = MATCH, most: int = MOST, *, prompt: int = 0,
                 reply_match: int = 0, pad: Sequence[int] = (), miss_most: int = 0,
                 hybrid: HybridSettings | None = None) -> None:
        """``prompt``: how many leading context tokens are the prompt (occurrences starting at or past it are the
        reply's, which need ``reply_match`` tokens when that is more than ``match``); ``pad``: window widths past
        GRAPHED rows to pad a proposal's window up to (``propose``); ``miss_most``: drafts a copied round after one
        that missed (0: ``most`` always); ``hybrid``: the request's rounds take ``plan`` and DSpark's vetting
        (``hybrid``) instead of ``propose``."""

        if match < 1 or most < 1:
            raise ValueError("match and most must be positive")
        self.match, self.most = int(match), int(most)
        self.prompt = int(prompt)
        self.reply_match = int(reply_match) if int(reply_match) > self.match else 0
        self.pad = tuple(sorted(int(r) for r in pad if int(r) > GRAPHED))
        self.miss_most = int(miss_most)
        self.hybrid = hybrid
        self.proposed = 0              # drafts of the last proposal not yet settled by ``extend``
        self.missed = False            # the last settled copied round kept fewer than all of its drafts
        self.trail: tuple[int, int] | None = None   # a trusted hybrid proposal: (occurrence start, drafts)
        self.trusted = -1              # where that occurrence continues after it was kept whole (-1: none)
        n = len(context)
        self.buf = np.empty((max(1024, 2 * n),), dtype=np.int32)
        self.buf[:n] = np.asarray(context, dtype=np.int32) if n else 0
        self.length = n

    def __len__(self) -> int:
        return self.length

    def tokens(self) -> list[int]:
        return self.buf[:self.length].tolist()

    def extend(self, tokens: Sequence[int]) -> None:
        """Committed tokens (the last one is the next round's pending token) join the context; after a proposal,
        they settle whether its round kept all of its copied drafts (they then number at least those plus one)."""

        k = len(tokens)
        if not k:
            return
        if self.proposed:
            self.missed = k <= self.proposed
            self.proposed = 0
        self.trusted = -1
        if self.trail is not None:
            start, q = self.trail
            self.trail = None
            if k >= q + 1:             # every copied draft kept: the occurrence continues k tokens on
                self.trusted = start + k
        if self.length + k > self.buf.shape[0]:
            grown = np.empty((2 * (self.length + k),), dtype=np.int32)
            grown[:self.length] = self.buf[:self.length]
            self.buf = grown
        self.buf[self.length:self.length + k] = np.asarray(tokens, dtype=np.int32)
        self.length += k

    def starts(self) -> np.ndarray:
        """Ascending starts of the earlier occurrences of the context's last ``match`` tokens (not the suffix)."""

        L, n = self.length, self.match
        if L <= n:
            return np.empty((0,), dtype=np.int64)
        ctx = self.buf
        q = ctx[L - n:L]
        # starts 0 .. L - n - 1 (each leaves a token after its match): by the last token, then the others
        hits = np.flatnonzero(ctx[n - 1:L - 1] == q[n - 1])
        for k in range(n - 1):
            if not hits.size:
                break
            hits = hits[ctx[hits + k] == q[k]]
        extra = self.reply_match - n if self.reply_match else 0
        if extra and hits.size:
            # an occurrence starting inside the reply: the extra tokens before it must match too
            inside = hits >= self.prompt
            ok = ~inside | ((hits >= extra) & (L - n - extra >= 0))
            for k in range(1, extra + 1):
                sel = inside & ok
                if not sel.any():
                    break
                ok[sel] = ctx[hits[sel] - k] == ctx[L - n - k]
            hits = hits[ok]
        return hits

    def propose(self, room: int | None = None) -> list[int]:
        """Up to ``min(most, room)`` drafts: what followed the latest earlier occurrence that has that many tokens
        after it, else the most after any occurrence (the earliest); [] when the suffix never occurred before. With
        ``pad``, a window of more than GRAPHED rows short of a padded width is filled up to it (``room`` allowing)
        by repeating the last draft."""

        most = min(self.most, self.miss_most) if self.missed and self.miss_most else self.most
        k = most if room is None else min(most, int(room))
        self.proposed = 0
        if k < 1:
            return []
        hits = self.starts()
        if not hits.size:
            return []
        L, n = self.length, self.match
        full = hits[hits <= L - n - k]
        s = int(full[-1]) if full.size else int(hits[0])
        out = self.buf[s + n:min(s + n + k, L)].tolist()
        self.proposed = len(out)                  # the copied drafts (a pad's repeats may miss without counting)
        return self.padded(out, room)

    def continuations(self, room: int | None = None, sources: int = SOURCES) -> list[tuple[int, list[int]]]:
        """(start, up to ``min(most, room)`` tokens that followed it) of the latest ``sources`` earlier occurrences
        whose continuations differ (among the latest SCAN): latest first those with that many tokens after them, then
        the others, earliest (longest) first ([] when the suffix never occurred before)."""

        k = self.most if room is None else min(self.most, int(room))
        self.trail = None
        if k < 1:
            return []
        L, n = self.length, self.match
        hits = self.starts()[::-1][:SCAN]               # a common short match in a long prompt: the latest SCAN
        # those with k tokens after them first (a run's latest occurrences overlap the suffix: little follows them)
        hits = np.concatenate([hits[hits <= L - n - k], hits[hits > L - n - k][::-1]])
        out: list[tuple[int, list[int]]] = []
        seen: set[bytes] = set()
        for s in hits:
            c = self.buf[int(s) + n:min(int(s) + n + k, L)]
            key = c.tobytes()
            if c.size and key not in seen:
                seen.add(key)
                out.append((int(s), c.tolist()))
                if len(out) >= sources:
                    break
        return out

    def match_length(self, start: int, most: int = 64) -> int:
        """How many tokens before the occurrence's continuation equal the context's last ones (at least ``match``
        for an occurrence ``starts`` found; at most ``most``)."""

        L, n, end = self.length, self.match, int(start) + self.match
        k = 0
        while k < most and end - 1 - k >= 0 and self.buf[end - 1 - k] == self.buf[L - 1 - k]:
            k += 1
        return k

    def follow(self, start: int, drafts: int) -> None:
        """The round proposes ``drafts`` tokens copied from the occurrence at ``start`` that DSpark agreed with at
        every slot (or a trusted continuation of one): ``extend`` notes whether they were all kept."""

        self.trail = (int(start), int(drafts))

    def trusted_drafts(self, room: int | None = None, least: int = 1) -> tuple[int, list[int]]:
        """After a trusted copy was kept whole: (start, its continuation, up to ``min(most, room)`` tokens) when the
        occurrence still matches the context's suffix and at least ``least`` tokens follow it; else (-1, [])."""

        s = self.trusted
        self.trusted = -1
        k = self.most if room is None else min(self.most, int(room))
        if s < 0 or k < max(1, least):
            return -1, []
        L, n = self.length, self.match
        if s + n >= L or not np.array_equal(self.buf[s:s + n], self.buf[L - n:L]):
            return -1, []
        c = self.buf[s + n:min(s + n + k, L)].tolist()
        return (s, c) if len(c) >= least else (-1, [])

    def plan(self, h: HybridSettings, room: int | None = None, least: int = 1) -> Plan:
        """The copy side of a hybrid round: a trusted copy kept whole last round that still matches ("s": its
        continuation, at least ``least`` tokens: the block's slots); else among the latest ``h.sources`` distinct
        continuations, of those whose match reaches the trust of their source (prompt or reply), the one with the
        most tokens after it (then the longest match, then the latest): alone, it is copied ("c"); beside other
        continuations it is ``trusted`` while the block vets them all ("v"); else every continuation for the block
        to vet ("v"); else none ("")."""

        if h.skip:
            start, drafts = self.trusted_drafts(room, least)
            if drafts:
                return Plan("s", start, drafts, copied=len(drafts))
        self.trusted = -1
        conts = self.continuations(room, h.sources)
        if not conts:
            return Plan()
        best, top = -1, (-1, -1)
        for k, (start, cont) in enumerate(conts):
            n = self.match_length(start)
            trust = h.trust_prompt if start < self.prompt else h.trust_reply
            if trust and n >= trust and (len(cont), n) > top:
                best, top = k, (len(cont), n)
        if best >= 0 and len(conts) == 1:
            return Plan("c", conts[best][0], conts[best][1], copied=len(conts[best][1]))
        return Plan("v", conts=conts, trusted=best)

    def padded(self, drafts: list[int], room: int | None = None) -> list[int]:
        """``drafts`` filled up to the next ``pad`` width's window by repeating the last one, when their window is
        wider than GRAPHED rows, narrower than that width, and ``room`` (default ``most``) leaves space for it."""

        rows = 1 + len(drafts)
        if not self.pad or not drafts or rows <= GRAPHED:
            return drafts
        width = next((w for w in self.pad if w >= rows), 0)
        if width <= rows or width - 1 > (self.most if room is None else int(room)):
            return drafts
        return drafts + [drafts[-1]] * (width - rows)


# -- hybrid copies: a copy checked (and extended) by DSpark's block ------------------------------------------------
# slot(d, prev) -> (pick scores over slot d's candidates [K], -inf where the target's rule excludes one; slot d's
# confidence), given the previous draft is slot d - 1's candidate ``prev`` (-1: the anchor, the pending token). The
# engine's comes from ``dspark.Drafter.slot_scorer``, the offline replay's from a dump record (``copy_sim``): the same
# float64 terms, so both walk the same chains.
Slot = Callable[[int, int], tuple[np.ndarray, float]]



@dataclass(frozen=True)
class HybridSettings:
    match: int = SHORT            # the index's match (shorter than MATCH: DSpark vets what a short match proposes)
    rank: int = RANK
    sources: int = SOURCES
    skip: bool = True             # a whole-kept trusted copy drafts its next round without the block
    trust_prompt: int = MATCH     # a prompt occurrence matching at least this many tokens is copied unvetted
    trust_reply: int = TRUST_REPLY    # likewise one in the reply (0: never)

    @classmethod
    def from_env(cls, env=None) -> HybridSettings | None:
        """TF_GLM_COPY_HYBRID=1 (off by default; it needs DSpark): copies vetted by the drafter's block, with
        TF_GLM_COPY_SHORT (the index's match, 2 to 64), TF_GLM_COPY_RANK (1 to 8), TF_GLM_COPY_SOURCES (1 to 16),
        TF_GLM_COPY_SKIP (0 or 1), TF_GLM_COPY_TRUST_PROMPT and TF_GLM_COPY_TRUST_REPLY (0: never, else at least the
        short match, up to 64)."""

        env = os.environ if env is None else env
        on = env.get("TF_GLM_COPY_HYBRID", "0").strip()
        if on in ("", "0"):
            return None
        if on != "1":
            raise ValueError(f"TF_GLM_COPY_HYBRID: 0 or 1, not {on!r}")

        def number(name: str, default: int, low: int, high: int) -> int:
            text = env.get(name, "").strip()
            if text == "":
                return default
            if not text.isdecimal() or not low <= int(text) <= high:
                raise ValueError(f"{name}: {low} to {high}, not {text!r}")
            return int(text)

        short = number("TF_GLM_COPY_SHORT", SHORT, 2, 64)
        trust = [number(f"TF_GLM_COPY_TRUST_{k}", d, 0, 64) for k, d in (("PROMPT", MATCH), ("REPLY", TRUST_REPLY))]
        for k, t in zip(("PROMPT", "REPLY"), trust):
            if 0 < t < short:
                raise ValueError(f"TF_GLM_COPY_TRUST_{k}: 0 (never) or at least TF_GLM_COPY_SHORT ({short}), not {t}")
        return cls(short, number("TF_GLM_COPY_RANK", RANK, 1, 8), number("TF_GLM_COPY_SOURCES", SOURCES, 1, 16),
                   bool(number("TF_GLM_COPY_SKIP", 1, 0, 1)), *trust)

    def code(self) -> list[int]:
        return [1, self.match, self.rank, self.sources, int(self.skip), self.trust_prompt, self.trust_reply]


@dataclass
class Plan:
    """A hybrid round's copy side (``CopyDrafts.plan``): ``drafts`` to propose without the block ("s": a trusted
    copy kept whole last round, "c": a long-matching one), or ``conts`` for the block to vet ("v"), or nothing."""

    kind: str = ""
    start: int = -1
    drafts: list[int] = field(default_factory=list)
    conts: list[tuple[int, list[int]]] = field(default_factory=list)
    copied: int = 0               # the drafts' copied tokens (``follow``; ``padded`` may add repeats after them)
    trusted: int = -1             # "v": the continuation trusted unless the block agrees with one at every slot


def best_depth(q, round_ms) -> int:
    """``dflash2.best_depth`` (same arithmetic; that module needs torch): the k maximizing E[tokens] / round_ms[k]."""

    alive = np.concatenate([[1.0], np.cumprod(np.asarray(q, dtype=np.float64))])
    expect = np.cumsum(alive)
    ms = np.asarray(round_ms[:len(q) + 1], dtype=np.float64)
    return int(np.argmax(expect / ms))


def walk(tokens: np.ndarray, slot: Slot, most: int, confidence: float = 0.0, round_ms=None,
         prefix: Sequence[int] = ()) -> tuple[list[int], list[int]]:
    """DSpark's chain over ``tokens`` [D, K] (``dspark.Drafter.chain``'s rule): (drafts, their candidate indices).
    ``prefix``: candidate indices forced at the first slots (a copy's agreed tokens), whose confidences count in the
    product; the chain continues after them to ``most`` drafts in all (at least the prefix)."""

    depth = min(max(int(most), len(prefix)), tokens.shape[0])
    idx: list[int] = []
    conf: list[float] = []
    alive = 1.0
    for d in range(depth):
        pick, c = slot(d, idx[-1] if d else -1)
        j = int(prefix[d]) if d < len(prefix) else int(np.argmax(pick))
        conf.append(c)
        if d >= len(prefix) and round_ms is None and confidence > 0:
            alive *= c
            if d > 0 and alive < confidence:
                conf.pop()
                break
        elif round_ms is None and confidence > 0:
            alive *= c
        idx.append(j)
    if round_ms is not None:
        idx = idx[:max(best_depth(conf, round_ms), len(prefix))]
    return [int(tokens[d, j]) for d, j in enumerate(idx)], idx


def agreement(tokens: np.ndarray, slot: Slot, cont: Sequence[int], rank: int = RANK) -> list[int]:
    """Candidate indices of ``cont``'s leading tokens that DSpark agrees with: at slot d (prev: the copy's token
    d - 1) the copy's token is a candidate the target's rule keeps, and fewer than ``rank`` candidates pick before it
    (higher score, or the same score at a lower index: ``np.argmax``'s order)."""

    idx: list[int] = []
    for d in range(min(len(cont), tokens.shape[0])):
        pick, _ = slot(d, idx[-1] if d else -1)
        where = np.flatnonzero(tokens[d] == int(cont[d]))
        if not where.size:
            break
        j = int(where[0])
        p = pick[j]
        if not np.isfinite(p) or int((pick > p).sum()) + int((pick[:j] == p).sum()) >= rank:
            break
        idx.append(j)
    return idx


def hybrid(tokens: np.ndarray, slot: Slot, conts: Sequence[Sequence[int]], *, chain_most: int,
           confidence: float = 0.0, round_ms=None, rank: int = RANK, most: int = MOST
           ) -> tuple[list[int], str, int, int]:
    """A round's drafts from DSpark's block (``tokens`` [D, K], ``slot``) and copy continuations ``conts`` (latest
    occurrence first): (drafts, kind, the continuation used (-1: none), its agreed slots).

    * no continuation agrees at slot 0: DSpark's own chain (``chain_most`` slots, ``confidence`` / ``round_ms``),
      kind "d";
    * the best continuation (most agreed slots; ties: the latest) agrees at every one of the block's D slots: the
      copy, up to ``most`` drafts (past the block's slots: what followed the occurrence), kind "x";
    * else its agreed prefix, then DSpark's chain conditioned on it (``walk`` with the prefix forced; no more than
      ``chain_most`` drafts unless the prefix is longer), kind "h".
    """

    D = tokens.shape[0]
    best, agreed = -1, []
    for k, c in enumerate(conts):
        a = agreement(tokens, slot, c, rank)
        if len(a) > len(agreed) or (a and len(a) == len(agreed) and len(c) > len(conts[best])):
            best, agreed = k, a
    if not agreed:
        return walk(tokens, slot, chain_most, confidence, round_ms)[0], "d", -1, 0
    c = conts[best]
    if len(agreed) == D:
        return [int(t) for t in c[:most]], "x", best, D
    drafts, _ = walk(tokens, slot, chain_most, confidence, round_ms, prefix=agreed)
    return drafts[:max(most, len(agreed))], "h", best, len(agreed)
