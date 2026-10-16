"""The engine's --drafter wiring for a DSpark speculator: the boot (capacity plan, settings comparison,
the Decoder's tap points, the drafter's own graphs), the prefill feed (a prompt's committed rows reach the
drafter chunk by chunk), and the prompt path's tap rows against the decode path's for the same tokens —
the deferred-residual fill's real check. The micro-batch prompt path (forward.prompt_layers) needs
world > 2 and a chunk of at least OVERLAP_ROWS (256), so that comparison only gets exercised at TP4;
at world 1 both sides run the plain layer loop and the test still checks the feed and the fill's placement.

One GPU plays every rank through a comm that answers each all_gather with n copies of this rank's data
(the TP numerics are wrong the same way on both sides of every comparison; only the plumbing is under test)."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

D, V = 128, 256
AUX = (1, 2)                       # the speculator taps the inputs of the target's layers 1 and 2
BLOCK, WINDOW = 8, 64
PROMPT = 300                       # one 256-row chunk (the micro-batch path's minimum) then a 44-row tail


class _NCopies:
    """Rank 0 of n on one GPU: every all-gather returns this rank's input n times."""

    def __init__(self, n: int):
        self.rank, self.world = 0, n

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        n = send.numel()
        flat = recv.view(-1)
        for r in range(self.world):
            flat[r * n:(r + 1) * n].copy_(send.reshape(-1))

    def grouped(self, sends, recvs) -> None:
        """Every rank is this one: each peer's recv gets that peer's send (this rank's slice for it)."""

        by = {peer: t for t, peer in recvs}
        for t, peer in sends:
            r = by.get(peer)
            if t.numel() and r is not None and r.numel() == t.numel():
                r.copy_(t)

    def barrier(self) -> None:
        torch.cuda.synchronize()

    def ready(self, label: str, **kwargs) -> None:
        pass                                  # the other ranks are this one


def _dsa_checkpoint(path, monkeypatch, world: int):
    """The tiny DSA target with CUDA-sized dims (the taps test's), its indexer heads splitable by the world."""

    import glm_dsa_fakes

    monkeypatch.setitem(glm_dsa_fakes.CFG, "q_lora_rank", 128)
    monkeypatch.setitem(glm_dsa_fakes.CFG, "qk_rope_head_dim", 16)
    # index_topk 512: the whole 300-token test stays in dense attention (the fake's sparse indexer over a
    # 2,560-slot cache reads never-written rows past the prompt — garbage taps, not a wiring fault; the
    # sparse path is the taps test's and the container TP4 run's to prove)
    monkeypatch.setitem(glm_dsa_fakes.CFG, "index_topk", 512)
    # the routed experts' kernel needs a multiple of 128 columns a rank (512: 128 at world 4), the o_proj's
    # bf16 matmul a K of at least 64 (v_head 64: 64 a rank at world 4), and every swiglu a width of at least
    # 64 (the dense MLP's 256: 64 a rank at world 4)
    monkeypatch.setitem(glm_dsa_fakes.CFG, "moe_intermediate_size", 512 if world > 1 else 128)
    if world > 1:
        monkeypatch.setitem(glm_dsa_fakes.CFG, "v_head_dim", 64)
        monkeypatch.setitem(glm_dsa_fakes.CFG, "intermediate_size", 256)
    glm_dsa_fakes.write_checkpoint(path)
    return path


def _dspark_checkpoint(path, world: int):
    """The tiny DSpark speculator beside the target: the same hidden size and vocabulary (no d2t), its MLP
    wide enough to split over the world in units of 128."""

    from tensorfold.families.glm_moe_dsa.cuda.dspark import DSparkConfig, expected_tensors

    inter = 512
    cfg = {
        "speculators_model_type": "dspark", "aux_hidden_state_layer_ids": list(AUX), "block_size": BLOCK,
        "confidence_head_with_markov": True, "draft_vocab_size": V, "enable_confidence_head": True,
        "markov_head_type": "vanilla", "markov_rank": 16, "mask_token_id": V - 1, "sample_from_anchor": True,
        "sliding_window_non_causal": False,
        "transformer_layer_config": {
            "head_dim": 64, "hidden_size": D, "intermediate_size": inter,
            "layer_types": ["sliding_attention"] * 2, "num_attention_heads": 4, "num_hidden_layers": 2,
            "num_key_value_heads": 4, "rms_norm_eps": 1e-05, "rope_parameters": {"rope_theta": 8000000.0},
            "sliding_window": WINDOW, "use_sliding_window": True, "vocab_size": V,
        },
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(cfg))
    g = torch.Generator().manual_seed(7)
    tensors = {name: torch.randn(shape, generator=g).to(torch.bfloat16)
               for name, shape in expected_tensors(DSparkConfig.from_dict(cfg)).items()}
    save_file(tensors, str(path / "model.safetensors"))
    return path


def _engine(root, monkeypatch, world: int, *, drafter: bool, context: int = PROMPT):
    """GlmEngine on the tiny pair, one GPU playing every rank; the drafter beside the target when asked.
    TENSORFOLD_MEMORY_RESERVE_GIB=2: the default reserve (a tenth of the 121-GB host) leaves this
    throwaway container beside the live model a budget too small for any window; the tiny model's
    actual allocations are unaffected — the knob only widens the startup estimate's headroom."""

    from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine

    monkeypatch.setenv("TF_TP_WORLD", str(world))
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "2")
    # the tiny drafter's per-rank slices stay under a q4 group's 64 columns at world 4 (o_proj 64, down 128);
    # bf16 copies keep the test independent of the quantizer's tile minimums (the wiring is under test —
    # the real checkpoint's 1,024-column slices quantize fine)
    monkeypatch.setenv("TF_GLM_DSPARK_QUANT", "bf16")
    _dsa_checkpoint(root / "model", monkeypatch, world)
    if drafter:
        _dspark_checkpoint(root / "dspark", world)
    return GlmEngine(root / "model", rank=0, master="", port=0, comm=_NCopies(world),
                     drafter=root / "dspark" if drafter else None,
                     context=context, context_explicit=True, prefill_rows=256)


def test_the_drafter_boots_and_the_prompt_feeds_it(tmp_path, monkeypatch):
    """The boot: the capacity plan counts the drafter's bytes, the Decoder holds its tap points, its graphs
    captured; a 300-token prefill leaves context_end at 300 with the position aligned, chunk by chunk."""

    from tensorfold.families.glm_moe_dsa.cuda.dspark import dspark_weights

    plain = _engine(tmp_path / "plain", monkeypatch, 1, drafter=False)
    with_drafter = _engine(tmp_path / "dspark", monkeypatch, 1, drafter=True)
    assert with_drafter.drafter is not None and with_drafter.dspark
    assert plain.drafter is None and not plain.dspark
    assert with_drafter.e.buf.taps and not plain.e.buf.taps
    assert with_drafter.e.buf.tap_at == {0: [0], 1: [1]}         # tap_layers (0, 1) -> slots in layer order
    extra = with_drafter.capacity_plan["total_bytes_estimate"] - plain.capacity_plan["total_bytes_estimate"]
    assert extra > dspark_weights(tmp_path / "dspark" / "dspark", 1).resident * 0.5   # weights + a slot's context

    from tensorfold.families.glm_moe_dsa.cuda.decode import prefill

    prompt = list(range(1, PROMPT + 1))
    first = prefill(with_drafter.e, prompt, None, drafter=with_drafter.drafter)
    assert 0 <= first < V
    d = with_drafter.drafter
    torch.cuda.synchronize()
    assert int(d.pos_dev.item()) == d.context_end == PROMPT
    assert d.context_end == with_drafter.e.st.pos
    drafts = d.propose(first, 8, None, confidence=0.0)
    assert 1 <= len(drafts) <= 8 and all(0 <= t < V for t in drafts)


def test_one_rank_without_the_drafter_is_refused(tmp_path, monkeypatch):
    """The settings comparison: a rank that booted without the drafter next to one that booted with it is
    refused at startup (the drafts would differ, so the verify rounds would diverge)."""

    class _SplitWorld:
        """Rank 0 with the drafter, rank 1 without: the first gather (admit's status) returns this rank's
        values twice — rank 1's plan is the same, it just booted without the drafter — and every later one
        (the settings comparison) returns this rank's settings, then rank 1's without the drafter."""

        rank, world = 0, 2
        gathers = 0

        def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
            n = send.numel()
            flat = recv.view(-1)
            flat[:n].copy_(send.reshape(-1))
            if self.gathers:
                flat[n:2 * n].copy_(send.reshape(-1))
                flat[n:n + 2] = 0            # rank 1's drafter flags: booted without the speculator
            else:
                flat[n:].copy_(send.reshape(-1))
            self.gathers += 1

        def barrier(self) -> None:
            torch.cuda.synchronize()

        def ready(self, label: str, **kwargs) -> None:
            pass

    from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine

    monkeypatch.setenv("TF_TP_WORLD", "2")
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "2")
    root = tmp_path / "split"
    _dsa_checkpoint(root / "model", monkeypatch, 2)
    _dspark_checkpoint(root / "dspark", 2)
    with pytest.raises(RuntimeError, match="different settings"):
        GlmEngine(root / "model", rank=0, master="", port=0, comm=_SplitWorld(),
                  drafter=root / "dspark", context=PROMPT, context_explicit=True, prefill_rows=256)


@pytest.mark.parametrize("world", [1, 4], ids=["plain-loop", "micro-batch-tp4"])
def test_a_prompt_chunks_tap_rows_equal_the_decode_paths(tmp_path, monkeypatch, world):
    """The same tokens at the same positions. At world 1 the prompt path's tap rows (the plain loop) match a
    decode-path forward's (8-row windows) to bf16 reduction-order ulps. At world 4 the 256-row chunk takes the
    micro-batch path (prompt_layers — only exercised at TP4: it needs world > 2, a comm and OVERLAP_ROWS rows),
    and the reference is a spy of that path itself: x recorded right after each layer's attention halves — the
    exact point the deferred-residual fill reads — so the comparison is bitwise and checks the two things that
    can break, the key (index - 1 -> the right slot) and the rows ([:R] across the micro-batch boundary). A
    decode-path reference cannot serve at world 4 on one GPU: the paths exchange partials through different
    collectives (grouped point-to-point vs all-gather) that a same-data comm fake cannot keep consistent."""

    from tensorfold.families.glm_moe_dsa.cuda import forward as fwd

    eng = _engine(tmp_path / f"w{world}", monkeypatch, world, drafter=True)
    e = eng.e
    tokens = list(range(1, PROMPT + 1))

    # the prompt path over the tokens: one 256-row chunk (the micro-batch path at world 4) and the 44-row
    # tail, each committed as prefill commits them; a spy records x after every attention half
    seen: dict[int, torch.Tensor] = {}       # layer index -> x after its attn halves (both micro-batches)
    plain_attn = fwd.attn_part
    taken = {"prompt_layers": 0}

    def spy_attn(layer, w, st, b, R, *a, **kw):
        out = plain_attn(layer, w, st, b, R, *a, **kw)
        torch.cuda.synchronize()              # the micro-batch path's writes ride the comm stream
        # a micro-batch call sees only its half's view; b.full.x holds the whole chunk, complete once the
        # second half's attention has run — the exact point fill_taps reads
        src = b.full.x[:b.chunk] if getattr(b, "micro", False) else b.x[:R]
        seen[layer.index] = src.clone()
        return out

    plain_prompt = fwd.prompt_layers

    def spy_prompt(w, st, b, R, nch, host_pos, sparse_np):
        taken["prompt_layers"] += 1
        return plain_prompt(w, st, b, R, nch, host_pos, sparse_np)

    e.reset()
    prompt_rows = []
    fwd.attn_part = spy_attn
    fwd.prompt_layers = spy_prompt
    try:
        for start in range(0, PROMPT, e.prefill_rows):
            chunk = tokens[start:start + e.prefill_rows]
            seen.clear()                     # the reference is per chunk: the tail's plain loop overwrites x
            R = fwd.stage(e.w, e.st, e.pbuf, chunk)
            fwd.compute(e.w, e.st, e.pbuf, R, nch=fwd.chunks_for(e.st, R), host_pos=e.st.pos)
            rows = e.tap_rows(R, e.pbuf).clone()
            if world > 2:
                # the fill's key: the slot keyed k holds the stream the spy saw after layer k + 1's
                # attention halves — the deferred residual makes that x layer k + 1's input, i.e. the
                # stream after layer k (the tail chunk's plain loop reads the same point)
                for key in sorted(e.pbuf.tap_at):
                    if key + 1 not in seen:
                        continue               # a tap after the final layer: filled past the loop, no key+1
                    slot = e.pbuf.tap_at[key][0]
                    got, want = rows[:, slot * D:(slot + 1) * D], seen[key + 1]
                    if not torch.equal(got, want):
                        bad = (got.view(torch.uint16) != want.view(torch.uint16)).any(dim=1) \
                            .nonzero().flatten()
                        raise AssertionError(f"tap slot {key} differs from the stream after layer {key}: "
                                             f"{bad.numel()} of {R} rows, first at {int(bad[0])}, "
                                             f"rows {' '.join(str(int(r)) for r in bad[:8].tolist())}")
            prompt_rows.append(rows)
            fwd.commit(e.w, e.st, e.pbuf, R, R)
    finally:
        fwd.attn_part = plain_attn
        fwd.prompt_layers = plain_prompt
    prompt = torch.cat(prompt_rows, dim=0)
    assert prompt.shape == (PROMPT, len(AUX) * D)
    if world > 2:
        assert taken["prompt_layers"] == 1          # the 256-row chunk took the micro-batch path (TP4 only)
    else:
        assert taken["prompt_layers"] == 0          # the plain loop (world 1: no micro-batch)
        # the decode path's rows for the same tokens: 8-row windows, each committed
        e.reset()
        decode_rows = []
        for start in range(0, PROMPT, 8):
            window = tokens[start:start + 8]
            R = fwd.stage(e.w, e.st, e.buf, window)
            fwd.compute(e.w, e.st, e.buf, R, nch=fwd.chunks_for(e.st, R), host_pos=e.st.pos)
            torch.cuda.synchronize()          # eager (no graph): the comm stream's embed reduce must land first
            decode_rows.append(e.tap_rows(R, e.buf).clone())
            fwd.commit(e.w, e.st, e.buf, R, R)
        decode = torch.cat(decode_rows, dim=0)
        assert decode.shape == prompt.shape
        # the fast prompt kernels and the decode kernels reduce in different orders: bf16 ulps (one row of
        # the 300), not a placement fault — a wrong tap point reads a different layer's stream, orders away.
        # equal_nan: the fake's random weights overflow bf16 at depth, identically on both sides.
        close = torch.allclose(prompt.float(), decode.float(), atol=1e-3, rtol=2e-2, equal_nan=True)
        if not close:
            bad = (~torch.isclose(prompt.float(), decode.float(), atol=1e-3, rtol=2e-2, equal_nan=True)) \
                .any(dim=1).nonzero().flatten()
            far = ~torch.isclose(prompt.float(), decode.float(), atol=1e-3, rtol=2e-2, equal_nan=True)
            diff = (prompt.float() - decode.float()).abs().nan_to_num(0.0)
            i, j = divmod(int(diff.argmax()), diff.shape[1])
            raise AssertionError(f"the prompt path's tap rows differ from the decode path's: {bad.numel()} rows, "
                                 f"first at {int(bad[0]) if bad.numel() else -1}; {int(far.sum())} of "
                                 f"{far.numel()} values; worst [{i},{j}] {float(prompt[i, j])} vs "
                                 f"{float(decode[i, j])}; row {i} max |x| {float(prompt[i].float().abs().max())}")
    # and the drafter, fed by a real prefill, holds the same context either way
    from tensorfold.families.glm_moe_dsa.cuda.decode import prefill

    e.reset()
    eng.drafter.reset()
    prefill(e, tokens, None, drafter=eng.drafter)
    torch.cuda.synchronize()
    assert int(eng.drafter.pos_dev.item()) == eng.drafter.context_end == PROMPT


def test_a_resumed_prompt_keeps_the_drafters_context(tmp_path, monkeypatch):
    """TF_GLM_KEEP_SLOTS: a prompt extending the last one resumes the target at a chunk boundary, and the drafter
    rolls back to the same point (Drafter.resume) instead of starting over — its context then equals a fresh
    prefill's of the whole prompt (same candidates, bit for bit: the rows before the cut came from the same chunk).
    A small resume spare (TF_GLM_DSPARK_RESUME_ROWS=64) keeps the drafter in ring mode, so the wrap bound is live;
    a gap past it resets instead."""

    import numpy as np

    from tensorfold.families.glm_moe_dsa.cuda import multi
    from tensorfold.families.glm_moe_dsa.cuda.decode import prefill

    monkeypatch.setattr(multi, "KEEP", True)
    monkeypatch.setenv("TF_GLM_DSPARK_RESUME_ROWS", "64")
    eng = _engine(tmp_path / "keep", monkeypatch, 1, drafter=True)
    e, d = eng.e, eng.drafter
    assert d.ring and d.ring - d.span - d.block >= 40          # ring mode, with room for this test's gap

    first_a = prefill(e, list(range(1, 281)), None, drafter=d)  # A: 280 tokens
    assert d.context_end == 280
    whole = list(range(1, 281)) + list(range(7, 27))            # A + 20: shares A, resumes at the chunk boundary
    first = prefill(e, whole, None, drafter=d)
    cut = e.cached
    assert 0 < cut < 280 and d.context_end == len(whole)       # rolled back to cut, then fed the rest
    torch.cuda.synchronize()
    assert int(d.pos_dev.item()) == len(whole)
    # the drafter's whole K/V ring, as raw bits (the fake's random weights overflow to NaN at depth, the same on
    # both sides: compare bits, not values) — the context the drafts read, before a block pass writes its rows
    ring_resumed = [t.view(torch.int16).clone() for t in d.kc + d.vc]
    resumed = d.candidates(first, BLOCK - 1)

    e.kept = None                                               # the fresh reference: no resume
    again = prefill(e, whole, None, drafter=d)
    assert e.cached == 0 and again == first and d.context_end == len(whole)
    ring_fresh = [t.view(torch.int16).clone() for t in d.kc + d.vc]
    fresh = d.candidates(again, BLOCK - 1)
    assert all(torch.equal(a, b) for a, b in zip(ring_resumed, ring_fresh))
    assert np.array_equal(resumed[0], fresh[0])
    assert np.array_equal(resumed[1], fresh[1], equal_nan=True)
    assert np.array_equal(resumed[2], fresh[2], equal_nan=True)

    # a gap past the ring's spare cannot keep the window: the position still moves to the cut (stale context)
    end = d.context_end
    cut = end - (d.ring - d.span - d.block) - 1
    assert not d.resume(cut)
    assert d.context_end == cut and int(d.pos_dev.item()) == cut
    del first_a


def test_dspark_refuses_parallel_streams(tmp_path, monkeypatch):
    """--parallel > 1 with a DSpark --drafter is refused before the capacity plan (it drafts single-stream only)."""

    from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine

    monkeypatch.setenv("TF_TP_WORLD", "1")
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "2")
    root = tmp_path / "par"
    _dsa_checkpoint(root / "model", monkeypatch, 1)
    _dspark_checkpoint(root / "dspark", 1)
    with pytest.raises(ValueError, match="single-stream"):
        GlmEngine(root / "model", rank=0, master="", port=0, comm=_NCopies(1), drafter=root / "dspark",
                  context=PROMPT, context_explicit=True, prefill_rows=256, parallel=2)


@pytest.mark.parametrize("sampled", [False, True], ids=["greedy", "seeded"])
def test_dspark_replies_equal_the_serial_reply(tmp_path, monkeypatch, sampled):
    """The exactness argument made executable: the reply of dflash_decode (DSpark drafts, with and without copy
    drafts) is the serial reply, token for token, greedy and seeded-sampled; the drafter's context tracks the
    committed prefix every round (context_end == st.pos at the end)."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.glm5_next.cuda.decode import DepthPolicy
    from tensorfold.families.glm_moe_dsa.cuda.copy_drafts import CopySettings
    from tensorfold.families.glm_moe_dsa.cuda.decode import dflash_decode, prefill, serial_decode

    eng = _engine(tmp_path / "x", monkeypatch, 1, drafter=True)
    e, d = eng.e, eng.drafter
    sampling = Sampling(seed=1234, temperature=0.8, top_k=40, top_p=0.9, min_p=0.0) if sampled else None
    prompt = [3, 9, 27, 81, 5, 25] * 20                      # a repeating prompt: copy drafts find matches
    n = 40

    first = prefill(e, prompt, sampling, drafter=d)
    want = serial_decode(e, first, n, sampling).tokens

    for copy in (False, True, "hybrid"):
        e.kept = None
        first2 = prefill(e, prompt, sampling, drafter=d)
        assert first2 == first
        env = {"TF_GLM_COPY_DRAFTS": "1", **({"TF_GLM_COPY_HYBRID": "1"} if copy == "hybrid" else {})}
        copies = (CopySettings.from_env(7, env=env).drafts(list(prompt) + [first2], hybrid=copy == "hybrid")
                  if copy else None)
        if copy == "hybrid":
            assert copies.hybrid is not None
        res = dflash_decode(e, d, first2, n, sampling, policy=DepthPolicy(7, fixed=True), copies=copies)
        assert res.tokens == want, (copy, res.tokens, want)
        assert d.context_end == e.st.pos                   # every kept row reached the drafter
        assert res.drafted > 0 and res.rounds <= n


@pytest.mark.parametrize("invariant_on", [False, True], ids=["chunked", "invariant"])
def test_identical_prompt_twice_keeps_the_drafter_aligned(tmp_path, monkeypatch, invariant_on):
    """The live failure 2026-10-16 01:45: the same prompt twice with the conversation cache (resume one row short of
    the prompt) and DSpark decode between them; dflash_decode refused: drafter 361 vs committed 362."""

    from tensorfold.families.glm5_next.cuda.decode import DepthPolicy
    from tensorfold.families.glm_moe_dsa.cuda import invariant, multi
    from tensorfold.families.glm_moe_dsa.cuda.decode import dflash_decode, prefill

    monkeypatch.setattr(multi, "KEEP", True)
    monkeypatch.setattr(invariant, "INVARIANT", invariant_on)
    eng = _engine(tmp_path / "rep", monkeypatch, 1, drafter=True)
    e, d = eng.e, eng.drafter
    prompt = [3, 9, 27, 81, 5, 25] * 15
    for i in range(3):
        first = prefill(e, prompt, None, drafter=d)
        print("round", i, "cut", e.cached, "drafter", d.context_end, "pos", e.st.pos)
        assert d.context_end == e.st.pos, (i, e.cached, d.context_end, e.st.pos)
        dflash_decode(e, d, first, 20, None, policy=DepthPolicy(7, fixed=True))


def test_a_long_prompt_then_an_unrelated_short_one_keeps_the_drafter_aligned(tmp_path, monkeypatch):
    """The live failure 2026-10-16 01:45: a long prompt, then a short unrelated one sharing only the template's first
    tokens: the target resumes at that tiny cut while the drafter's ring cannot keep its window (small resume spare
    here); the drafter must still sit at the cut, then the feed brings it to the target's position."""

    from tensorfold.families.glm5_next.cuda.decode import DepthPolicy
    from tensorfold.families.glm_moe_dsa.cuda import invariant, multi
    from tensorfold.families.glm_moe_dsa.cuda.decode import dflash_decode, prefill

    monkeypatch.setattr(multi, "KEEP", True)
    monkeypatch.setattr(invariant, "INVARIANT", True)
    monkeypatch.setenv("TF_GLM_DSPARK_RESUME_ROWS", "64")
    eng = _engine(tmp_path / "rep2", monkeypatch, 1, drafter=True)
    e, d = eng.e, eng.drafter
    assert d.ring
    long_prompt = [3, 4] + [7, 11, 13, 17, 19] * 56                       # 282 tokens
    first = prefill(e, long_prompt, None, drafter=d)
    dflash_decode(e, d, first, 8, None, policy=DepthPolicy(7, fixed=True))
    short = [3, 4] + [23, 29, 31] * 40      # 122 tokens (past tiny_rows), sharing only the first 2 tokens
    first = prefill(e, short, None, drafter=d)
    assert e.cached == 1                                                     # resumed at a tiny cut
    assert d.context_end == e.st.pos == len(short)
    res = dflash_decode(e, d, first, 12, None, policy=DepthPolicy(7, fixed=True))
    assert d.context_end == e.st.pos and len(res.tokens) == 12
