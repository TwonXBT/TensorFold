"""glm_moe_dsa's restored tap buffers: the residual stream at the tap points (the INPUT of layers
2, 20, 39, 58, 75 for DSpark = after layers 1, 19, 38, 57, 74) is captured for every prompt-chunk
and verify-window row, the capture changes no reply's bits, and tap_rows still refuses when no tap
points were set. One GPU plays world 1 over the tiny synthetic EXL3 checkpoint (glm_dsa_fakes)."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

TAPS = (0, 2)              # after layers 0 and 2 of the fake's 3 (any subset; DSpark uses 5 of 76)


def _checkpoint(tmp_path, monkeypatch):
    """The tiny DSA checkpoint with CUDA-sized dims: the fake's defaults suit the CPU tests, but
    the kernels need normed widths in multiples of 64 (rmsnorm's 64-group sums reshape) and EXL3
    expert tiles of 64 — q_lora_rank 32 fails _rmsnorm's SUMS path, moe_intermediate_size 32 has
    no expert tile. 128 / 64 are tests/cuda/test_glm_engine.py's sizes."""

    import glm_dsa_fakes

    monkeypatch.setitem(glm_dsa_fakes.CFG, "q_lora_rank", 128)
    monkeypatch.setitem(glm_dsa_fakes.CFG, "qk_rope_head_dim", 16)   # qk_dim 16 + 16 = 32: a power of
    monkeypatch.setitem(glm_dsa_fakes.CFG, "moe_intermediate_size", 128)  # 2 (latent._absorb_q's arange);
    return glm_dsa_fakes.write_checkpoint(tmp_path)                   # experts a multiple of 128 (epilogue)


def _engine(folder, monkeypatch, *, taps, capacity: int = 64):
    monkeypatch.setenv("TF_TP_WORLD", "1")    # weights.load reads it per call: one rank, whole table
    from tensorfold.families.glm_moe_dsa.cuda import decode, weights

    w = weights.load(folder, rank=0)
    w.comm = None
    return decode.Engine(w, capacity=capacity, max_rows=8, prefill_rows=16, graphs=False,
                         long_context=True, taps=taps), w


def _snapshots(e, tokens):
    """Run a forward recording b.x after every layer (a manual reference for the tap buffers)."""

    from tensorfold.families.glm_moe_dsa.cuda import forward as fwd

    got: dict[int, torch.Tensor] = {}
    plain = fwd.layer_forward

    def spy(layer, w, st, b, R, *a, **kw):
        plain(layer, w, st, b, R, *a, **kw)
        got[layer.index] = b.x[:R].clone()

    fwd.layer_forward = spy
    try:
        logits = e.forward(tokens)
    finally:
        fwd.layer_forward = plain
    return got, logits


def test_taps_hold_the_residual_at_the_tap_points(tmp_path, monkeypatch):
    """tap_rows equals the residual stream recorded after each tap layer, for a prompt chunk (the
    prefill buffers) and for a decode window (the decode buffers), every row."""

    from tensorfold.families.glm_moe_dsa.cuda import forward as fwd
    from tensorfold.families.glm_moe_dsa.cuda.decode import prefill

    e, _ = _engine(_checkpoint(tmp_path, monkeypatch), monkeypatch, taps=TAPS)
    D = e.w.cfg.hidden

    # a prompt chunk: prefill runs compute on the prefill buffers; the spy records x after each layer
    prompt = [1, 2, 3, 4, 5, 6, 7]                     # 7 rows: one chunk under prefill_rows=16
    got = {}
    plain = fwd.layer_forward

    def spy(layer, w, st, b, R, *a, **kw):
        plain(layer, w, st, b, R, *a, **kw)
        got[layer.index] = b.x[:R].clone()

    fwd.layer_forward = spy
    try:
        first = prefill(e, prompt, None)
    finally:
        fwd.layer_forward = plain
    torch.cuda.synchronize()
    rows = e.tap_rows(len(prompt), e.pbuf)
    assert rows.shape == (len(prompt), len(TAPS) * D)
    for i, layer in enumerate(TAPS):
        assert torch.equal(rows[:, i * D:(i + 1) * D], got[layer]), layer
    assert set(got) == {0, 1, 2}                       # the spy saw every layer; taps kept only TAPS

    # a decode window at the committed position: 4 rows over the prompt's 7
    window = [first, 9, 10, 11]
    got, logits = _snapshots(e, window)
    torch.cuda.synchronize()
    rows = e.tap_rows(len(window), e.buf)
    for i, layer in enumerate(TAPS):
        assert torch.equal(rows[:, i * D:(i + 1) * D], got[layer]), layer
    assert logits.shape[0] == len(window)              # the forward still returned its logits


def test_replies_are_byte_identical_with_and_without_taps(tmp_path, monkeypatch):
    """Setting tap points changes no reply's bits: greedy and seeded-sampled replies, prompt and
    decode, are bit-for-bit the same with taps set as without them."""

    from tensorfold.families.glm_moe_dsa.cuda.decode import prefill, serial_decode

    folder = _checkpoint(tmp_path, monkeypatch)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for sampling in (None, Sampling(seed=1234, temperature=0.8, top_k=40, top_p=0.9, min_p=0.0)):
        outs = []
        for taps in ((), TAPS):
            e, _ = _engine(folder, monkeypatch, taps=taps)
            first = prefill(e, prompt, sampling)
            res = serial_decode(e, first, 24, sampling)
            outs.append((first, res.tokens))
        assert outs[0] == outs[1]


def test_tap_rows_still_refuses_without_tap_points(tmp_path, monkeypatch):
    """No tap points set: tap_rows raises (a drafter cannot attach), as before the restore."""

    e, _ = _engine(_checkpoint(tmp_path, monkeypatch), monkeypatch, taps=())
    e.forward([1, 2, 3])
    with pytest.raises(ValueError, match="no tap points"):
        e.tap_rows(2)
