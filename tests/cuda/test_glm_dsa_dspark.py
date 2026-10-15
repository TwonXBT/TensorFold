"""The DSpark drafter on a tiny synthetic checkpoint: the loader's split and memory estimate, the
context (add_taps) and block pass (propose) on one GPU rank, and the two helpers extracted from
DFlash2 (merge_candidates, best_depth) pinned against numpy references. Sizes are scaled down from
~/models/GLM-5.3-speculator.dspark (D 6144 -> 128, heads 64 -> 4, layers 3 -> 2, window 2048 -> 64,
markov_rank 256 -> 16, vocab 154880 -> 256); the config fields and tensor names are the real ones."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.glm5_next.cuda.qmm import make_b16  # noqa: E402

D, V = 128, 256
HEADS, HD, INTER, LAYERS = 4, 32, 128, 2
WINDOW, BLOCK, RANK = 64, 8, 16
AUX = (1, 2)                       # inputs of layers 1 and 2 -> tap points after 0 and 1

CONFIG = {
    "speculators_model_type": "dspark", "speculators_version": "0.7.0.dev108",
    "aux_hidden_state_layer_ids": list(AUX), "block_size": BLOCK, "confidence_head_with_markov": True,
    "draft_vocab_size": V, "dtype": "bfloat16", "enable_confidence_head": True, "markov_head_type": "vanilla",
    "markov_rank": RANK, "mask_token_id": V - 1, "sample_from_anchor": True, "sliding_window_non_causal": False,
    "transformer_layer_config": {
        "head_dim": HD, "hidden_size": D, "intermediate_size": INTER, "layer_types": ["sliding_attention"] * LAYERS,
        "max_position_embeddings": 1048576, "num_attention_heads": HEADS, "num_hidden_layers": LAYERS,
        "num_key_value_heads": HEADS, "rms_norm_eps": 1e-05, "rope_parameters": {"rope_theta": 8000000.0},
        "sliding_window": WINDOW, "use_sliding_window": True, "vocab_size": V,
    },
}


def _checkpoint(path):
    """The tiny DSpark speculator: every tensor ``expected_tensors`` names, seeded bf16."""

    from tensorfold.families.glm_moe_dsa.cuda.dspark import DSparkConfig, expected_tensors

    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(CONFIG))
    g = torch.Generator().manual_seed(7)
    tensors = {name: torch.randn(shape, generator=g).to(torch.bfloat16)
               for name, shape in expected_tensors(DSparkConfig.read(path)).items()}
    save_file(tensors, str(path / "model.safetensors"))
    return path


def _verifier(dev):
    """The verifier's pieces the drafter reads: whole-table embedding, a BF16 head, one rank."""

    g = torch.Generator().manual_seed(11)
    return type("W", (), {
        "device": dev, "rank": 0, "world": 1, "comm": None, "cfg": type("C", (), {"vocab": V})(),
        "embed": torch.randn((V, D), generator=g).to(torch.bfloat16).to(dev),
        "head": make_b16(torch.randn((V, D), generator=g).to(torch.bfloat16).to(dev)),
        "draft_head": None, "vocab_offset": 0,
    })()


def _drafter(path, w, **kw):
    from tensorfold.families.glm_moe_dsa.cuda.dspark import Drafter

    return Drafter(path, w, capacity=256, **kw)


def test_config_and_split(tmp_path):
    """The config read: tap points after layers 0 and 1, the causal sliding window, the vanilla Markov head."""

    from tensorfold.families.glm_moe_dsa.cuda.dspark import DSparkConfig, draft_parts, memory, ring_rows

    c = DSparkConfig.read(_checkpoint(tmp_path))
    assert c.tap_layers == (0, 1)
    assert (c.window, c.block, c.causal, c.markov_rank, c.confidence, c.confidence_markov) == \
        (WINDOW, BLOCK, True, RANK, True, True)
    assert draft_parts(HEADS, HEADS, INTER, 1) == ([HEADS], [HEADS], [INTER])
    assert draft_parts(HEADS, HEADS, 512, 2) == ([2, 2], [2, 2], [256, 256])   # the MLP in units of 128
    assert draft_parts(HEADS, HEADS, 512, 4) == ([1, 1, 1, 1], [1, 1, 1, 1], [128, 128, 128, 128])
    assert draft_parts(64, 64, 12288, 4) == ([16] * 4, [16] * 4, [3072] * 4)   # the real checkpoint's split
    assert ring_rows(c) == 192                       # the window, the block and a tile: ceil((63+8+63)/64)*64
    m = memory(c, 0, 1, capacity=256)
    assert m["host"] == 2 * V * RANK * 2             # the replicated Markov tables, bf16


def test_load_and_memory(tmp_path):
    """The loader: every expected tensor lands (q4 copies), and nbytes matches the memory estimate."""

    from tensorfold.families.glm_moe_dsa.cuda.dspark import DSparkConfig, memory

    dev = torch.device("cuda")
    path = _checkpoint(tmp_path)
    c = DSparkConfig.read(path)
    w = _verifier(dev)
    d = _drafter(path, w)
    assert (d.heads, d.kvh, d.inter, d.block, d.top_k) == (HEADS, HEADS, INTER, BLOCK, 64)
    assert d.tap_layers == (0, 1)
    assert d.quant == "q4"
    assert d.w1.shape == (V, RANK) and d.w1.dtype == torch.bfloat16 and d.w1.device.type == "cpu"
    assert d.conf_h is not None and d.conf_h.shape == (D,) and d.conf_m is not None and len(d.conf_m) == RANK
    # nbytes: the exact q4 arithmetic (words + scales + biases = n * k * 9/16) of the matrices and
    # the K/V context at the drafter's flat cap (ring off: capacity + block rows); the small bf16
    # norms are not counted (Mia's nbytes, kept)
    q4 = lambda n, k: n * k * 9 // 16
    weights = q4(D, len(AUX) * D) + LAYERS * (
        q4(HEADS * HD + 2 * HEADS * HD, D) + q4(D, HEADS * HD) + q4(2 * INTER, D) + q4(D, INTER)
        + q4(2 * HEADS * HD, D))
    kv = 2 * LAYERS * HEADS * (256 + BLOCK) * HD * 2
    assert d.nbytes() == weights + kv, (d.nbytes(), weights + kv)
    # memory(): the capacity-planning estimate — rows padded to 128 and the fixed buffers counted wide
    est = memory(c, 0, 1, capacity=256, ring=False)
    assert d.nbytes() <= est["device"] and est["host"] == 2 * V * RANK * 2


def test_context_and_propose(tmp_path):
    """add_taps commits context rows (the ring's slots), then propose drafts a chain: ids in range, the
    learned confidence recorded, greedy and seeded runs both work, and depth / round_ms cut it."""

    from tensorfold.families.glm_moe_dsa.cuda.dspark import Drafter

    dev = torch.device("cuda")
    path = _checkpoint(tmp_path)
    w = _verifier(dev)
    d = _drafter(path, w)
    d.capture()                                      # the tap graphs (1..block) and the block graph
    taps = torch.randn((9, len(AUX) * D), dtype=torch.bfloat16, device=dev)
    d.add_taps(taps[:5])
    d.add_taps(taps[5:])
    assert d.context_end == 9
    drafts = d.propose(3, 8, None, confidence=0.0)
    assert 1 <= len(drafts) <= 8 and all(0 <= t < V for t in drafts)
    assert len(d.last_confidence) == len(drafts)
    from tensorfold.engine.exact_sampling import Sampling

    s = Sampling(seed=1234, temperature=0.8, top_k=40, top_p=0.9, min_p=0.0)
    sampled = d.propose(3, 8, s, confidence=0.0)
    assert 1 <= len(sampled) <= 8 and all(0 <= t < V for t in sampled)
    high = d.propose(3, 8, None, confidence=0.99)
    assert 1 <= len(high) <= len(drafts) and drafts[:len(high)] == high   # a prefix; the first always kept
    flat = d.propose(3, 8, None, confidence=0.0, round_ms=(1.0,) * 9)
    assert 1 <= len(flat) <= len(drafts) and drafts[:len(flat)] == flat   # the same chain, best_depth's cut
    none = d.propose(3, 8, None, confidence=0.0, round_ms=(0.01,) + (1e9,) * 8)
    assert none == []                                   # a serial round is cheapest: depth 0 wins


def test_merge_candidates_and_best_depth():
    """The extracted helpers, pinned: merge_candidates against a numpy reference, best_depth's argmax."""

    from tensorfold.families.glm5_next.cuda.dflash2 import best_depth, merge_candidates

    g = torch.randn((2, 3, 2 * 5), dtype=torch.float32)
    ids = torch.randint(0, 1000, (2, 3, 5), dtype=torch.int32)
    packed = torch.cat([g[:, :, :5], ids.view(torch.float32)], dim=2)
    proj = torch.randn((3, 4), dtype=torch.float32)
    tokens, values, p = merge_candidates(packed, proj, 5)
    assert p.shape == (3, 4) and tokens.shape == (3, 5) and values.shape == (3, 5)
    for dd in range(3):
        vals = (-g[:, dd, :5]).flatten().tolist()
        tks = ids[:, dd].flatten().tolist()
        ref = sorted(zip(vals, tks))[:5]                     # value desc, id asc, over both ranks
        assert [t for _, t in ref] == tokens[dd].tolist()
        assert [-v for v, _ in ref] == values[dd].tolist()
    q = [0.9, 0.5, 0.5]
    ms = (1.0, 1.2, 1.4, 1.6)
    alive = np.concatenate([[1.0], np.cumprod(q)])
    expect = int(np.argmax(np.cumsum(alive) / np.asarray(ms)))
    assert best_depth(q, ms) == expect
    assert best_depth(q, (1.0, 1.0, 1.0, 1.0)) == 3   # flat cost: every draft pays
