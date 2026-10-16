"""DSpark drafts (RedHatAI/GLM-5.3-speculator.dspark) for the GLM engine: a DFlash-style parallel block pass over
committed target taps, then a sequential low-rank Markov bias and a learned per-draft confidence. Drafts only
propose; the verify loop keeps drafted == serial, so nothing here can change a reply, only acceptance.

The model (``config.json`` with a nested ``transformer_layer_config``):

* taps: the target residual stream at the INPUT of layers ``aux_hidden_state_layer_ids`` (2, 20, 39, 58, 75), i.e.
  the engine's tap points "after layer" id - 1 (``tap_layers`` = 1, 19, 38, 57, 74, DFlash2's convention);
* context row j: c_j = hidden_norm(fc(concat taps)); per draft layer K_j = RoPE(k_norm(k_proj c_j), j),
  V_j = v_proj c_j (no input_layernorm on the context, every layer reads the same c);
* block: [anchor = pending token, mask x (block - 1)] at positions p .. p + block - 1, embedded with the VERIFIER's
  embed_tokens; plain Qwen3 pre-norm layers (q_norm / k_norm before neox RoPE), causal inside the block, each slot
  sees the context rows [p - sliding_window, p) (the training mask's per-block window) and block slots 0..k;
* h = norm(x), base logits = the VERIFIER's lm_head(h) for every slot (``sample_from_anchor``: slot k predicts the
  token at p + k + 1, so a pass gives ``block`` drafts);
* Markov (vanilla): logits_k += W2 @ W1[prev_k], prev_0 = anchor, prev_k = draft k - 1, applied sequentially on the
  host over the merged top-K base candidates (vLLM's ``dspark_draft_topk`` mode);
* confidence: c_k = sigmoid(proj_fp32([h_k ; W1[prev_k]]) + b); the survival cumprod(c) picks the depth
  (``dflash2.best_depth`` with ``round_ms``) or stops at the ``confidence`` product threshold like DFlash2.

Tensor parallelism over ``world`` ranks as DFlash2 (``draft_parts`` below): query / KV heads in whole groups, MLP
width in units of 128; o_proj / down_proj row-parallel partials summed in rank order (``dflash2.Drafter._row``); fc,
norms, the confidence head and Markov W1 / W2 replicated; each rank's top-K of its vocabulary slice of the verifier
head, merged by value then id (``dflash2.merge_candidates``).

This family's two adaptations from Mia-AiLab's ``glm5_next/cuda/dspark.py`` (Apache-2.0, adapted in turn from
vllm-project/speculators and vllm-project/vllm — see her module docstring for the upstream provenance):

* the verifier's head is an EXL3 ``X3`` here, not a bf16 matrix: the block pass calls it through a drafter-owned
  ``X3Scratch`` (fp32 logits, the -inf padding past the rank's crop ignored by topk), and the embedding is split by
  vocabulary rows per rank — the block pass reuses this family's ``forward.embed`` gather verbatim;
* ``draft_parts`` is local (this tree has no ``tp.py``): ``cuda.geometry.share`` / ``share_lo`` with unit 1 for the
  heads and 128 for the MLP, the same split DFlash2's ``tp.draft_parts`` makes.

Reused from ``dflash2`` (imported, unmodified): ``Drafter`` (subclassed: ``reset``, ``_slots``, ``_rotary``, ``_prep``,
``_norm``, ``_row``, ``_taps_compute``, ``add_taps``, ``capture``, the ring and the tap / block CUDA graphs),
``_prep_kernel`` (via ``_prep``), ``merge_candidates``, ``best_depth``, ``NOISE`` (the keyed-noise weight), ``_quantize4``
and ``uniform_rows`` keyed noise. New here: a causal multi-head block attention kernel (head_dim 64, the block padded
to 16 rows for tl.dot), plain Qwen3 layers, the Markov / confidence chain and the loader.

Adapted (Apache-2.0) from:
* vllm-project/speculators @ 36a19ca74e5e65a4934042df318314239f3e64f0:
  src/speculators/models/dspark/model_definitions.py (MarkovHead, ConfidenceHead),
  src/speculators/models/dspark/core.py (prev-token alignment for sample_from_anchor, confidence features),
  src/speculators/models/dflash/core.py and dflash/attention.py (anchored-block forward, the anchor-block mask:
  context before the anchor within the sliding window, causal within the block),
  src/speculators/models/dspark/config.py (config fields and defaults);
* vllm-project/vllm @ 5f30fc7031cae49bf51073fc953d419b08f8887c:
  vllm/model_executor/models/qwen3_dspark.py (DSparkMarkovHead.apply_bias_gathered, DSparkConfidenceHead in fp32),
  vllm/v1/worker/gpu/spec_decode/dspark/speculator.py (_sample_sequential_topk: Markov bias over base top-k
  candidates, left to right from the anchor).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from safetensors import safe_open

from tensorfold.engine.exact_sampling import Sampling, uniform_rows

from tensorfold.families.glm5_next.cuda import glue, qmm
from tensorfold.families.glm5_next.cuda.dflash2 import (NO_LIMIT, NOISE, Drafter as _DFlash2, _quantize4,
                                                        best_depth, merge_candidates)

TOP_K = 64                 # merged base candidates a slot the Markov bias is applied over
QUANTS = ("q4", "bf16")


# -- config -------------------------------------------------------------------------------------------------------
@dataclass
class DSparkConfig:
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    inter: int
    layers: int
    eps: float
    theta: float
    window: int                    # context rows a slot sees (sliding_window); 0: unbounded
    block: int
    mask_id: int
    aux_ids: tuple[int, ...]       # the checkpoint's aux_hidden_state_layer_ids (inputs of those layers)
    vocab: int                     # draft vocabulary (== the verifier's: no d2t)
    markov_rank: int
    markov_type: str
    confidence: bool
    confidence_markov: bool
    causal: bool
    sample_from_anchor: bool
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def tap_layers(self) -> tuple[int, ...]:
        """The engine's tap points: the residual after layer id - 1 is the input of layer id."""

        return tuple(i - 1 for i in self.aux_ids)

    @classmethod
    def read(cls, draft_dir: str | Path) -> "DSparkConfig":
        return cls.from_dict(json.loads((Path(draft_dir) / "config.json").read_text()), draft_dir)

    @classmethod
    def from_dict(cls, cfg: dict, what: str | Path = "the config") -> "DSparkConfig":
        if cfg.get("speculators_model_type", "dspark") != "dspark":
            raise ValueError(f"{what}: not a DSpark speculator ({cfg.get('speculators_model_type')})")
        t = cfg["transformer_layer_config"]
        heads = int(t["num_attention_heads"])
        hd = int(t.get("head_dim") or int(t["hidden_size"]) // heads)
        types = t.get("layer_types") or []
        sliding = bool(t.get("use_sliding_window", True)) and t.get("sliding_window") is not None
        if types and any(x != "sliding_attention" for x in types):
            if any(x == "sliding_attention" for x in types):
                raise ValueError("mixed sliding / full draft layers are not ported")
            sliding = False
        mtype = str(cfg.get("markov_head_type", "vanilla"))
        if int(cfg.get("markov_rank", 0)) > 0 and mtype != "vanilla":
            raise ValueError(f"only the vanilla Markov head is ported, not {mtype!r}")
        if not cfg.get("sample_from_anchor", True):
            raise ValueError("only sample_from_anchor=True (anchor-as-first-prediction) is ported")
        theta = (t.get("rope_parameters") or {}).get("rope_theta", t.get("rope_theta", 10000.0))
        return cls(hidden=int(t["hidden_size"]), heads=heads, kv_heads=int(t["num_key_value_heads"]), head_dim=hd,
                   inter=int(t["intermediate_size"]), layers=int(t["num_hidden_layers"]),
                   eps=float(t["rms_norm_eps"]), theta=float(theta),
                   window=int(t["sliding_window"]) if sliding else 0, block=int(cfg["block_size"]),
                   mask_id=int(cfg["mask_token_id"]), aux_ids=tuple(int(i) for i in cfg["aux_hidden_state_layer_ids"]),
                   vocab=int(cfg.get("draft_vocab_size") or t["vocab_size"]), markov_rank=int(cfg.get("markov_rank", 0)),
                   markov_type=mtype, confidence=bool(cfg.get("enable_confidence_head", False)),
                   confidence_markov=bool(cfg.get("confidence_head_with_markov", False)),
                   causal=not bool(cfg.get("sliding_window_non_causal", False)) and sliding,
                   sample_from_anchor=True, raw=cfg)


def expected_tensors(c: DSparkConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor a DSpark checkpoint holds, with its shape (no embed_tokens / lm_head: the verifier's)."""

    D, hd = c.hidden, c.head_dim
    out = {"fc.weight": (D, len(c.aux_ids) * D), "hidden_norm.weight": (D,), "norm.weight": (D,)}
    for i in range(c.layers):
        p = f"layers.{i}."
        out.update({p + "input_layernorm.weight": (D,), p + "post_attention_layernorm.weight": (D,),
                    p + "self_attn.q_proj.weight": (c.heads * hd, D), p + "self_attn.k_proj.weight": (c.kv_heads * hd, D),
                    p + "self_attn.v_proj.weight": (c.kv_heads * hd, D), p + "self_attn.o_proj.weight": (D, c.heads * hd),
                    p + "self_attn.q_norm.weight": (hd,), p + "self_attn.k_norm.weight": (hd,),
                    p + "mlp.gate_proj.weight": (c.inter, D), p + "mlp.up_proj.weight": (c.inter, D),
                    p + "mlp.down_proj.weight": (D, c.inter)})
    if c.markov_rank > 0:
        out["markov_head.markov_w1.weight"] = (c.vocab, c.markov_rank)
        out["markov_head.markov_w2.weight"] = (c.vocab, c.markov_rank)
    if c.confidence:
        out["confidence_head.proj.weight"] = (1, D + (c.markov_rank if c.confidence_markov else 0))
        out["confidence_head.proj.bias"] = (1,)
    return out


def draft_parts(heads: int, kv_heads: int, inter: int, world: int) -> tuple[list[int], list[int], list[int]]:
    """DFlash2's split (Mia's ``tp.draft_parts``): KV heads in whole groups (each with its heads // kv_heads query
    heads), the MLP in units of 128 -> (query heads, KV heads, MLP width) per rank. This tree's ``share`` /
    ``share_tile``: the first n % world ranks get one extra unit, so two ranks hold exact halves and four (the DSpark
    checkpoint's 64 / 64 / 12288) hold exact quarters."""

    from tensorfold.cuda.geometry import share, share_tile

    if heads % kv_heads:
        raise ValueError(f"drafter: {heads} query heads over {kv_heads} KV heads")
    group = heads // kv_heads
    kv = [share(kv_heads, world, r) for r in range(world)]
    return [k * group for k in kv], kv, [share_tile(inter, world, r, 128 if inter % 128 == 0 else 1)
                                         for r in range(world)]


def shares(c: DSparkConfig, rank: int, world: int) -> dict[str, tuple[int, int]]:
    """This rank's (start, size) of query heads, KV heads and MLP width (``draft_parts``)."""

    from tensorfold.cuda.geometry import share_lo

    qs, kvs, mlps = draft_parts(c.heads, c.kv_heads, c.inter, world)
    unit = 128 if c.inter % 128 == 0 else 1
    return {"q": (share_lo(c.heads, world, rank), qs[rank]),
            "kv": (share_lo(c.kv_heads, world, rank), kvs[rank]),
            "mlp": (unit * share_lo(c.inter // unit, world, rank), mlps[rank])}


def resume_rows() -> int:
    """TF_GLM_DSPARK_RESUME_ROWS (default 4,096): spare context-ring rows past the window, so a kept prompt's resume
    (TF_GLM_KEEP_SLOTS) rolls the drafter back instead of resetting it — exact while the rows written past the resume
    point (the last reply and the chunk-boundary slack) fit in the spare (``Drafter.resume``). About 24 KB a row a rank
    for GLM-5.3's drafter (3 layers, 16 KV heads a rank, head 128): 4,096 rows ~ 100 MB a rank."""

    return max(0, int(os.environ.get("TF_GLM_DSPARK_RESUME_ROWS", "") or 4096))


def ring_rows(c: DSparkConfig) -> int:
    """Rows of the context ring: the window, the resume spare, the block and a tile (``geometry.draft_ring_rows``)."""

    from tensorfold.cuda.geometry import draft_ring_rows

    return draft_ring_rows(c.window - 1 + resume_rows(), c.block) if c.window > 0 else 0


def memory(c: DSparkConfig, rank: int, world: int, *, quant: str = "q4", capacity: int = 2560,
           ring: bool = True) -> dict[str, int]:
    """Bytes the drafter holds on rank ``rank``: device weights, its K/V context and fixed buffers, and host (the
    replicated Markov tables). The verifier's embedding and head are shared, not counted."""

    s = shares(c, rank, world)
    D, hd = c.hidden, c.head_dim
    q, kv, m = s["q"][1] * hd, s["kv"][1] * hd, s["mlp"][1]

    def mat(n: int, k: int) -> int:
        return -(-n // 128) * 128 * k * 9 // 16 if quant == "q4" else n * k * 2

    per_layer = mat(q + 2 * kv, D) + mat(D, q) + mat(2 * m, D) + mat(D, m)
    if quant == "q4":
        per_layer += mat(2 * kv, D)                     # the context's [k | v] copy (bf16 shares the qkv rows)
    weights = mat(D, len(c.aux_ids) * D) + c.layers * per_layer + (3 + 2 * c.layers) * D * 2 + c.layers * 4 * hd
    rows = (ring_rows(c) if ring and c.window > 0 else 0) or capacity + c.block
    kvcache = 2 * c.layers * s["kv"][1] * rows * hd * 2
    fixed = 64 * len(c.aux_ids) * D * 2 + 16 * c.block * (D + m) * 4
    host = 2 * c.vocab * c.markov_rank * 2
    return {"weights": weights, "kv": kvcache, "fixed": fixed, "device": weights + kvcache + fixed, "host": host}


# TF_GLM_DSPARK_NOISE: the weight of the target's keyed noise in a sampled draft (DFlash2's 0.7 by default; 1.0 when
# the drafter's scores are the target's); TF_GLM_DSPARK_FILTER=1 (default): a sampled draft only among the candidates
# the target's own rule could pick (``exact_sampling.choose_rows``: top_k, then top_p, then min_p, over the drafter's
# scores): a draft from the tail is a draft the target never takes. Drafts only propose: replies are unchanged.
# Both change proposals, so the engine's startup comparison must carry them (every rank the same or none serves).
DSPARK_NOISE = float(__import__("os").environ.get("TF_GLM_DSPARK_NOISE", "") or NOISE)
DSPARK_FILTER = os.environ.get("TF_GLM_DSPARK_FILTER", "1") != "0"


def _bits(w: torch.Tensor) -> np.ndarray:
    """A host bf16 tensor's raw 16-bit words as a NumPy view (no copy)."""

    assert w.device.type == "cpu" and w.dtype == torch.bfloat16
    return w.view(torch.uint16).numpy()


def _bf16_rows(bits: np.ndarray, idx) -> np.ndarray:
    """Rows ``idx`` of a host bf16 matrix as float64 (bf16 -> f32 -> f64, the exact widening)."""

    u16 = np.take(bits, np.asarray(idx, dtype=np.int64), axis=0).astype(np.uint32) << 16
    return u16.view(np.float32).astype(np.float64)


def target_kept(score: np.ndarray, ids: np.ndarray, s) -> np.ndarray:
    """Which of a slot's candidates the target's sampling rule keeps, judged on the drafter's scores (already over the
    temperature): ``choose_rows``' order (score descending, id ascending), top_k, the top_p prefix, min_p."""

    n = score.shape[0]
    order = np.lexsort((ids, -score))
    k = max(1, min(int(s.top_k) if s.top_k else n, n))
    order = order[:k]
    sc = score[order].astype(np.float64)
    if 0.0 < s.top_p < 1.0:
        probs = np.exp(sc - sc.max())
        probs /= probs.sum()
        keep = int((np.cumsum(probs) < s.top_p).sum()) + 1
        order, sc = order[:keep], sc[:keep]
    if s.min_p > 0.0:
        order = order[sc >= sc[0] + s.min_log]
    kept = np.zeros(n, dtype=bool)
    kept[order] = True
    return kept


@triton.jit
def _block_attn_kernel(Q, K, V, OUT, POS, W, scale, N: tl.constexpr, MP: tl.constexpr, NH: tl.constexpr,
                       HD: tl.constexpr, CAP: tl.constexpr, BK: tl.constexpr, CAUSAL: tl.constexpr,
                       RING: tl.constexpr):
    """One head (MHA): the block's N queries (padded to MP rows) attend the context rows [s - W, s) and the block's
    keys (causal: up to their own slot); the window's tiles from (s - W) rounded down to BK, in order."""

    h = tl.program_id(0)
    m = tl.arange(0, MP)
    valid = m < N
    d = tl.arange(0, HD)
    q = tl.load(Q + (h * N + m[:, None]) * HD + d[None, :], mask=valid[:, None], other=0.0)
    s = tl.load(POS).to(tl.int32)
    klen = s + N
    qpos = s + m
    lo = tl.maximum(s - W, 0) // BK * BK
    m_i = tl.full([MP], -1e30, tl.float32)
    l_i = tl.zeros([MP], tl.float32)
    acc = tl.zeros([MP, HD], tl.float32)
    for start in range(lo, klen, BK):
        kk = start + tl.arange(0, BK)
        kin = kk < klen
        if RING:
            row = kk % CAP
        else:
            row = kk
        k = tl.load(K + (h * CAP + row[:, None]) * HD + d[None, :], mask=kin[:, None], other=0.0)
        v = tl.load(V + (h * CAP + row[:, None]) * HD + d[None, :], mask=kin[:, None], other=0.0)
        sc = tl.dot(q, tl.trans(k)) * scale
        ctx = (kk < s) & (kk >= s - W)
        blk = kk[None, :] >= s
        if CAUSAL:
            blk = blk & (kk[None, :] <= qpos[:, None])
        ok = kin[None, :] & (ctx[None, :] | blk)
        sc = tl.where(ok, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(OUT + m[:, None] * (NH * HD) + h * HD + d[None, :], out.to(tl.bfloat16), mask=valid[:, None])


# -- the drafter --------------------------------------------------------------------------------------------------
@dataclass
class DSparkLayer:
    in_norm: torch.Tensor
    post_norm: torch.Tensor
    qkv: qmm.Q4 | qmm.B16
    kv: qmm.Q4 | qmm.B16
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    o: qmm.Q4 | qmm.B16
    gu: qmm.Q4 | qmm.B16
    down: qmm.Q4 | qmm.B16


def draft_quant(value: str | None = None) -> str:
    """The drafter's weight copies: "q4" (4-bit, the default) or "bf16" (TF_GLM_DSPARK_QUANT)."""

    kind = (os.environ.get("TF_GLM_DSPARK_QUANT", "") if value is None else value) or "q4"
    if kind not in QUANTS:
        raise ValueError(f"TF_GLM_DSPARK_QUANT is {' or '.join(QUANTS)}, not {kind!r}")
    return kind


class Drafter(_DFlash2):
    """DSpark on one sequence, the engine-facing API of ``dflash2.Drafter``: ``tap_layers``, ``add_taps``,
    ``propose(pending, depth, sampling, confidence, *, beta, round_ms)``, ``context_end``, ``pos_dev``, ``reset``,
    ``capture``, ``window`` / ``ring`` / ``kc`` / ``vc`` (snapshots), ``nbytes``. ``w``: the verifier's weights on this
    rank (``embed``, ``head`` / ``draft_head`` with ``vocab_offset``, ``rank``, ``world``, ``comm``, ``device``)."""

    def __init__(self, draft_dir: str | Path, w, *, block: int | None = None, capacity: int = 2560,
                 tap_rows: int = 0, ring: bool = False, top_k: int = TOP_K, quant: str | None = None) -> None:
        path = Path(draft_dir)
        c = self.config = DSparkConfig.read(path)
        self.w = w
        self.dev = torch.device(w.device)
        self.rank, self.world = w.rank, w.world
        self.D, self.hd = c.hidden, c.head_dim
        if c.heads != c.kv_heads:
            raise ValueError("the block attention kernel is multi-head (heads == kv_heads)")
        sh = shares(c, self.rank, self.world)
        q0, self.heads = sh["q"]
        kv0, self.kvh = sh["kv"]
        m0, self.inter = sh["mlp"]
        self.eps = c.eps
        self.mask_id = c.mask_id
        self.block = int(block or c.block)
        if self.block > c.block:
            raise ValueError(f"block {self.block} past the trained {c.block}")
        self.top_k = int(top_k)
        self.tap_layers = c.tap_layers
        self.span = c.window if c.window > 0 else 1 << 30    # context rows a slot sees
        self.window = self.span - 1                          # DFlash2's meaning: kept rows = window + 1
        self.causal = c.causal
        self.quant = draft_quant(quant)      # "q4" (minmax ranges; mse is not ported) or "bf16"
        H, KV, hd, D, dev = self.heads, self.kvh, self.hd, self.D, self.dev

        def dense(t: torch.Tensor):
            t = t.to(dev, torch.bfloat16).contiguous()
            return _quantize4(t) if self.quant == "q4" else qmm.make_b16(t)

        def vec(t: torch.Tensor) -> torch.Tensor:
            return t.to(dev, torch.bfloat16).contiguous()

        with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as f:
            def rows(name: str, a: int, b: int) -> torch.Tensor:
                return f.get_slice(name)[a:b]

            def cols(name: str, a: int, b: int) -> torch.Tensor:
                return f.get_slice(name)[:, a:b]

            self.fc = dense(f.get_tensor("fc.weight"))
            self.hidden_norm = vec(f.get_tensor("hidden_norm.weight"))
            self.norm = vec(f.get_tensor("norm.weight"))
            self.layers: list[DSparkLayer] = []
            for i in range(c.layers):
                p = f"layers.{i}."
                q = rows(p + "self_attn.q_proj.weight", q0 * hd, (q0 + H) * hd)
                k = rows(p + "self_attn.k_proj.weight", kv0 * hd, (kv0 + KV) * hd)
                v = rows(p + "self_attn.v_proj.weight", kv0 * hd, (kv0 + KV) * hd)
                qkv = dense(torch.cat((q, k, v)))
                if isinstance(qkv, qmm.B16):
                    kvw = qmm.B16(qkv.weight[H * hd:], 2 * KV * hd, D)       # contiguous rows: no copy
                else:
                    kvw = dense(torch.cat((k, v)))
                g = rows(p + "mlp.gate_proj.weight", m0, m0 + self.inter)
                u = rows(p + "mlp.up_proj.weight", m0, m0 + self.inter)
                self.layers.append(DSparkLayer(
                    in_norm=vec(f.get_tensor(p + "input_layernorm.weight")),
                    post_norm=vec(f.get_tensor(p + "post_attention_layernorm.weight")),
                    qkv=qkv, kv=kvw,
                    q_norm=vec(f.get_tensor(p + "self_attn.q_norm.weight")),
                    k_norm=vec(f.get_tensor(p + "self_attn.k_norm.weight")),
                    o=dense(cols(p + "self_attn.o_proj.weight", q0 * hd, (q0 + H) * hd)),
                    gu=dense(torch.cat((g, u))),
                    down=dense(cols(p + "mlp.down_proj.weight", m0, m0 + self.inter))))
                del q, k, v, g, u
            # the sequential heads, replicated on the host (rows are gathered per candidate: bf16 kept, 2 x V x r x 2 B)
            self.w1 = self.w2 = None
            if c.markov_rank > 0:
                self.w1 = f.get_tensor("markov_head.markov_w1.weight").to(torch.bfloat16).contiguous()
                self.w2 = f.get_tensor("markov_head.markov_w2.weight").to(torch.bfloat16).contiguous()
            self.conf_h = None           # device fp32 [D]: the confidence head's hidden part
            self.conf_m = None           # host float64 [r]: its Markov part
            self.conf_b = 0.0
            if c.confidence:
                cw = f.get_tensor("confidence_head.proj.weight").float()[0]
                self.conf_h = cw[:D].to(dev).contiguous()
                if c.confidence_markov:
                    self.conf_m = cw[D:].double().numpy().copy()
                self.conf_b = float(f.get_tensor("confidence_head.proj.bias").float()[0])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.inv_freq = 1.0 / c.theta ** (torch.arange(hd // 2, device=dev, dtype=torch.float32) * 2 / hd)
        self.capacity = capacity
        rows_ = ring_rows(c) if ring else 0
        self.ring = rows_ if 0 < rows_ < capacity + self.block else 0
        self.cap = self.ring or capacity + self.block
        self.kc = [torch.zeros((KV, self.cap, hd), dtype=torch.bfloat16, device=dev) for _ in self.layers]
        self.vc = [torch.zeros((KV, self.cap, hd), dtype=torch.bfloat16, device=dev) for _ in self.layers]
        self.pos_dev = torch.zeros((1,), dtype=torch.int64, device=dev)
        self.context_end = 0
        n = self.block
        self.ids = torch.full((n,), self.mask_id, dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((1,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.ar = torch.arange(max(64, n), device=dev)
        self.tap_in = torch.zeros((64, len(self.tap_layers) * D), dtype=torch.bfloat16, device=dev)
        # a pass's candidates (every rank's [block, 2 * top_k], values then ids as fp32 bits) and each slot's
        # confidence logit from h alone [block] fp32, one device buffer read back by one pinned copy
        self.gathered = self.world if self.world > 1 and w.comm is not None else 1
        self.cand_n = self.gathered * n * 2 * self.top_k
        cuda = torch.cuda.is_available()
        self.cand = torch.zeros((self.cand_n + n,), dtype=torch.float32, device=dev)
        self.cand_host = torch.zeros(self.cand.shape, dtype=torch.float32, pin_memory=cuda)
        self.cand_ready = torch.cuda.Event() if cuda else None
        self.packed = self.cand[:self.cand_n].view(self.gathered, n, 2 * self.top_k)
        self.hconf = self.cand[self.cand_n:]
        self.pool = None
        self.block_graph = None
        self.tap_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.tap_rows = min(max(self.block, int(tap_rows)), self.tap_in.shape[0])
        self.last_confidence: list[float] = []      # the last chain's per-draft confidences (walked part)
        self.debug = False                           # eager passes keep h and this rank's logits (tests)
        self.last_h = self.last_logits = None
        # this family's verifier: an EXL3 head called through a scratch the drafter owns (never the engine's:
        # capture replays this one beside its own), and an embedding split by vocabulary rows (forward.embed's
        # gather over every rank's span; the stub carries only what embed reads: .gath)
        from . import forward as dsa_forward
        from .x3 import X3, X3Scratch

        self._embed = dsa_forward.embed
        head = getattr(w, "draft_head", None) or w.head
        self._head_is_x3 = isinstance(head, X3)
        if self._head_is_x3:
            self.x3 = X3Scratch([head], dev, prefill=False)
            self.logits = torch.empty((n, head.n), dtype=torch.float32, device=dev)
        else:
            self.x3, self.logits = None, None
        self._stub = type("_B", (), {})()            # forward.embed's b: only .gath is read
        self._stub.gath = (torch.empty((self.world * n * D // 2,), dtype=torch.float32, device=dev)
                           if w.embed.shape[0] != w.cfg.vocab else None)

    def nbytes(self) -> int:
        def size(q) -> int:
            return q.nbytes() if hasattr(q, "nbytes") else q.numel() * q.element_size()

        total = size(self.fc)
        for L in self.layers:
            total += sum(size(q) for q in (L.qkv, L.o, L.gu, L.down))
            if self.quant == "q4":
                total += size(L.kv)
        return total + 2 * sum(t.numel() * 2 for t in self.kc)

    # -- the block pass -------------------------------------------------------------------------------------------
    def _attend(self, i: int, q: torch.Tensor) -> torch.Tensor:
        rows = q.shape[1]
        out = torch.empty((rows, self.heads * self.hd), dtype=torch.bfloat16, device=self.dev)
        _block_attn_kernel[(self.heads,)](q, self.kc[i], self.vc[i], out, self.pos_dev, self.span, self.hd ** -0.5,
                                          N=rows, MP=max(16, triton.next_power_of_2(rows)), NH=self.heads,
                                          HD=self.hd, CAP=self.cap, BK=64, CAUSAL=self.causal,
                                          RING=bool(self.ring), num_warps=4)
        return out

    def _layer(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        L = self.layers[i]
        rows = x.shape[0]
        normed, xs = self._norm(x, L.in_norm)
        q, k, v = self._prep(qmm.matmul(normed, L.qkv, xs), L, cos, sin, self.heads)
        self.kc[i].index_copy_(1, idx, k)
        self.vc[i].index_copy_(1, idx, v)
        x = x + self._row(self._attend(i, q), L.o)
        normed, xs = self._norm(x, L.post_norm)
        gu = qmm.matmul(normed, L.gu, xs)
        act = torch.empty((rows, self.inter), dtype=torch.bfloat16, device=self.dev)
        axs = torch.empty((rows, self.inter // 64), dtype=torch.float32, device=self.dev)
        glue.swiglu(gu, act, axs, NO_LIMIT)
        return x + self._row(act, L.down, axs)

    def _block_compute(self) -> None:
        """[pending, mask x (block - 1)] at the committed length -> each rank's top-K of its vocabulary slice for
        every slot (gathered) and each slot's hidden-part confidence logit."""

        n = self.block
        x = torch.empty((n, self.D), dtype=torch.bfloat16, device=self.dev)
        if self.w.embed.shape[0] != self.w.cfg.vocab:
            # this family's split embedding: each rank writes its span's rows, every rank gathers all
            # ranks' rows and keeps each token's from its holder (forward.embed: exactly the bf16 rows)
            self._embed(self.w, self._stub, self.ids, x)
        else:
            glue.embed(self.ids, self.w.embed, self.D, 1, x)
        cos, sin = self._rotary(n)
        idx = self._slots(n)
        for i in range(len(self.layers)):
            x = self._layer(i, x, cos, sin, idx)
        h, hs = self._norm(x, self.norm)
        head = self.w.draft_head if getattr(self.w, "draft_head", None) is not None else self.w.head
        if self._head_is_x3:
            head(h, self.logits, self.x3)                # fp32 rows; the -inf past the rank's crop topk ignores
            logits = self.logits
        else:
            logits = qmm.matmul(h, head, hs, f32=True)
        if self.debug:
            self.last_h, self.last_logits = h.clone(), logits.clone()
        vals, local = torch.topk(logits, self.top_k, dim=-1)
        gids = (local + self.w.vocab_offset).to(torch.int32)
        packed = torch.cat([vals, gids.view(torch.float32)], dim=1).contiguous()
        if self.gathered > 1:
            self.w.comm.all_gather(packed.view(-1), self.cand[:self.cand_n])
        else:
            self.cand[:self.cand_n].copy_(packed.view(-1))
        if self.conf_h is not None:
            self.hconf.copy_(h.float() @ self.conf_h)
        else:
            self.hconf.zero_()

    @torch.no_grad()
    def candidates(self, pending: int, depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merged base candidate ids [depth, top_k], their base logits, each slot's hidden-part confidence logit."""

        self.ids_host[0] = pending
        self.ids[:1].copy_(self.ids_host, non_blocking=True)
        if self.block_graph is not None:
            self.block_graph.replay()
        else:
            self._block_compute()
        if self.cand_ready is not None:
            self.cand_host.copy_(self.cand, non_blocking=True)
            self.cand_ready.record()
            self.cand_ready.synchronize()
        else:
            self.cand_host.copy_(self.cand)
        g = self.cand_host[:self.cand_n].view(self.gathered, self.block, 2 * self.top_k)[:, :depth]
        return merge_candidates(g, self.cand_host[self.cand_n:self.cand_n + depth], self.top_k)

    # -- the sequential stage ---------------------------------------------------------------------------------------
    def markov_embed(self, token: int) -> np.ndarray | None:
        return _bf16_rows(_bits(self.w1), int(token)) if self.w1 is not None else None

    def markov_bias(self, cand: np.ndarray, emb: np.ndarray | None) -> np.ndarray:
        """W2[cand] @ W1[prev] in float64 (the bias of the candidates only)."""

        if emb is None:
            return np.zeros(cand.shape, dtype=np.float64)
        return _bf16_rows(_bits(self.w2), np.asarray(cand, dtype=np.int64)) @ emb

    def chain(self, tokens: np.ndarray, values: np.ndarray, hconf: np.ndarray, anchor: int, first: int,
              sampling: Sampling | None, confidence: float = 0.0, *, round_ms=None,
              confs: list | None = None) -> list[int]:
        """Left to right from the anchor: slot d's score = (base + W2 W1[prev]) / T over its candidates, the pick
        its argmax (with the target's keyed noise, weight ``NOISE``, when sampling); its confidence
        sigmoid(hconf + w_m . W1[prev] + b). Cut at ``best_depth`` with ``round_ms``, else at the ``confidence``
        product threshold (the first draft always kept); ``confs``: walk whole, append each confidence."""

        depth = tokens.shape[0]
        sampled = sampling is not None and sampling.temperature > 0
        temp = float(sampling.temperature) if sampled else 1.0
        noise = None
        if sampled:
            noise = -np.log(-np.log(uniform_rows(sampling.seed, first + np.arange(depth), tokens)))
        out: list[int] = []
        conf: list[float] = []
        prev, alive = int(anchor), 1.0
        for d in range(depth):
            emb = self.markov_embed(prev)
            score = (values[d] + self.markov_bias(tokens[d], emb)) / temp
            pick = score + DSPARK_NOISE * noise[d] if noise is not None else score
            if sampled and DSPARK_FILTER:
                pick = np.where(target_kept(score, tokens[d], sampling), pick, -np.inf)
            j = int(np.argmax(pick))
            if self.conf_h is not None:
                z = float(hconf[d]) + self.conf_b
                if self.conf_m is not None and emb is not None:
                    z += float(self.conf_m @ emb)
                c = float(1.0 / (1.0 + np.exp(-z)))
            else:
                p = np.exp(score - score.max())
                c = float(p[j] / p.sum())
            conf.append(c)
            if confs is not None:
                confs.append(c)
            elif round_ms is None and confidence > 0:
                alive *= c
                if d > 0 and alive < confidence:
                    conf.pop()
                    break
            prev = int(tokens[d, j])
            out.append(prev)
        self.last_confidence = conf
        if round_ms is not None and confs is None:
            out = out[:best_depth(conf, round_ms)]
        return out

    def slot_scorer(self, tokens: np.ndarray, values: np.ndarray, hconf: np.ndarray, anchor: int, first: int,
                    sampling: Sampling | None):
        """``copy_drafts.Slot`` over a pass's candidates: ``chain``'s per-slot arithmetic (Markov bias after any of
        the previous slot's candidates, keyed noise, the target filter, the confidence), memoized."""

        sampled = sampling is not None and sampling.temperature > 0
        temp = float(sampling.temperature) if sampled else 1.0
        noise = None
        if sampled:
            noise = -np.log(-np.log(uniform_rows(sampling.seed, first + np.arange(tokens.shape[0]), tokens)))
        memo: dict[tuple[int, int], tuple[np.ndarray, float]] = {}

        def slot(d: int, prev: int) -> tuple[np.ndarray, float]:
            key = (d, prev if d else -1)
            if key in memo:
                return memo[key]
            emb = self.markov_embed(anchor if d == 0 else int(tokens[d - 1, prev]))
            score = (values[d] + self.markov_bias(tokens[d], emb)) / temp
            pick = score + DSPARK_NOISE * noise[d] if noise is not None else score
            if sampled and DSPARK_FILTER:
                pick = np.where(target_kept(score, tokens[d], sampling), pick, -np.inf)
            if self.conf_h is not None:
                z = float(hconf[d]) + self.conf_b
                if self.conf_m is not None and emb is not None:
                    z += float(self.conf_m @ emb)
                c = float(1.0 / (1.0 + np.exp(-z)))
            else:
                p = np.exp(score - score.max())
                c = float(p[int(np.argmax(pick))] / p.sum())
            memo[key] = (pick, c)
            return memo[key]

        return slot

    def propose_hybrid(self, pending: int, conts, depth: int, sampling: Sampling | None, confidence: float = 0.0,
                       *, rank: int = 1, most: int = 15, beta: float = 0.0,
                       round_ms: tuple[float, ...] | None = None) -> tuple[list[int], str, int]:
        """``copy_drafts.hybrid`` over this pass: every block slot vets the copy continuations ``conts`` (latest
        first); DSpark's own chain (``depth`` drafts at most, ``propose``'s rule) where none agrees. (drafts, kind
        "d" / "h" / "x", the continuation used or -1)."""

        from .copy_drafts import hybrid

        if self.context_end == 0:
            return [], "d", -1
        tokens, values, hconf = self.candidates(pending, self.block)
        slot = self.slot_scorer(tokens, values, hconf, pending, self.context_end + 1, sampling)
        drafts, kind, k, _ = hybrid(tokens, slot, conts, chain_most=min(depth, self.block), confidence=confidence,
                                    round_ms=round_ms, rank=rank, most=most)
        return drafts, kind, k

    def propose(self, pending: int, depth: int, sampling: Sampling | None, confidence: float = 0.0, *,
                beta: float = 0.0, round_ms: tuple[float, ...] | None = None) -> list[int]:
        """Up to ``depth`` (<= block) drafts for the positions after the pending token (at context_end). ``round_ms``
        (ms of a round of k drafts, k = 0..): cut at ``best_depth`` of the learned confidences; else stop below the
        ``confidence`` product. ``beta`` is accepted for DFlash2's call shape and unused (the head is learned)."""

        depth = min(depth, self.block)
        if depth < 1 or self.context_end == 0:
            return []
        tokens, values, hconf = self.candidates(pending, depth)
        return self.chain(tokens, values, hconf, pending, self.context_end + 1, sampling, confidence,
                          round_ms=round_ms)

    def resume(self, cut: int) -> bool:
        """Roll the context back to position ``cut`` for a kept prompt's resume (TF_GLM_KEEP_SLOTS; decode.prefill):
        the rows before ``cut`` were fed from the same tokens, so the drafter keeps them. Exact while the ring still
        holds the window before ``cut``: every row written past it (the last reply, the chunk slack) must not have
        wrapped onto a slot that window reads (end - cut <= ring - span - block). Otherwise — a cut past the context,
        or rows past it that wrapped — the position still moves to ``cut`` (the target resumes there, and every
        later row must land at its own position), over a context that is stale before it: drafts start shallow
        until the window refills (replies are unchanged: drafts only propose). True when the context was kept.
        (2026-10-16: resetting to 0 here left the drafter ``cut`` rows behind the target after a long prompt was
        followed by an unrelated short one; dflash_decode refused the request.)"""

        end = self.context_end
        kept = 0 < cut <= end and (not self.ring or end - cut <= self.ring - self.span - self.block)
        self.context_end = max(0, int(cut))
        self.pos_dev.fill_(self.context_end)
        return kept


# -- the engine's wiring -------------------------------------------------------------------------------------------
def is_dspark(draft_dir: str | Path) -> bool:
    """Whether ``draft_dir`` holds a DSpark speculator (``speculators_model_type``), for the engine's --drafter path."""

    try:
        cfg = json.loads((Path(draft_dir) / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return cfg.get("speculators_model_type", "dspark") == "dspark"


def dspark_weights(draft_dir, world: int) -> "Weights":
    """The DSpark drafter as held on each rank (its q4 matrices, bf16 norms, K/V context and fixed buffers, and the
    host Markov tables), and its staging: the widest rank's shares (rank 0's: the n % world ranks hold one more
    unit), the peak one matrix's bf16 slice reaches while the loader quantizes it."""

    from tensorfold.cuda.capacity import Weights

    c = DSparkConfig.read(draft_dir)
    m = memory(c, 0, world, capacity=0, ring=True)
    s = shares(c, 0, world)
    D, hd = c.hidden, c.head_dim
    q, kv, ml = s["q"][1] * hd, s["kv"][1] * hd, s["mlp"][1]
    staging = max((q + 2 * kv) * D, D * q, 2 * ml * D, D * ml, D * len(c.aux_ids) * D) * 2
    return Weights(m["device"] + m["host"], max(staging, m["device"]), 0)


def dspark_geometry(t: dict, world: int, reserve: int, *, ring: bool = True) -> "Geometry":
    """The DSpark drafter's per-slot bytes beside the target's: its K/V context rows (a ring, or capacity + block)
    and the fixed buffers; the weights are counted once by ``dspark_weights``. ``t``: the speculator's config dict."""

    from tensorfold.cuda.geometry import Geometry

    c = DSparkConfig.from_dict(t)
    rows = ring_rows(c) if ring and c.window > 0 else 0

    def bytes_at(capacity: int) -> int:
        s = shares(c, 0, world)
        slots = capacity + c.block if not rows else min(rows, capacity + c.block)
        kv = 2 * c.layers * s["kv"][1] * slots * c.head_dim * 2
        fixed = 64 * len(c.aux_ids) * c.hidden * 2 + 16 * c.block * (c.hidden + s["mlp"][1]) * 4
        return kv + fixed

    return Geometry(bytes_at, 0, 0)
