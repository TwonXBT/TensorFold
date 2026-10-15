# DSpark drafter → our engine (glm_moe_dsa): port design

Design only — no code changed. Target: `~/glm53-tf/TensorFold` (deployed build), worktree `tf-dspark`
(branch `claude/dspark` from 52b7cd8). Drafter checkpoint: `~/models/GLM-5.3-speculator.dspark`
(config: `aux_hidden_state_layer_ids` [2,20,39,58,75], `block_size` 8, `markov_rank` 256,
`draft_vocab_size` 154880, confidence head with markov, 3 Qwen3-style layers, hidden 6144, heads 64=kv,
head_dim 64, inter 12288, sliding_window 2048, mask_token_id 154856, `sample_from_anchor` true).

Reference: Mia-AiLab `~/claude-workspace/mia-ref/glm5_next/cuda/` (Apache-2.0): `dspark.py` (643 lines,
subclasses her `dflash2.Drafter`), `dflash2.py`, `decode.py` (`dflash_decode`, `hybrid_round`),
`engine.py`, `forward.py` (`set_taps` / `tap_at` / `stream_mean`).

Ours: `src/tensorfold/families/glm5_next/cuda/dflash2.py` (481 lines, an OLDER base Drafter),
`src/tensorfold/families/glm_moe_dsa/cuda/{forward,decode,engine,mtp,copy_drafts}.py`.

Two structural facts drive the whole design:

1. **Our verifier head is EXL3, not bf16.** `w.head` is an `X3` (trellis) object called as
   `head(x, out, scratch)` (`glm_moe_dsa/cuda/x3.py:97`), not `qmm.matmul`-compatible. Mia's full-GLM
   engine has the same problem and dspark.py already carries the fix pattern (`_block_compute`,
   dspark.py:460-505): the drafter must own a small `X3Scratch` and call the head through it, and the
   embedding is vocab-row-split per rank (`w.embed_lo`, all-gather of fp32 parts) — our
   `forward.embed` (`glm_moe_dsa/cuda/forward.py:746`) is exactly that gather and can be reused
   verbatim by passing the drafter a stub `b` with `.gath` and `.ids`.
2. **Our family deleted the inherited taps** (`glm_moe_dsa/cuda/forward.py:256`,
   `del self.taps, self.tap_at`) because GLM-5.3 has one residual stream, no hyper-connections —
   Mia's taps are a *mean of streams* (`glue.stream_mean`), ours would be a plain copy of `b.x`.
   The parent class `glm5_next/cuda/forward.py` still defines `set_taps` (:134) and the tap fill
   (:478-479), so restoring is additive, not a rewrite.

---

## 1. The API `dspark.Drafter` needs from its base class

What Mia's `dspark.Drafter` (subclass) uses from her `dflash2.Drafter`, and whether OUR
`glm5_next/cuda/dflash2.py` `Drafter` provides it. "Same" = same signature/semantics, usable as-is.

| Base-class member | Used by dspark for | Ours (`glm5_next/cuda/dflash2.py`) | Verdict |
|---|---|---|---|
| `Drafter.__init__` (not called — dspark sets its own attrs) | — | :176 | n/a (dspark constructs everything itself) |
| `reset()` | clear context (`context_end`, `pos_dev`) | :280 | **same** |
| `_slots(n)` | ring slot indices for new rows | :284 | **same** |
| `_rotary(rows)` | cos/sin for positions `context_end..` | :291 | **same** |
| `_prep(qkv, L, cos, sin, heads)` | split+norm+rope QKV via `_prep_kernel` | :296 (+kernel :51) | **same** (kernel identical; dspark passes `L.q_norm/k_norm` — our `DraftLayer` has those fields) |
| `_norm(x, weight)` | RMSNorm + 64-group sums | :306 | **same** |
| `_row(x, q, xs)` | row-parallel partial, all-gather, rank-order sum → bf16 | :313 | **same** (ours gathers into a fresh buffer; Mia's is identical logic — dspark only needs the return value) |
| `_taps_compute(n)` | context rows from `tap_in` → fc → hidden_norm → per-layer K/V into `kc/vc` | :349 | **same** (dspark overrides nothing; it relies on `self.fc`, `self.hidden_norm`, `self.layers[i].kv`, `tap_in`, `kc`, `vc`, `pos_dev`) |
| `add_taps(taps)` | chunked context update, graph replay for n ≤ `tap_rows` | :360 | **different**: ours chunks at `tap_in.shape[0]` (64) and replays graphs only for `n ≤ self.block` (capture loop `1..block`, :406-413). Mia's takes `tap_rows` param, replays up to `tap_rows` (her :418-434, capture `1..tap_rows`). **Needed**: the `tap_rows` constructor arg + capture-range change (small, backward-compatible; default `tap_rows=block` keeps today's behavior). A verify window keeps ≤ MAX_ROWS=8 rows and prompt chunks feed 64-row slices, so 64 covers both even without the change — but Mia's `tap_rows` makes prompt-chunk feeds single-slice. |
| `capture()` | CUDA graphs for block + tap updates | :401 | **different** (same reason: `1..block` vs `1..tap_rows`); otherwise same |
| `nbytes` | memory reporting | :276 | **different**: ours sums DFlash2-specific weights (`a_kp/m_kp/kv…`). dspark **overrides** `nbytes` (dspark.py:424) — no base support needed. |
| `_block_compute` / `candidates` / `chain` / `propose` | the DFlash2 block pass | :377/:428/:448/:474 | **overridden** by dspark (its own block attention, Markov/confidence chain). Only the *call shape* of `candidates` matters: dspark reimplements it. |
| `self.tap_layers` | engine reads it to call `set_taps` | set in `__init__` :251 | dspark sets its own from config — fine |
| `self.window`, `ring`, `kc`, `vc`, `cap`, `capacity`, `pos_dev`, `context_end`, `ids`, `ids_host`, `ar`, `tap_in`, `cand*`, `pool`, `block_graph`, `tap_graphs` | engine/snapshot plumbing | all present (init :236-274) | **same** (dspark allocates its own `cand` buffer with `block` slots instead of `block-1`, and adds `hconf`, `last_confidence`) |

Module-level names dspark imports (`dspark.py:61`): `NO_LIMIT` (:23 ✔), `NOISE` (:22 ✔),
`_quantize4` (:146 ✔ — ours is plain `qmm.quantize4`; Mia's is `qmm.draft_quantize4` with
TF_GLM_DRAFT_QUANT=mse. **Missing in our qmm**: `draft_quantize4`/`DRAFT_QUANTS`/`draft_quant`.
Port them or pin ours to minmax — minmax is the default and what the checkpoint was quantized for),
`best_depth` (**missing** in ours — Mia's dflash2.py:41, ~12 lines, pure numpy; copy it),
`merge_candidates` (**missing** in ours — Mia's dflash2.py:187, ~15 lines; ours inlines the same
logic in `candidates` :434-446; extract/copy).

Other imports: `tensorfold.engine.exact_sampling.Sampling/uniform_rows` (ours ✔
`engine/exact_sampling.py:60`), `glue.rmsnorm/swiglu/embed` (ours ✔ glm5_next glue :75/:271/:46),
`qmm.matmul/make_b16/Q4/B16` (ours ✔ qmm.py:18/142/153), `tp.draft_parts` (**missing** — Mia's
tp.py:203; ours has no tp.py. Equivalent: `share`/`share_lo` in `cuda/geometry.py:100/114` with unit
1 for heads and 128 for MLP. At TP4 everything divides evenly (16/16/16/16 heads, 3072 MLP), so a
6-line local helper suffices; port `draft_parts` for generality), `dsa_full.embed_part` (Mia-only;
**do not port** — reuse our `forward.embed` instead, see §2), `geometry.draft_ring_rows`
(ours ✔ geometry.py:322).

**Weights interface** dspark's docstring requires of `w`: `embed`, `head`/`draft_head` +
`vocab_offset`, `rank`, `world`, `comm`, `device` — our `glm_moe_dsa/cuda/weights.py:176-191` has
all (plus `meta`, `embed_lo`). Two adaptations inside dspark's `_block_compute`:
- head: `qmm.matmul(h, head, hs, f32=True)` → `head(h, logits_f32, self.x3_scratch)` (X3 writes
  fp32 when the out buffer is fp32; `n_pad`/`crop` already pad `-inf` past the rank's vocab slice,
  which `torch.topk` ignores — same merged candidates as a bf16 head would give, modulo EXL3's own
  quantization, which the MTP path already accepts for drafts).
- embed: Mia's `embed_offset` meta branch → our `forward.embed(w, stub_b, self.ids, x)` where
  `stub_b` is any object with `.gath` (a `[world * n * D]` fp32 buffer the drafter allocates) —
  our embed does the same write-zeros/gather/pick dance bit-for-bit.

**Bottom line §1**: our dflash2 `Drafter` is a viable base. Required base changes: `tap_rows`
parameter (init + capture range), extract `merge_candidates`, add `best_depth`, optionally
`draft_quantize4`. Everything else dspark overrides or reuses unchanged.

---

## 2. Taps: capturing the residual stream at layers 2, 20, 39, 58, 75

DSpark wants the target residual stream at the **input** of layers 2,20,39,58,75 — i.e. after
layers 1,19,38,57,74 have fully run (attention + MoE + residual). DFlash2's convention (dspark.py
docstring): `tap_layers = aux_ids - 1 = (1, 19, 38, 57, 74)`.

### Where in OUR forward

Our layer loop is `compute()` (`glm_moe_dsa/cuda/forward.py:766-787`): for each `layer in w.layers`
it calls `layer_forward` (:697-700: `residual(x, x, attn_part(...))` then `residual(x, x, ffn_part(...))`),
then after the loop `glue.rmsnorm(b.x[:R], …)`. `LayerW.index` (`weights.py:152`) is the absolute
layer id. The tap capture is therefore:

```python
for layer in w.layers:
    layer_forward(layer, w, st, b, R, nch, host_pos, sparse_np)
    for slot in b.tap_at.get(layer.index, ()):     # ← the only addition
        b.taps[slot][:R].copy_(b.x[:R])            # plain copy: one stream (no stream_mean)
```

- **Decode / verify windows**: `compute()` runs on `e.buf` (rows = MAX_ROWS = 8). Every row of the
  window gets its taps written; after sampling, `e.tap_rows(keep)` reads `taps[:keep]` and
  `drafter.add_taps(...)` absorbs only the kept rows. Taps are separate buffers, so `commit()`
  (:825-830, only advances `st.pos`) and the next window's overwrite of `b.x` cannot corrupt them.
  The copy is inside `compute`, so the captured CUDA graphs (`Graphs.main`, decode.py:381-421, and
  the sparse-bucket graphs) replay it automatically — no graph changes needed, just capture after
  `set_taps`.
- **Prompt chunks**: `prefill()` (decode.py:83-127) runs `fwd.compute` on `e.pbuf` (rows up to
  8192). Same insertion point; `prefill` then calls `drafter.add_taps(e.tap_rows(R, e.pbuf))` per
  chunk (Mia's `prefill_chunk`, her decode.py:694-718, is the model — ours must add the one line).
  `add_taps` internally slices into ≤64-row pieces (`tap_in` is [64, 5·D]), so any chunk size works.
- **Micro-batch overlap path** (`prompt_layers`, :703-723, chunks ≥ 256 rows on TP4): rows are
  processed as two micro-batches with pipelined collectives; `b.x` rows are final only after both
  halves' `residual` at layer end. The tap copy must go **after the layer's final residual loop**
  (after the `for i … residual(v.x[:r], …)` at :721-722), copying the full `b.x[:R]` — by then every
  row's stream is complete for that layer. Same bits as the non-overlapped path (row-invariant).
- **Long context (sparse)**: same loop, same insertion — works unchanged; the sparse graphs capture it.

### Buffer shapes

`set_taps(layers, hidden)` (parent glm5_next forward.py:134-138) allocates, per tap layer, one
`[rows, hidden]` bf16 buffer (`rows` = the Buffers' row count: 8 for `e.buf`, PROMPT_ROWS=8192 for
`e.pbuf`). For 5 taps: decode 5×8×6144×2 B = 480 KiB; prefill 5×8192×6144×2 B = 480 MiB. That
prefill cost is the price of capturing every prompt row's taps; Mia pays the same (her `alloc_rows`
version, forward.py:185). `e.tap_rows(n, b)` = `torch.cat([t[:n] for t in b.taps], dim=1)` →
`[n, 5·6144]` bf16, exactly `add_taps`' expected input.

### How Mia does it

Her forward has hyper-connections (S streams): after each layer, `sp.glue(..., taps=tuple(
b.tap_at.get(layer.index, ())))` folds the layer's streams and writes `glue.stream_mean(b.x[:R],
b.taps[slot][:R])` — the *mean of the S streams* (her forward.py:1890-1894, 2016-2022). Her
`Buffers.set_taps` (:181-186) is the same dict + list allocation over `alloc_rows`.

### What our Buffers deleted and why restoring is safe

`glm_moe_dsa/cuda/forward.py:256` does `del self.taps, self.tap_at` with the comment "no
hyper-connections: x holds one stream". The deletion is an assertion that nothing in this family
uses taps — `Engine.tap_rows` (decode.py:367-369) even raises "GLM-5.3 has no hyper-connection taps:
a DFlash2 drafter cannot attach". Restoring is safe because:

1. The parent class still fully supports them (`set_taps` + the fill site at glm5_next
   forward.py:478-479 — which our family never reaches since it overrides `compute`).
2. Nothing in glm_moe_dsa reads `b.taps`/`b.tap_at` today (grep: only the `del` and the raising
   `tap_rows`); with `tap_at` empty (`set_taps` never called), the added loop body is a no-op —
   zero behavior change when no drafter is attached.
3. The semantics differ only in the fill: `stream_mean` (S streams) vs plain `copy_` (1 stream).
   We keep the parent's `set_taps` (allocation + `tap_at` mapping) and add our own one-line fill in
   our `compute`/`prompt_layers`, never touching the parent's fill site.
4. Memory: +480 MiB prefill / +480 KiB decode only when a drafter is attached (`set_taps` called
   with non-empty layers); the startup capacity estimate (§3/step 1) must add the prefill term.

`Engine.tap_rows` changes from `raise` to the parent's `torch.cat` implementation (decode.py:367).

---

## 3. Tensor parallelism over 4 ranks

DSpark splits exactly like DFlash2 (`tp.draft_parts`, Mia tp.py:203-212): **KV heads in whole
groups** (query heads follow: `heads // kv_heads` per KV head; here 64/64 → group size 1), **MLP
width in units of 128**. At world=4 everything divides evenly:

| Piece | Total | Per rank (TP4) | Split |
|---|---|---|---|
| query heads | 64 | 16 | `split_sizes(64, 4, 1)` = 16/16/16/16 |
| KV heads | 64 | 16 | 16/16/16/16 |
| q/k/v rows | 16·64 = 1024 each | — | row slice `[q0·hd, (q0+16)·hd)` of each proj |
| o_proj | [6144, 4096] | columns `[:, q0·hd:(q0+16)·hd]` | row-parallel via `_row` |
| gate/up | [12288, 6144] | rows `[m0, m0+3072)` | `split_sizes(12288, 4, 128)` = 3072×4 |
| down_proj | [6144, 12288] | columns `[:, m0:m0+3072)` | row-parallel via `_row` |

(Mia's 3-rank split is 22/21/21 query, 22/21/21 KV, 4096/4096/4096 MLP — the *same rule*, uneven
only because 64 and 12288 don't divide by 3. Nothing in the algorithm depends on evenness: `_row`
sums fp32 partials in rank order, `merge_candidates` merges by value-then-id.)

**Replicated** (every rank holds the whole thing):
- `fc` [6144, 5·6144], `hidden_norm`, final `norm` (3 + 2·layers norm vectors);
- per-layer `q_norm`/`k_norm` [64] (4·hd·layers bytes);
- the verifier's embedding rows the rank holds + the all-gather in `_block_compute` (shared, not
  drafter memory);
- **Markov `w1`/`w2`** [154880, 256] bf16 each and the **confidence head** — on the *host*, not the
  device (dspark.py:381-387, 389-393): the sequential chain gathers single rows
  (`_bf16_rows(_bits(w1), token)`) per candidate step, so host RAM is the right place.

**Memory per rank (TP4, q4 weights, ring on)** — from `dspark.memory` (dspark.py:168-189) with our
`geometry.draft_ring_rows(2047, 8) = 2176` rows:

| Component | Bytes | MiB |
|---|---|---|
| weights (fc + 3 layers q4 + norms) | 265,507,584 | 253.2 |
| K/V context ring (3 layers × 16 kvh × 2176 × 64 × 2 × 2 B) | 26,738,688 | 25.5 |
| fixed buffers (tap_in 64×5·D, cand/proj) | 8,650,752 | 8.3 |
| **device total** | **300,897,024** | **~287** |
| host (w1+w2 bf16) | 158,597,120 | ~151 |

bf16 weights instead of q4 would be ~828 MiB device — q4 (the default `TF_GLM_DSPARK_QUANT=q4`)
is the plan. ~287 MiB/rank device + ~151 MiB/rank host is well inside the GB10's budget; the
startup estimate (`admit(..., draft_weights=..., draft_geometry=...)`, capacity.py:276-307 — our
tree already supports both hooks) must be fed `dspark.memory`-based numbers so capacity planning
stays honest. The drafter also needs one `X3Scratch` for the verifier head (§1) — sized by the
head's `lin.k/n`, a few MiB, counted with the fixed buffers.

---

## 4. Decode loop: `dflash_decode` for our family

Port Mia's `dflash_decode` (her decode.py:984-1058) into `glm_moe_dsa/cuda/decode.py`, restated
over our `commit` exactly the way our `mtp_decode` (decode.py:199-284) already is. Structure:

```
out = [pending]; depth = policy.next(0,0)
while len(out) < count and not stop:
    copied = copies.propose(copy_room(copies, count, out)) if copies else []
    drafts = copied if copied else (
        drafter.propose(out[-1], depth, sampling, policy.confidence, **policy.chain_rule)
        if depth > 0 else [])
    tokens = e.verify_window([out[-1]] + drafts); R = len(tokens)
    logits = e.forward(tokens)                      # our Engine.forward (graph or eager)
    sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
    keep = 1 + (first i where sampled[i] != drafts[i], stop at mismatch / eos)
    fwd.commit(w, st, b, R, keep)                   # our commit: st.set_pos(st.pos + keep)
    drafter.add_taps(e.tap_rows(keep))              # ← the drafter context grows by the kept rows
    e.follow(sampled[:keep]); out.extend(sampled[:keep])
    if copies: copies.extend(sampled[:keep])
    depth = min(policy.next(*((0, 0) if copied else (len(drafts), keep - 1))), count - len(out))
```

Differences from our `mtp_decode` (the loop it replaces when a drafter is attached):
- **No MTP absorb step**: the DSpark drafter's context is its K/V ring, fed by taps; the MTP head
  is simply not loaded/used when dspark is on (engine picks one drafter; Mia's engine refuses
  DFlash2-on-full-GLM the same way, her engine.py:709-711).
- **`drafter.add_taps(e.tap_rows(keep))` after every round** — the taps of the *kept* rows only,
  from the decode buffers (`e.buf.taps`), positions `context_end .. +keep-1` matching the ring.
  This is the line that makes drafted == serial (below).
- **Copy drafts first, plain**: `copies.propose(room)` returns non-empty → those tokens ARE the
  round's drafts; the DSpark block is not run that round (saves the block pass), but the kept
  rows' taps are still absorbed, so the drafter context never goes stale. A copied round feeds
  `(0, 0)` to the depth policy, exactly like our `mtp_decode` does (decode.py:270) — copy
  acceptance says nothing about DSpark's chain quality. This is Mia's `dflash_decode` with
  `copies` and without `hybrid`; her `hybrid_round` (TF_GLM_COPY_HYBRID, her decode.py:1061-1083,
  vetting copy continuations against the DSpark block via `propose_hybrid`/`slot_scorer`) is a
  **later, optional** addition — our `copy_drafts.py` already ships `plan`/`HybridSettings`/
  `padded`/`hybrid` (ours :269-318, :380-426 mirror hers), so it ports without new machinery,
  but it is not needed for the first working version.
- **Depth policy**: our `AcceptPolicy`/`DepthPolicy` (decode.py:24-62, shared DepthPolicy at
  glm5_next decode.py:473) work as-is; dspark's `propose` accepts `beta`/`round_ms` kwargs
  (`policy.chain_rule()`) — with our `DEPTH_COST=0` default, `AcceptPolicy` is not constructed
  and `DepthPolicy(fixed=True)` passes `confidence` only. `round_ms`-based `best_depth` cutting
  needs a calibration pass (Mia's `_calibrate`, her engine.py:1195-1235) — optional step 5.

**Exactness argument.** The reply is produced by the same serial rule as without a drafter:
1. `e.forward` is the unmodified family forward on the window `[pending + drafts]`; its logits
   for row r depend only on tokens up to row r (causal), never on later rows — dropping drafts
   changes only how many rows are computed, not any row's bits.
2. `e.sample` is the shared keyed sampler over the same gathered candidates; the sampled token at
   position p is a function of (seed, p, logits row) — identical to a serial step at p.
3. The loop keeps `sampled[:keep]` where `keep` is the first mismatch: every kept token equals
   what serial decoding would have produced (by 2), and the first new token after a mismatch is
   the serial sample at that position. Drafts only ever *propose* window rows.
4. `drafter.add_taps` consumes taps of committed rows only, after `commit` advanced `st.pos` by
   `keep` — the drafter's context is exactly the committed prefix, so its proposals are a pure
   function of committed tokens (plus its own weights). A wrong or stale drafter can only lower
   acceptance, never change a reply. Same argument our copy-drafts port used (TF_GLM_COPY_DRAFTS
   verified 4/4 byte-identical).
5. All ranks run the same host arithmetic (`merge_candidates`, the Markov/confidence chain in
   float64 on replicated host tables) from the same gathered candidates → identical drafts on
   every rank → identical windows → identical logits → the existing TP exactness invariant holds.

One subtlety: `stop_eos` — the loop condition and the `keep` loop already treat EOS as a mismatch
(copied from our `mtp_decode` :243-246), so a draft that continues past EOS is truncated the same
way.

---

## 5. Step plan (each step independently testable in the engine container)

**Step 1 — Restore taps in glm_moe_dsa (no drafter yet).**
Files: `families/glm_moe_dsa/cuda/forward.py` (drop the `del self.taps, self.tap_at` at :256; add
the `b.taps[slot][:R].copy_(b.x[:R])` fill after `layer_forward` in `compute` :781 and after the
final residual in `prompt_layers` :721-722), `families/glm_moe_dsa/cuda/decode.py` (`Engine.__init__`
gains `taps=()` param → `self.buf.set_taps(...)`/`self.pbuf.set_taps(...)` before `Graphs`;
`tap_rows` :367 returns `torch.cat` instead of raising), `families/glm_moe_dsa/cuda/engine.py`
(pass `taps=()` — empty until a drafter exists; add the prefill-tap MiB term to `_geometry`).
Test (no model needed): CPU/GPU unit test in the style of `tests/cuda/test_glm_draft_ring.py` —
build the tiny fake checkpoint (`tests/glm_dsa_fakes.py` / `test_glm_engine.py` fixtures), run
`prefill` + a few `e.forward` windows with `set_taps((1,))`, assert `tap_rows(n)` equals
`b.x[:n]` snapshots taken at the right layer (compare against a manual hook), and assert
**byte-identical replies** with taps on vs off (the fill is a no-op when `tap_at` is empty).
Also `pytest tests/cuda/test_glm_engine.py` unchanged (regression: no drafter → no behavior).

**Step 2 — Base-class upgrades in glm5_next/dflash2.py + dspark.py itself.**
Files: `families/glm5_next/cuda/dflash2.py` (`tap_rows` ctor param + capture range `1..tap_rows`;
extract `merge_candidates` from `candidates`; add `best_depth`), `families/glm5_next/cuda/qmm.py`
(`draft_quantize4` minmax/mse, or pin minmax), new `families/glm_moe_dsa/cuda/dspark.py`
(Mia's file with the two adaptations from §1: X3 head call + our `forward.embed`; local
`draft_parts` via `share`/`share_lo`), `families/glm5_next/cuda/tp.py` or a small split helper.
Test: pure-CPU test with a **fake DSpark checkpoint** (extend the `_drafter` fixture writer to
emit dspark tensor names from `expected_tensors`, config from our real
`~/models/GLM-5.3-speculator.dspark/config.json` scaled down): `Drafter(path, fake_w,
world=1)` loads, `add_taps` + `propose` run on CPU (triton kernels need GPU — on the container's
GPU use tiny sizes), `nbytes` matches `memory()`, `merge_candidates`/`best_depth` pinned against
numpy references. No full model anywhere.

**Step 3 — Engine wiring: load, capacity, prefill feed.**
Files: `families/glm_moe_dsa/cuda/engine.py` (`--drafter` path: `is_dspark` check → construct
`dspark.Drafter(drafter, w, capacity=…, tap_rows=MAX_ROWS, ring=True)`; `admit(..., draft_weights=
dspark_weights, draft_geometry=dspark_geometry)`; pass `taps=drafter.tap_layers` to `Decoder`;
`drafter.capture()` after graphs; add `int(dspark)` to the ranks' settings comparison list
`mine` at :100), `families/glm_moe_dsa/cuda/decode.py` (`prefill` gains `drafter=None` param,
calls `drafter.add_taps(e.tap_rows(R, e.pbuf))` per chunk + `drafter.reset()` on reset).
Test: fake-checkpoint `GlmEngine(path, rank=0, comm=_TwoCopies(), drafter=path/"dspark")` boots,
prefill of 200 tokens leaves `drafter.context_end == 200`, `pos_dev` aligned, capacity plan
accounts for the drafter's bytes; settings-mismatch refusal fires when one rank has the drafter
and another doesn't.

**Step 4 — `dflash_decode` + copy-draft combination.**
Files: `families/glm_moe_dsa/cuda/decode.py` (new `dflash_decode`, §4; dispatch in engine's
`_run_once` :316-336: drafter attached → `dflash_decode` instead of `mtp_decode`, `copies =
self.copy.drafts(..., hybrid=False)`).
Test: fake engine, greedy + seeded-sampling replies **byte-identical** across four modes: serial,
MTP (if fake has MTP), dspark, dspark+copies — the exactness argument made executable. Then the
real container: `TF_GLM_DSPARK=1` vs off on the 4 reference replies + one code-edit reply
(same harness as the copy-drafts verification), plus the decode benchmark (tok/s, rounds,
accepted).

**Step 5 (optional) — Hybrid copy vetting + `round_ms` depth.**
Files: `decode.py` (`hybrid_round`), `engine.py` (`_calibrate` for `round_ms`, `TF_GLM_COPY_HYBRID`
setting in the comparison list).
Test: hybrid on vs off byte-identical; acceptance/throughput delta reported, not asserted.

## 6. Risks and unknowns

1. **EXL3 head in the block pass** — the biggest unknown. `X3.__call__` needs an `X3Scratch`
   (fp16 rotations, fp32 K-split partials) and writes bf16 or fp32; dspark wants fp32 logits for
   `topk` fidelity. The MTP path already proves X3-in-a-draft-loop works (`mm` wrapper,
   forward.py:179-187), but the drafter's scratch must be allocated **before** `drafter.capture()`
   (X3 allocates counters in `__post_init__` — fine — but the scratch tensors must exist at
   capture). Fallback if fp32-out is awkward: bf16 logits + `.float()` before topk (the MTP
   sampler already does `logits.float()`).
2. **`_prep_kernel` head-count mismatch**: dspark is MHA (64=64); our `_prep` handles any
   heads/kvh split, but the kernel's grid is `(rows, heads + 2*kvh)` — at 16+32=48 programs/row
   it's fine. Verified same kernel in both trees; low risk.
3. **Prefill tap memory**: 480 MiB of `pbuf` taps at PROMPT_ROWS=8192. If tight, `set_taps` can
   allocate prefill taps lazily/smaller and feed `add_taps` in slices — but the simple full-size
   version matches Mia and the GB10 has headroom. Must be in the capacity estimate either way.
4. **`tap_in` is [64, 5·D]**: a verify window keeps ≤ 8 rows (fine); prompt chunks slice at 64 —
   with `tap_rows=64` (step 2) each slice is a single graph replay; without it, 8-row replays +
   eager 64-row computes. Not a correctness risk, only speed.
5. **Ring wrap at long contexts**: the drafter ring (2176 rows) holds a 2047-row window; contexts
   past 2048 wrap and drafts stay exact (ring arithmetic is position-modulo, tested for DFlash2 in
   `test_glm_draft_ring.py`) — but *acceptance* on very long contexts is unmeasured. The window
   semantics (`span=2048`) come from the checkpoint's sliding_window, not our engine's sparse
   attention — a mismatch in what "the last 2048 tokens" means (our DSA selects top-k; DSpark
   attends a dense window) is fine for proposals but may lower acceptance; unknown until benched.
6. **Host Markov tables at 4 ranks**: 151 MiB/rank host RAM × 4 — trivial, but the chain runs
   per-round on rank 0's path only? No — **every rank** runs the chain (drafts must agree); the
   tables are replicated so they do. Confirmed by Mia's design; just noting the RAM.
7. **`DSPARK_FILTER`/`DSPARK_NOISE` env knobs** (dspark.py:191-197) change *which* drafts are
   proposed under sampling — they must join the ranks' startup comparison (`mine` list) or a
   divergent env var silently desyncs ranks (drafts differ → windows differ → hang). Same class
   of bug the settings list already guards; add them in step 3.
8. **Mia's dflash2 is newer than ours** (`tap_rows`, `merge_candidates`, `best_depth`,
   `noisy_confidence`, pinned-copy candidate readback, `draft_quantize4`). We take only the
   pieces dspark imports — deliberately NOT her `chain`/`propose` changes (noise-aware stops),
   which would alter MTP behavior too. Keep the diff minimal and dspark-scoped.
9. **`world=1` tests vs TP4**: the fake-weights tests run one rank; TP4-only failures (uneven
   splits don't exist at 4, but `_row` gather order, `merge_candidates` over 4 shards) need the
   real container. The copy-drafts port validated the same path; repeat its 4-reference
   byte-identity check before any benchmark claim.
