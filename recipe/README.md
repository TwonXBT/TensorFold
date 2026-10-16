# Full GLM-5.3 on four DGX Sparks: the `glm53-4spark` build

Full GLM-5.3 (the 753B non-Flash model, 40B active) in EXL3 3.0 bpw, tensor-parallel over four NVIDIA DGX Spark (GB10)
nodes, served through an OpenAI-compatible API by TensorFold. **This is a fork, not a new engine.** Everything here stands
on other people's work (credits below). What this fork adds is speed and conversation handling for that one setup.

> Unofficial and unaffiliated with Z.AI, NVIDIA, ashhart, bertholomus, Red Hat or turboderp. Use at your own risk.
> The numbers below are our own measurements on one cluster, not independently reproduced.

**Want an AI to set it up for you?** Open [`AI-SETUP-PROMPT.md`](AI-SETUP-PROMPT.md), copy everything below its divider into
your AI coding agent (one that can run shell commands on, or ssh into, your four nodes), and follow its questions.

## What runs where

| Piece | Source | Pinned to |
|---|---|---|
| Engine | this repo, branch `glm53-4spark` | tag `glm53-4spark-v1` |
| Main weights (292.7 GB, 39 shards) | `bertholomus/GLM-5.3-EXL3-3.0bpw` | revision `7f0573836c44803030912ffe5d051ca7587edfd6` |
| DSpark drafter (2.5 GB) | `RedHatAI/GLM-5.3-speculator.dspark` | revision `b374b95663447ea0e935151be4f3d6666e36e6d7` |
| Container | `nvcr.io/nvidia/pytorch:26.07-py3` | tag |

SHA-256 of every weight shard is in [`checksums/`](checksums). The weights are **not** in this repo and keep their own
licenses (GLM-5.3 License by Z.AI for both).

## Hardware and network we ran

- 4 x NVIDIA DGX Spark (GB10, 128 GB unified memory, driver 580.x, Docker 29, NVIDIA container runtime).
- One 200G ConnectX-7 cable per Spark into a MikroTik CRS804 switch, bridge MTU 9000. Direct links also work (upstream's recipe used direct RoCE).
- RoCE v2 working between all nodes; both RDMA devices of the cabled port used together ("dual rail", `NCCL_MIN_NCHANNELS=2`).
- About 330 GB free disk per node (weights are read by every rank; each rank keeps about 70 GiB resident).
- Optional: cap the GPU clock for heat. We run `nvidia-smi -lgc 0,2200` at boot (upstream saw 86 C under long prefills).
- A GB10 that runs out of memory tends to freeze instead of failing. Do not run other GPU jobs on these nodes while serving.

## Steps (what the AI prompt automates)

```bash
# on each node
git clone https://github.com/TwonXBT/TensorFold.git ~/glm53-tf/TensorFold && git -C ~/glm53-tf/TensorFold checkout glm53-4spark-v1
bash ~/glm53-tf/TensorFold/recipe/scripts/setup_node.sh        # engine checkout, tfrun.sh, container image
bash ~/glm53-tf/TensorFold/recipe/scripts/download_weights.sh  # pinned weights + sha256 check (about 295 GB)

# on any machine that can ssh to all four
cd ~/glm53-tf/TensorFold/recipe/scripts && $EDITOR env.sh      # node addresses, NIC and RDMA device names
DRY_RUN=1 ./tp4_start.sh && ./tp4_start.sh && ./tp4_wait.sh
```

First start compiles CUDA kernels on every rank: expect 10 to 25 minutes. Later starts take a few minutes.
The API listens on `127.0.0.1:18090` of rank 0 with **no authentication**; keep it on loopback or put your own gateway in front.

## What this fork changes (on top of bertholomus's `glm-dsa-tp4` and the TensorFold 0.6.1 release)

All additions are behind `TF_GLM_*` flags, listed with their values in `scripts/env.sh`. Output stays identical to plain serial
decoding: drafted, split and cached replies were checked equal to the serial reference.

- **DSpark drafter** (Red Hat's drafter checkpoint, engine code ported from Mia-AiLab's `glm5_next` implementation), with per-round
  confidence cuts, hybrid copy drafts, wide copy windows (up to 15 verify rows) and drafter resume.
- **Copy (prompt-lookup) drafts** for edit-heavy and code-quoting replies.
- **Conversation cache** (`TF_GLM_CONV_CACHE`, `TF_GLM_KEEP_SLOTS`, `TF_GLM_PROMPT_INVARIANT`): parks other conversations in free window rows; switching back
  does not re-prefill.
- **Selection and attention tuning for GB10**: position-split DSA selection, `topk_rows.cu`, sparse-attention launch shapes, EXL3 plan table, L2 prefetch.
- RoCE v2 GID index picked per NIC when `NCCL_IB_GID_INDEX < 0`.

The same changes are split into seven reviewable pull-request branches (`pr/1-...` to `pr/7-...`) for upstream.

## Measured on our cluster (engine = `glm53-4spark-v1`, DSpark + MTP-3, 262,144-token window)

| Test | Result |
|---|---|
| community-v1 suite (`tensorfold benchmark`, attached, **unranked**), code, T=0 / T=1 | 63.3 / 48.5 tok/s |
| same, chat, T=0 / T=1 | 38.8 / 36.8 tok/s |
| first text after request | 0.13 to 0.20 s |
| decode at 12K context, code / prose | 58.6 / 33.9 tok/s |
| decode at 50K context, code / prose | 57.2 / 33.1 tok/s |
| edit workload (reply quotes the prompt) | about 70 tok/s cold, up to about 88 tok/s warm |
| prefill, 27K-token prompt | 29.1 s (about 930 tok/s) |
| switching back to a parked 104K conversation, first token | 0.5 to 0.9 s |

For reference, upstream's published single-stream decode with MTP-3 only is 39.6 to 48.4 tok/s and its prefill is about 1,000 tok/s with the
replicated cache: our prefill is about the same; the gains here are in decode and in conversation switching.
Reproduce with `python recipe/bench/bench_community.py --url http://127.0.0.1:18090 --model GLM-5.3-EXL3-3.0bpw`
(a reimplementation of the community-v1 suite; the official CLI is in ashhart/TensorFold 1.0.x).

## Credits

- **Z.AI**: GLM-5.3 model and weights.
- **ashhart**: TensorFold, the engine this all runs on (Apache-2.0), including the EXL3 expert kernels and the `glm5_next` family.
- **bertholomus / BertholomusAI (Albert Lee)**: the `glm_moe_dsa` family (full GLM-5.3 on CUDA tensor-parallel ranks), the EXL3 3.0 bpw conversion, the `tfrun.sh` wrapper and the original 4-node recipe ([bertholomus/glm-5.3-tensorfold-tp4-4xgb10](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10)).
- **turboderp**: EXL3 format, exllamav3 and the reference implementation.
- **Red Hat AI**: the DSpark speculator checkpoint for GLM-5.3.
- **Mia's AI Lab ([MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold), Apache-2.0)**: the DSpark / DFlash2 / copy-draft reference implementation this fork's drafter port follows, and the three-Spark GLM-5.3 recipe that came before this one.
- **drowzeys**: TensorFold PR #159 (decode context parallelism design).
- **NVIDIA**: PyTorch container and NCCL.
- This fork's changes were written with Claude (Anthropic) and a locally served GLM-5.3, directed and tested by TwonXBT; see commit trailers.

## License

Apache-2.0 for the engine and these scripts, as upstream (see `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`). Files modified in this fork are marked
by git history. The model weights are under the GLM-5.3 License.
