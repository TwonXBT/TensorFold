# Paste-into-your-AI setup prompt

Copy everything below the line into an AI coding agent that can run shell commands (Claude Code, Codex, Cursor agent, etc.), running on a
machine that can `ssh` to all four of your DGX Sparks. The agent will ask you a few questions, then set up the cluster.

---

You are my ops engineer. Set up the full GLM-5.3 model (EXL3 3.0 bpw) on my four NVIDIA DGX Sparks (GB10) with tensor parallelism 4, using the
recipe in https://github.com/TwonXBT/TensorFold (engine tag glm53-4spark-v1, read recipe/README.md first) and the exact pinned sources below. Work step by step, verify each step before the next, and stop and ask me if anything differs from what
is written here. Do not improvise versions, do not use different weights, and do not skip checksum verification.

## Ground rules
- Do not run `sudo`, change system settings, install system packages, reboot a node or delete anything outside `~/glm53-tf` and `~/models` without asking me first and telling me exactly what the command does.
- Never print, store or ask for my passwords, SSH keys or tokens. Use my existing key-based ssh. If a Hugging Face login is needed, tell me to run `hf auth login` myself.
- The model server must listen on `127.0.0.1` only. Do not expose it to the network.
- A GB10 that runs out of memory freezes instead of failing. Do not start any other GPU workload on these nodes while this runs.
- Show me a dry run of the launch (`DRY_RUN=1`) and wait for my OK before the real start.

## Pinned sources (all public)
- Engine: `https://github.com/TwonXBT/TensorFold.git`, tag `glm53-4spark-v1` (branch `glm53-4spark`). Apache-2.0. It is a fork of `bertholomus/TensorFold` (branch `glm-dsa-tp4`), which is a fork of `ashhart/TensorFold`.
- Main weights: Hugging Face `bertholomus/GLM-5.3-EXL3-3.0bpw`, revision `7f0573836c44803030912ffe5d051ca7587edfd6` (292.7 GB, 39 safetensors shards; GLM-5.3 License).
- Drafter: Hugging Face `RedHatAI/GLM-5.3-speculator.dspark`, revision `b374b95663447ea0e935151be4f3d6666e36e6d7` (2.5 GB).
- Container: `nvcr.io/nvidia/pytorch:26.07-py3` (about 25 GB).
- Checksums of every weight shard are in the repo: `recipe/checksums/*.sha256`. The scripts check them.
- Full guide: `recipe/README.md` in the repo. Scripts: `recipe/scripts/`.

## Step 0: ask me
Ask me for: (1) the ssh address of each of the 4 nodes, rank 0 first (rank 0 will serve the API), (2) the ssh user, (3) whether the nodes are
connected by a 200G switch or direct cables, and the fast-fabric IP of each node. Then check, on every node, and report a table:
- `nvidia-smi` shows `NVIDIA GB10`, driver 580 or newer; `docker --version`; `docker run --rm --gpus all ubuntu nvidia-smi` works;
- `df -h ~` shows at least 330 GB free;
- the fast-fabric interface name (`ip -br link`, MTU 9000 on a switched fabric is best) and the RDMA devices of that port (`rdma link` or `ibv_devices`, the state must be ACTIVE);
- RDMA works between nodes (for example `ib_write_bw`, if installed; otherwise tell me and continue);
- `python3 --version` is 3.10 or newer.
Do not continue until every node passes, or tell me exactly what is missing.

## Step 1: engine and container on every node
On each node run `recipe/scripts/setup_node.sh` (clone the repo into `~/glm53-tf/TensorFold`, check out the tag, install `tfrun.sh`, pull the container image).
Because the repo is not cloned yet on first use, fetch the script with:
`curl -fsSL https://raw.githubusercontent.com/TwonXBT/TensorFold/glm53-4spark-v1/recipe/scripts/setup_node.sh | bash`.
Confirm `git -C ~/glm53-tf/TensorFold describe --tags` prints `glm53-4spark-v1` on all four nodes.

## Step 2: weights on every node
On each node run `bash ~/glm53-tf/TensorFold/recipe/scripts/download_weights.sh`. It downloads the two pinned revisions into `~/models/GLM-5.3-EXL3-3.0bpw` and
`~/models/GLM-5.3-speculator.dspark` and runs `sha256sum -c`. Every line must say `OK`. If a shard fails, delete that file and re-run (the download resumes).
To save bandwidth you may download on one node and `rsync -a` the two directories over the fast fabric to the others, then run the same checksum step on each.
The weights are about 295 GB per node: warn me about time and disk before starting.

## Step 3: configure
Edit `recipe/scripts/env.sh` (the "YOUR CLUSTER" block only): `SSH_USER`, `NODES`, `SOCKET_IFNAME`, `NCCL_IB_HCA`. If I have only one RDMA device cabled, set
`SERVE_ENV_PREFIX=""` and a single device in `NCCL_IB_HCA`. Leave every other value alone, and do not change the `TF_GLM_*` flags: all four ranks must match.

## Step 4: dry run, then start
Run `DRY_RUN=1 ./tp4_start.sh` from `recipe/scripts` and show me the four rank commands. After I approve, run `./tp4_start.sh`, then `./tp4_wait.sh`.
The first start compiles CUDA kernels on every rank: expect 10 to 25 minutes. Watch `docker logs --tail 20 tf-tp4` on each node; if a rank exits, show me its log and stop.
If a build hangs on a stale lock after an interrupted start, stop all ranks (`./tp4_stop.sh`), remove `~/glm53-tf/cache/torch_extensions` on that node
(the files may be root-owned: use `docker run --rm -v ~/glm53-tf/cache:/c alpine rm -rf /c/torch_extensions`) and start again.

## Step 5: verify
- `curl -s http://127.0.0.1:18090/v1/models` on rank 0 lists `GLM-5.3-EXL3-3.0bpw`.
- `tp4_wait.sh` prints a completion for "The capital of France is" that continues with Paris.
- Run `python3 recipe/bench/bench_community.py --url http://127.0.0.1:18090 --model GLM-5.3-EXL3-3.0bpw --reps 3` and report the table. On the author's cluster the medians
  were roughly: code T=0 63 tok/s, chat T=0 39 tok/s, code T=1 48, chat T=1 37. Different cabling or thermals can move these; large gaps (below half) mean something is wrong, usually the network
  (check `NCCL_IB_HCA` and that both ports are ACTIVE) or a missing `--drafter` flag.
- Tell me how to stop (`./tp4_stop.sh`) and restart (`./tp4_start.sh`).

## If something fails
Report the exact command, the last 30 log lines from the failing rank and your diagnosis. Do not change versions, flags or weights to "make it work" without asking.
Known facts: `TF_GLM_*` values must be identical on all ranks; the API has no authentication; the context window is 262,144 tokens; context larger than that may not fit in memory.
