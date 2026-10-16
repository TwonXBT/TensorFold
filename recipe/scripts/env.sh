# Settings for the 4-node GLM-5.3 TP4 recipe. EDIT the "YOUR CLUSTER" block; leave the rest unless you are tuning.
# Sourced by tp4_start.sh, tp4_stop.sh and tp4_wait.sh. Run those from any machine that can ssh to all four nodes.

# ---- YOUR CLUSTER ---------------------------------------------------------------------------------
SSH_USER="${SSH_USER:-$(id -un)}"                 # same login on all four nodes, key-based ssh
NODES=(10.100.0.1 10.100.0.2 10.100.0.3 10.100.0.4)   # ssh address per rank, rank 0 first (rank 0 serves the API)
MASTER_ADDR="${NODES[0]}"                     # rank 0's address on the fast fabric (the RoCE-connected interface)
MASTER_PORT=29661
SOCKET_IFNAME="enp1s0f0np0"                   # NCCL bootstrap netdev (the cabled 200G port); see `ip -br link`
NCCL_IB_HCA="rocep1s0f0,roceP2p1s0f0"         # RDMA devices of that port; list yours with `ibv_devices` / `rdma link`
                                              # (the GB10 exposes two devices per port, one per PCIe half: using both = dual rail)
NCCL_IB_GID_INDEX="-1"                        # -1: NCCL picks each device's RoCE v2 GID
NCCL_NET_PLUGIN="spcx"                        # present in the NGC pytorch image
WORKDIR='$HOME/glm53-tf'                      # on every node: TensorFold/ (git checkout), tfrun.sh, cache/ (expanded on the node)

# ---- serve settings ---------------------------------------------------------------------------------
CTX=262144                                    # context window in tokens (fits the 128 GB budget at TF_GLM_KV=q5)
PORT=18090                                    # rank 0's OpenAI-compatible API, bound to 127.0.0.1 (no authentication!)
SERVED_NAME="GLM-5.3-EXL3-3.0bpw"
CONTAINER_NAME="tf-tp4"
MODEL_DIR=/models/GLM-5.3-EXL3-3.0bpw         # as the container sees it (tfrun.sh mounts ~/models at /models, read-only)
DRAFTER_DIR=/models/GLM-5.3-speculator.dspark
SERVE_EXTRA="--mtp-drafts 3 --drafter $DRAFTER_DIR"   # drop the --drafter part to run MTP-3 only (the upstream configuration)
# Two NCCL channels = one per RDMA device (dual rail). Set to an empty string if you cabled one device only.
SERVE_ENV_PREFIX="NCCL_MIN_NCHANNELS=2 NCCL_NCHANNELS_PER_NET_PEER=2 "

# ---- engine flags: ALL RANKS MUST HAVE THE SAME VALUES (the engine checks this at startup) -------------
export TF_TP_WORLD=4
export TF_GLM_KV=q5                 # 5-bit MLA latent cache, FP8 indexer keys (upstream recipe)
export TF_GLM_EMBED_SPLIT=1         # vocabulary-split embedding: +1.33 GiB a rank for cache (upstream recipe)
export TF_GLM_MTP_REUSE=2           # MTP layer reuses the main model's DSA selection (upstream recipe)
# Below: added by this fork (each is off in the engine unless set here)
export TF_GLM_KEEP_SLOTS=1          # keep the last conversation warm: an agent turn only prefills its new tokens
export TF_GLM_COPY_DRAFTS=1         # prompt-lookup ("copy") drafts
export TF_GLM_COPY_HYBRID=1         # copy drafts vetted by the DSpark block
export TF_GLM_COPY_MAX=14           # copy drafts a round
export TF_GLM_MAX_ROWS=15           # verify windows up to 15 rows
export TF_GLM_WIDE_GRAPHS=12,15     # CUDA-graph widths past 8 rows
export TF_GLM_DSPARK_CONFIDENCE=0.25   # DSpark chain cut (0.2 to 0.3 measured best)
export TF_GLM_X3_PLAN=gb10          # GB10 plan table for the EXL3 body linears
export TF_GLM_PROMPT_INVARIANT=1    # conversation cache resumes at any token, not only at 8,192-token chunk edges
export TF_GLM_DECODE_CH=192         # sparse-attention key chunk
export TF_GLM_DECODE_LAUNCH=8,2     # sparse-attention warps,stages
export TF_GLM_SELECT_SPLIT=1        # position-split DSA token selection
export TF_GLM_CONV_CACHE=4          # park up to 4 other conversations in free window rows
export TF_GLM_PREFETCH=3            # MiB of the next kernel's weights pulled into L2 while a decode gather waits

# ---- helper -------------------------------------------------------------------------------------------
rssh() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 "$SSH_USER@$1" "${@:2}"; }
