#!/usr/bin/env bash
# Start one TensorFold rank per node (ranks 3,2,1 first, then rank 0 which serves the API).
#   DRY_RUN=1 ./tp4_start.sh     print each rank's command, change nothing
#   ./tp4_start.sh               start (the FIRST start compiles CUDA kernels on every rank: allow 10 to 25 minutes)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./env.sh
(( ${#NODES[@]} == 4 )) || { echo "NODES must list 4 addresses"; exit 1; }
NCCL="NCCL_SOCKET_IFNAME=$SOCKET_IFNAME NCCL_IB_HCA=$NCCL_IB_HCA NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN NCCL_DEBUG=WARN"
TFENV=$(env | grep -E '^(TF_TP_WORLD|TF_GLM_[A-Z0-9_]+)=[A-Za-z0-9_.,:+-]*$' | tr '\n' ' ')
for r in 3 2 1 0; do
  if (( r == 0 )); then
    serve="${SERVE_ENV_PREFIX}tensorfold serve $MODEL_DIR --tp 4 --rank 0 --master $MASTER_ADDR --master-port $MASTER_PORT $SERVE_EXTRA --context $CTX --host 127.0.0.1 --port $PORT --name $SERVED_NAME --temperature 0"
  else
    serve="${SERVE_ENV_PREFIX}tensorfold serve $MODEL_DIR --tp 4 --rank $r --master $MASTER_ADDR --master-port $MASTER_PORT $SERVE_EXTRA --context $CTX"
  fi
  cmd="cd $WORKDIR && $TFENV $NCCL ./tfrun.sh $CONTAINER_NAME \"$serve\""
  if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "[dry-run] rank $r on ${NODES[$r]}: ssh -n $SSH_USER@${NODES[$r]} $cmd"; continue; fi
  echo "starting rank $r on ${NODES[$r]}"
  rssh "${NODES[$r]}" "$cmd" >/dev/null
done
[[ "${DRY_RUN:-0}" == 1 ]] || echo "launched. Next: ./tp4_wait.sh"
