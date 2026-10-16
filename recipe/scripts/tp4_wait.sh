#!/usr/bin/env bash
# Wait (up to 40 min) until rank 0 answers, then send one short completion as a smoke test.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./env.sh
for i in $(seq 1 240); do
  if rssh "${NODES[0]}" "curl -sf http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then break; fi
  (( i % 6 == 0 )) && echo "waiting ($((i/6)) min)... rank 0 log: ssh ${NODES[0]} docker logs --tail 5 $CONTAINER_NAME"
  sleep 10
done
rssh "${NODES[0]}" "curl -s http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' -d '{\"model\":\"$SERVED_NAME\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}'" | head -c 600; echo
