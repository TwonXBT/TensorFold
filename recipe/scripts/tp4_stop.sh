#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./env.sh
for r in 3 2 1 0; do rssh "${NODES[$r]}" "docker rm -f $CONTAINER_NAME >/dev/null 2>&1 && echo rank $r stopped || echo rank $r not running"; done
