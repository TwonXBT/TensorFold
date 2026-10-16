#!/usr/bin/env bash
# Run ON EACH NODE (or: for n in ...; do ssh $n 'bash -s' < setup_node.sh; done). Installs nothing system-wide.
# Needs: docker with the NVIDIA runtime, git, ~330 GB free for the weights, 25 GB for the container image.
set -euo pipefail
TAG="${TAG:-glm53-4spark-v1}"
mkdir -p "$HOME/glm53-tf/cache" "$HOME/models"
cd "$HOME/glm53-tf"
if [ ! -d TensorFold/.git ]; then git clone https://github.com/TwonXBT/TensorFold.git TensorFold; fi
git -C TensorFold fetch --tags origin
git -C TensorFold checkout -f "$TAG"
cp TensorFold/recipe/scripts/tfrun.sh ./tfrun.sh && chmod +x ./tfrun.sh
docker pull nvcr.io/nvidia/pytorch:26.07-py3
echo "node ready: $(git -C TensorFold rev-parse --short HEAD) ($TAG)"
