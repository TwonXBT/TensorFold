#!/usr/bin/env bash
# Run ON EACH NODE. Downloads the exact pinned revisions and verifies SHA-256. ~295 GB; resumable (re-run after an interruption).
# Alternative: download once, then copy ~/models/<dir> to the other nodes (rsync -a over the fast fabric) and run the verify step there.
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
MAIN_REPO=bertholomus/GLM-5.3-EXL3-3.0bpw
MAIN_REV=7f0573836c44803030912ffe5d051ca7587edfd6
DRAFT_REPO=RedHatAI/GLM-5.3-speculator.dspark
DRAFT_REV=b374b95663447ea0e935151be4f3d6666e36e6d7
python3 -m pip install -q --user -U "huggingface_hub[cli]" 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
DL="hf download"; command -v hf >/dev/null || DL="huggingface-cli download"
$DL "$MAIN_REPO"  --revision "$MAIN_REV"  --local-dir "$HOME/models/GLM-5.3-EXL3-3.0bpw"
$DL "$DRAFT_REPO" --revision "$DRAFT_REV" --local-dir "$HOME/models/GLM-5.3-speculator.dspark"
(cd "$HOME/models/GLM-5.3-EXL3-3.0bpw"        && sha256sum -c "$HERE/checksums/GLM-5.3-EXL3-3.0bpw.sha256")
(cd "$HOME/models/GLM-5.3-speculator.dspark"  && sha256sum -c "$HERE/checksums/GLM-5.3-speculator.dspark.sha256")
echo "weights verified"
