#!/usr/bin/env bash
set -euo pipefail
# Separate from the detector's Python environment. Run on Hopper, not the Mac.
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Requires Hopper Linux x86_64' >&2; exit 1; }
runtime="${CONTAINER_RUNTIME:-apptainer}"
if ! command -v "$runtime" >/dev/null; then runtime=singularity; fi
target="${1:-/containers/dgx/UserContainers/$USER/nerfstudio-1.1.3.sif}"
[[ ! -e "$target" ]] || { echo "Already exists: $target" >&2; exit 1; }
mkdir -p "$(dirname "$target")" "/scratch/$USER/turtlehero/container-cache" "/scratch/$USER/turtlehero/container-tmp"
export APPTAINER_CACHEDIR="/scratch/$USER/turtlehero/container-cache"
export APPTAINER_TMPDIR="/scratch/$USER/turtlehero/container-tmp"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
image='docker://ghcr.io/nerfstudio-project/nerfstudio@sha256:e0da96a781c45b6bf46d08774a50dd8785c4d63be0a033c20e31bbce3e9a5121'
"$runtime" pull "$target" "$image"
sha256sum "$target" > "$target.sha256"
printf 'Nerfstudio 1.1.3 container: %s\n' "$target"
