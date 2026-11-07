#!/usr/bin/env bash
set -euo pipefail
# Run on Hopper after loading an available Python 3.12 module. Never copy a Mac venv.
root="$(cd "$(dirname "$0")/../.." && pwd)"
target="${1:-/scratch/$USER/turtlehero/env}"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -c 'import sys,platform; assert sys.version_info[:2] == (3,12), "Load Python 3.12 first"; assert platform.system()=="Linux" and platform.machine()=="x86_64", "Build this environment on Hopper Linux x86_64"'
if [[ -e "$target" ]]; then
  echo "Environment already exists: $target. Use a new path to avoid changing an existing run." >&2
  exit 1
fi
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/turtlehero/pip-cache}"
"$python_bin" -m venv "$target"
"$target/bin/python" -m pip install --upgrade pip
"$target/bin/python" -m pip install 'torch==2.14.0+cu126' 'torchvision==0.29.0+cu126' --index-url https://download.pytorch.org/whl/cu126
"$target/bin/python" -m pip install -r "$root/cluster/hopper/requirements.txt"
"$target/bin/python" -m pip check
"$target/bin/python" -m pip freeze > "$target/environment.lock.txt"
echo "Environment created: $target. Run a submitted smoke job to validate CUDA on the allocated GPU."
