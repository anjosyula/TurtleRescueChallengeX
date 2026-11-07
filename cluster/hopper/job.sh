#!/usr/bin/env bash
set -euo pipefail
umask 0027
: "${SLURM_JOB_ID:?Submit through hopper.py; do not train on a login node}"
root="$1"
config="$2"
venv="$3"
scratch="$4"
action="$5"
cd "$root"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export WANDB_MODE=disabled
export YOLO_CONFIG_DIR="$scratch/runtime/$SLURM_JOB_ID/yolo"
export MPLCONFIGDIR="$scratch/runtime/$SLURM_JOB_ID/matplotlib"
export TORCH_HOME="$scratch/cache/torch"
export TMPDIR="$scratch/runtime/$SLURM_JOB_ID/tmp"
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR" "$TMPDIR"
"$venv/bin/python" -m pip freeze > "$scratch/runtime/$SLURM_JOB_ID/environment.lock.txt"
nvidia-smi > "$scratch/runtime/$SLURM_JOB_ID/nvidia-smi.txt"
srun --ntasks=1 "$venv/bin/python" cluster/hopper/run.py --config "$config" --scratch "$scratch" --action "$action"
