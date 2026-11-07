#!/usr/bin/env bash
#SBATCH --job-name=turtle-scene
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/turtle-scene-%j.out
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
# Arguments: project root, SIF image, prepared scene directory, new output directory.
root="$1"
sif="$2"
scene="$3"
output="$4"
scratch="/scratch/$USER"
runtime="${CONTAINER_RUNTIME:-apptainer}"
if ! command -v "$runtime" >/dev/null; then runtime=singularity; fi
mkdir -p "$scratch/turtlehero/container-cache" "$scratch/turtlehero/container-tmp"
export APPTAINER_CACHEDIR="$scratch/turtlehero/container-cache"
export APPTAINER_TMPDIR="$scratch/turtlehero/container-tmp"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export APPTAINERENV_MPLCONFIGDIR="$scratch/turtlehero/container-cache/matplotlib"
export APPTAINERENV_TORCH_HOME="$scratch/turtlehero/container-cache/torch"
export APPTAINERENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export SINGULARITYENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm must assign a GPU}"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export SINGULARITYENV_MPLCONFIGDIR="$APPTAINERENV_MPLCONFIGDIR"
export SINGULARITYENV_TORCH_HOME="$APPTAINERENV_TORCH_HOME"
srun --ntasks=1 "$runtime" exec --cleanenv --nv --bind "$root:$root,$scratch:$scratch,$sif:$sif:ro" \
  --pwd "$root" "$sif" python cluster/hopper/reconstruct.py \
  --scene "$scene" --output "$output" --container "$sif"
