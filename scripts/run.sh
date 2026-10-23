#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${1:?Usage: run.sh StereoGS|TWINGS|StableGS|tools script.py [arguments]}"
shift
# Conda environment (override with CERTISPLAT_ENV) and CUDA 12.1 toolkit (override with CUDA_HOME).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CERTISPLAT_ENV:-certisplat}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_HOME="$ROOT/checkpoints/torch"
export HF_HOME="$ROOT/checkpoints/huggingface"
export QT_QPA_PLATFORM=offscreen
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
case "$METHOD" in
    StereoGS|TWINGS) cd "$ROOT/$METHOD" ;;
    StableGS)
        export PYTHONPATH="$ROOT/runtimes/stablegs${PYTHONPATH:+:$PYTHONPATH}"
        cd "$ROOT/StableGS"
        ;;
    tools) cd "$ROOT" ;;
    *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac
exec python "$@"
