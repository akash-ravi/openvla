#!/bin/bash -l
# DRAFT -- FOR THE CARC AGENT TO REVIEW. NOT SUBMITTED, NOT TESTED ON CARC.
# One resume-safe collection job for ONE task. Re-submitting the same command continues where the last job stopped.
#
#   sbatch -J collect_t3 --export=ALL,TASK_ID=3,EP_START=0,EP_END=10,OUT_ROOT=/scratch1/akashrav/data/pilot,\
#          CKPT=/path/to/openvla-7b-finetuned-libero-10,COLLECT_ENV=<conda env>,CUDA_HOME=<toolkit dir> \
#          experiments/robot/libero/slurm/collect_task.sbatch
#
# Values to verify on CARC (marked UNVERIFIED): account, partition/GPU spec, conda env name, CUDA_HOME, flash-attn availability.
#SBATCH -A yzhao010_1531
#SBATCH -p debug
#SBATCH -n 1
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/%x-%j.out

set -euo pipefail

: "${TASK_ID:?set TASK_ID}"; : "${EP_START:?set EP_START}"; : "${EP_END:?set EP_END}"
: "${OUT_ROOT:?set OUT_ROOT (scratch path with >=150 GB free)}"; : "${CKPT:?set CKPT (local checkpoint dir)}"
: "${COLLECT_ENV:?set COLLECT_ENV (conda env with the openvla fork, torch 2.2.0, transformers 4.40.1, LIBERO)}"
: "${CUDA_HOME:?set CUDA_HOME explicitly (toolkit matching torch/flash-attn)}"
ATTN="${ATTN:-flash_attention_2}"     # must be identical for the WHOLE collection (recorded in meta.attn_implementation)
SEED="${SEED:-7}"
TOKEN_LEVEL_EPS="${TOKEN_LEVEL_EPS:-0}"   # >0 only for verification episodes (large)

mkdir -p logs "$OUT_ROOT"
conda activate "$COLLECT_ENV"

export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1          # checkpoint must already be on disk (no downloads on compute nodes)
export HF_HOME="${HF_HOME:-/scratch1/$USER/hf_cache}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

echo "job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) task=$TASK_ID eps=[$EP_START,$EP_END) attn=$ATTN"
nvidia-smi --query-gpu=name,memory.total --format=csv
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "transformers", transformers.__version__, "cuda", torch.cuda.is_available(), "bf16", torch.cuda.is_bf16_supported())
PY
git rev-parse HEAD

cd "$(git rev-parse --show-toplevel)/experiments/robot/libero"    # the scripts expect to run from this directory
python run_libero_collect.py \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name libero_10 \
    --task_ids "[$TASK_ID]" \
    --episode_start "$EP_START" --episode_end "$EP_END" \
    --out_root "$OUT_ROOT" \
    --attn_implementation "$ATTN" \
    --seed "$SEED" \
    --token_level_episodes_per_task "$TOKEN_LEVEL_EPS"
