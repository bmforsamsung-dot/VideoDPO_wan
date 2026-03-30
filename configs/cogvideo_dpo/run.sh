#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

current_time=$(date +%Y%m%d%H%M%S)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXPNAME="${EXPNAME:-cogvideox-dpo-train}"
CONFIG="${CONFIG:-configs/cogvideo_dpo/config.yaml}"
LOGDIR="${LOGDIR:-./results/dpo-cogvideox}"
DEVICES="${DEVICES:-0,1,2,3}"

IFS=',' read -r -a GPU_LIST <<< "$DEVICES"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_LIST[@]}}"

cd "${ROOT_DIR}"

python -m torch.distributed.run \
--nnodes=1 \
--nproc_per_node="${NPROC_PER_NODE}" \
scripts/train.py \
-t --devices "${DEVICES}" \
lightning.trainer.num_nodes=1 \
--base "${CONFIG}" \
--name "${EXPNAME}_${current_time}" \
--logdir "${LOGDIR}" \
--auto_resume True
