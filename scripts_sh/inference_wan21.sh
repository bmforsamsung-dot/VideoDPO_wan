#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Override these with environment variables if needed.
# Example:
#   MODEL_PATH=Wan-AI/Wan2.1-T2V-14B-Diffusers \
#   PROMPT_FILE=prompts/my_prompts.txt \
#   bash scripts_sh/inference_wan21.sh
config="${CONFIG:-configs/inference/inference_wan21_14b.yaml}"
model_path="${MODEL_PATH:-Wan-AI/Wan2.1-T2V-14B-Diffusers}"
ckpt_path="${CKPT_PATH:-}"  # keep empty for base model, or set a full fine-tuned checkpoint
prompt_file="${PROMPT_FILE:-prompts/test_prompts.txt}"
res_dir="${RES_DIR:-results}"
name="${NAME:-wan21_14b}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" scripts/inference_wan21.py \
--seed 123 \
--config "${config}" \
--model_path "${model_path}" \
--ckpt_path "${ckpt_path}" \
--prompt_file "${prompt_file}" \
--savedir "${res_dir}/${name}" \
--num_videos_per_prompt 1 \
--height 480 --width 832 \
--num_frames 81 \
--guidance_scale 5.0 \
--num_inference_steps 50 \
--flow_shift 3.0 \
--fps 16 \
--dtype bf16 \
--enable_slicing \
--enable_tiling
