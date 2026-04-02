#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Override these with environment variables if needed.
# Example:
#   CKPT_PATH=results/dpo-cogvideox-lora/your_exp/checkpoints/last.ckpt \
#   PROMPT_FILE=prompts/my_prompts.txt \
#   bash scripts_sh/inference_cogvideox_lora.sh
config="${CONFIG:-configs/inference/inference_cogvideox_2b_lora.yaml}"
model_path="${MODEL_PATH:-THUDM/CogVideoX-2b}"
ckpt_path="${CKPT_PATH:-}"  # e.g. results/dpo-cogvideox-lora/your_exp/checkpoints/last.ckpt
prompt_file="${PROMPT_FILE:-prompts/test_prompts.txt}"
res_dir="${RES_DIR:-results}"
name="${NAME:-cogvideox_2b_lora}"

if [ -z "${ckpt_path}" ]; then
  echo "Set CKPT_PATH to a CogVideoX LoRA checkpoint, or edit scripts_sh/inference_cogvideox_lora.sh."
  exit 1
fi

cd "${ROOT_DIR}"

"${PYTHON_BIN}" scripts/inference_cogvideox.py \
--seed 123 \
--config "${config}" \
--model_path "${model_path}" \
--ckpt_path "${ckpt_path}" \
--prompt_file "${prompt_file}" \
--savedir "${res_dir}/${name}" \
--num_videos_per_prompt 1 \
--height 480 --width 720 \
--num_frames 49 \
--guidance_scale 6.0 \
--num_inference_steps 50 \
--fps 8 \
--dtype bf16 \
--enable_slicing \
--enable_tiling
