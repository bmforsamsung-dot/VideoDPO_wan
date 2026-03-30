config='configs/inference/inference_cogvideox_2b.yaml'
model_path='THUDM/CogVideoX-2b'
ckpt_path=''  # e.g. results/dpo-cogvideox/your_exp/checkpoints/last.ckpt ; keep empty for base model
prompt_file="prompts/test_prompts.txt"
res_dir="results"
name="cogvideox_2b"

python3 scripts/inference_cogvideox.py \
--seed 123 \
--config $config \
--model_path $model_path \
--ckpt_path $ckpt_path \
--prompt_file $prompt_file \
--savedir $res_dir/$name \
--num_videos_per_prompt 1 \
--height 480 --width 720 \
--num_frames 49 \
--guidance_scale 6.0 \
--num_inference_steps 50 \
--fps 8 \
--dtype bf16 \
--enable_slicing \
--enable_tiling
