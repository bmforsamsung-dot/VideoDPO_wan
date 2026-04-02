import argparse
import os
from pathlib import Path

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from peft import LoraConfig
from pytorch_lightning import seed_everything


_DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20230211)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--savedir", type=str, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--num_videos_per_prompt", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--enable_model_cpu_offload", action="store_true", default=None)
    parser.add_argument("--enable_slicing", action="store_true", default=None)
    parser.add_argument("--enable_tiling", action="store_true", default=None)
    args = parser.parse_args()
    return merge_config_args(args)


def merge_config_args(args):
    defaults = {
        "height": 480,
        "width": 720,
        "num_frames": 49,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
        "num_videos_per_prompt": 1,
        "fps": 8,
        "dtype": "bf16",
        "device": "cuda",
        "enable_model_cpu_offload": False,
        "enable_slicing": False,
        "enable_tiling": False,
    }

    config_values = {}
    if args.config:
        cfg = OmegaConf.load(args.config)
        for section_name in ("model", "inference"):
            section = cfg.get(section_name)
            if section is not None:
                config_values.update(OmegaConf.to_container(section, resolve=True))

    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, config_values.get(key, value))

    for key in ("model_path", "ckpt_path", "prompt_file", "savedir"):
        if getattr(args, key) is None:
            setattr(args, key, config_values.get(key))
    args.lora_args = config_values.get("lora_args")

    missing = [key for key in ("model_path", "prompt_file", "savedir") if not getattr(args, key)]
    if missing:
        raise ValueError(
            "Missing required inference arguments: "
            + ", ".join(missing)
            + ". Provide them via CLI or --config."
        )
    return args


def load_prompts(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def slugify(text, max_len=48):
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return (safe or "sample")[:max_len]


def infer_lora_config(transformer_state_dict):
    lora_a_keys = [key for key in transformer_state_dict.keys() if ".lora_A." in key]
    if len(lora_a_keys) == 0:
        return None

    adapter_name = lora_a_keys[0].split(".lora_A.", 1)[1].rsplit(".weight", 1)[0]
    rank = int(transformer_state_dict[lora_a_keys[0]].shape[0])
    return {
        "adapter_name": adapter_name,
        "lora_rank": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "lora_scale": 1.0,
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
    }


def add_lora_adapter(pipe, lora_config_dict):
    adapter_name = lora_config_dict.get("adapter_name", "default")
    lora_config = LoraConfig(
        r=int(lora_config_dict.get("lora_rank", 64)),
        lora_alpha=int(lora_config_dict.get("lora_alpha", lora_config_dict.get("lora_rank", 64))),
        lora_dropout=float(lora_config_dict.get("lora_dropout", 0.0)),
        bias="none",
        target_modules=list(
            lora_config_dict.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"])
        ),
    )
    pipe.transformer.add_adapter(lora_config, adapter_name=adapter_name)
    pipe.transformer.set_adapters(
        [adapter_name], adapter_weights=[float(lora_config_dict.get("lora_scale", 1.0))]
    )
    return adapter_name


def load_transformer_checkpoint(pipe, ckpt_path, lora_config_override=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    transformer_state_dict = {
        key[len("transformer.") :]: value
        for key, value in state_dict.items()
        if key.startswith("transformer.")
    }
    if len(transformer_state_dict) == 0:
        raise ValueError(f"No transformer weights found in checkpoint: {ckpt_path}")

    lora_config = lora_config_override or ckpt.get("cogvideox_lora_config")
    has_lora_weights = any(".lora_A." in key or ".lora_B." in key for key in transformer_state_dict)
    if has_lora_weights:
        if lora_config is None:
            lora_config = infer_lora_config(transformer_state_dict)
        if lora_config is None:
            raise ValueError(
                f"Detected LoRA weights in checkpoint but failed to infer LoRA config: {ckpt_path}"
            )
        adapter_name = add_lora_adapter(pipe, lora_config)
    else:
        adapter_name = None

    missing, unexpected = pipe.transformer.load_state_dict(transformer_state_dict, strict=False)
    print(
        f"Loaded transformer checkpoint from {ckpt_path}. "
        f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}"
    )
    if adapter_name is not None:
        print(f"Activated LoRA adapter: {adapter_name}")


def main():
    args = parse_args()
    seed_everything(args.seed)

    dtype_key = args.dtype.lower()
    if dtype_key not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    torch_dtype = _DTYPE_MAP[dtype_key]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    prompts = load_prompts(args.prompt_file)
    os.makedirs(args.savedir, exist_ok=True)

    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype)
    if args.enable_slicing:
        pipe.vae.enable_slicing()
    if args.enable_tiling:
        pipe.vae.enable_tiling()

    ckpt_path = args.ckpt_path or None
    if ckpt_path is not None and ckpt_path.strip():
        load_transformer_checkpoint(pipe, ckpt_path, lora_config_override=args.lora_args)

    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    for idx, prompt in enumerate(prompts, start=1):
        generator_device = "cpu" if args.enable_model_cpu_offload else device
        generator = torch.Generator(device=generator_device).manual_seed(args.seed + idx - 1)
        result = pipe(
            prompt=prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_videos_per_prompt=args.num_videos_per_prompt,
            generator=generator,
        )

        for sample_idx, frames in enumerate(result.frames):
            filename = f"{idx:04d}_{sample_idx:02d}_{slugify(prompt)}.mp4"
            save_path = Path(args.savedir) / filename
            export_to_video(frames, str(save_path), fps=args.fps)
            print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
