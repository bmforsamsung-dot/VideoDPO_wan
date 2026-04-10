import logging
import random
from contextlib import contextmanager
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel
from peft import LoraConfig
from transformers import AutoTokenizer, UMT5EncoderModel


mainlogger = logging.getLogger("mainlogger")


_DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


class WanVideoDPO(pl.LightningModule):
    """Wan2.1-based DPO trainer compatible with the existing VideoDPO training stack."""

    def __init__(
        self,
        model_path,
        beta_dpo=5000.0,
        first_stage_key="video",
        cond_stage_key="caption",
        video_length=81,
        image_size=(480, 832),
        log_every_t=100,
        uncond_prob=0.0,
        torch_dtype="bf16",
        gradient_checkpointing=True,
        enable_slicing=True,
        enable_tiling=True,
        optimizer_beta1=0.9,
        optimizer_beta2=0.95,
        optimizer_epsilon=1e-8,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        max_sequence_length=512,
        lora_args=None,
        logdir=None,
        **unused_kwargs,
    ):
        super().__init__()
        del unused_kwargs

        self.model_path = model_path
        self.beta_dpo = beta_dpo
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.lora_args = lora_args if lora_args is not None else []
        self.use_lora = len(self.lora_args) != 0
        self.temporal_length = video_length
        self.image_size = list(image_size)
        self.log_every_t = log_every_t
        self.uncond_prob = uncond_prob
        self.logdir = logdir
        self.max_sequence_length = max_sequence_length

        self.optimizer_beta1 = optimizer_beta1
        self.optimizer_beta2 = optimizer_beta2
        self.optimizer_epsilon = optimizer_epsilon
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        dtype_key = str(torch_dtype).lower()
        if dtype_key not in _DTYPE_MAP:
            raise ValueError(
                f"Unsupported torch_dtype: {torch_dtype}. Expected one of {sorted(_DTYPE_MAP)}"
            )
        self.weight_dtype = _DTYPE_MAP[dtype_key]
        self.lora_adapter_name = getattr(self.lora_args, "adapter_name", "default")
        self.lora_rank = int(getattr(self.lora_args, "lora_rank", 64))
        self.lora_alpha = int(getattr(self.lora_args, "lora_alpha", self.lora_rank))
        self.lora_dropout = float(getattr(self.lora_args, "lora_dropout", 0.0))
        self.lora_scale = float(getattr(self.lora_args, "lora_scale", 1.0))
        default_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        self.lora_target_modules = list(
            getattr(self.lora_args, "target_modules", default_target_modules)
        )
        self.lora_injected = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=self.weight_dtype
        )
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=self.weight_dtype
        )
        self.ref_transformer = None
        if not self.use_lora:
            self.ref_transformer = WanTransformer3DModel.from_pretrained(
                model_path, subfolder="transformer", torch_dtype=self.weight_dtype
            )
        self.vae = AutoencoderKLWan.from_pretrained(
            model_path, subfolder="vae", torch_dtype=self.weight_dtype
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        if enable_slicing:
            self.vae.enable_slicing()
        if enable_tiling:
            self.vae.enable_tiling()
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        if self.ref_transformer is not None:
            self.ref_transformer.requires_grad_(False)
        self.transformer.requires_grad_(not self.use_lora)

        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_encoder.eval()
        if self.ref_transformer is not None:
            self.ref_transformer.eval()
        self.vae.eval()
        return self

    def inject_lora(self):
        if not self.use_lora:
            return
        if self.lora_injected:
            return

        self.transformer.requires_grad_(False)
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            target_modules=self.lora_target_modules,
        )
        self.transformer.add_adapter(lora_config, adapter_name=self.lora_adapter_name)
        self.transformer.set_adapters(
            [self.lora_adapter_name], adapter_weights=[self.lora_scale]
        )
        self._cast_trainable_params_to_fp32(self.transformer)

        trainable_params = sum(
            param.numel() for param in self.transformer.parameters() if param.requires_grad
        )
        total_params = sum(param.numel() for param in self.transformer.parameters())
        mainlogger.info(
            "Injected Wan LoRA adapter '%s' (rank=%d, alpha=%d, scale=%.4f). "
            "Trainable params: %d / %d"
            % (
                self.lora_adapter_name,
                self.lora_rank,
                self.lora_alpha,
                self.lora_scale,
                trainable_params,
                total_params,
            )
        )
        self.lora_injected = True

    @staticmethod
    def _cast_trainable_params_to_fp32(module: torch.nn.Module):
        for parameter in module.parameters():
            if parameter.requires_grad and parameter.dtype != torch.float32:
                parameter.data = parameter.data.to(torch.float32)

    @contextmanager
    def reference_transformer_context(self):
        if self.ref_transformer is not None:
            yield self.ref_transformer
            return

        if not self.use_lora or not self.lora_injected:
            yield self.transformer
            return

        self.transformer.disable_adapters()
        try:
            yield self.transformer
        finally:
            self.transformer.enable_adapters()
            self.transformer.set_adapters(
                [self.lora_adapter_name], adapter_weights=[self.lora_scale]
            )

    def configure_optimizers(self):
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            betas=(self.optimizer_beta1, self.optimizer_beta2),
            eps=self.optimizer_epsilon,
            weight_decay=self.weight_decay,
        )

    def _prepare_batch(self, batch: Dict, random_uncond: bool = False) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        videos = batch[self.first_stage_key].to(memory_format=torch.contiguous_format).float()
        if videos.dim() != 5:
            raise ValueError(f"Expected video tensor with 5 dims, got shape {tuple(videos.shape)}")
        if videos.shape[1] % 2 != 0:
            raise ValueError(
                f"DPO batch expects winner/loser videos concatenated on channel dim, got channel size {videos.shape[1]}"
            )

        videos = torch.cat(videos.chunk(2, dim=1), dim=0)

        prompts = list(batch[self.cond_stage_key])
        if random_uncond and self.uncond_prob > 0:
            prompts = ["" if random.random() < self.uncond_prob else prompt for prompt in prompts]

        dupfactor = batch.get("dupfactor")
        if dupfactor is None:
            dupfactor = torch.ones(len(prompts), device=self.device, dtype=torch.float32)
        else:
            dupfactor = dupfactor.to(self.device, dtype=torch.float32)
        return videos, prompts, dupfactor

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        tokenized = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        return self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def _norm_latents_for_wan(self, latents: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.vae.config, "latents_mean") or not hasattr(self.vae.config, "latents_std"):
            return latents

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
        latents_mean = latents_mean.view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype)
        latents_std = latents_std.view(1, self.vae.config.z_dim, 1, 1, 1)
        return (latents - latents_mean) * latents_std

    @torch.no_grad()
    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        videos = videos.to(self.device, dtype=self.vae.dtype)
        latent_dist = self.vae.encode(videos).latent_dist
        latents = latent_dist.sample()
        return self._norm_latents_for_wan(latents)

    def _predict_model_losses(
        self,
        latents: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        model: WanTransformer3DModel,
    ) -> torch.Tensor:
        batch_size = latents.shape[0]

        noisy_latents = self.scheduler.scale_noise(latents, timesteps, noise)
        model_pred = model(
            hidden_states=noisy_latents,
            encoder_hidden_states=prompt_embeddings.to(dtype=noisy_latents.dtype),
            timestep=timesteps,
            return_dict=False,
        )[0]

        target = noise - latents
        return torch.mean((model_pred.float() - target.float()) ** 2, dim=(1, 2, 3, 4)).reshape(batch_size)

    def compute_dpo_loss(
        self,
        latents: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        dupfactor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = latents.shape[0]
        if batch_size % 2 != 0:
            raise ValueError(f"DPO expects an even latent batch size, got {batch_size}")

        timestep_indices = torch.randint(
            0,
            len(self.scheduler.timesteps),
            (batch_size,),
            device=self.device,
        ).long()
        timesteps = self.scheduler.timesteps.to(self.device)[timestep_indices]
        noise = torch.randn_like(latents)

        model_losses = self._predict_model_losses(
            latents, prompt_embeddings, timesteps, noise, self.transformer
        )
        with self.reference_transformer_context() as ref_transformer:
            with torch.no_grad():
                ref_losses = self._predict_model_losses(
                    latents, prompt_embeddings, timesteps, noise, ref_transformer
                )

        model_losses_w, model_losses_l = model_losses.chunk(2)
        ref_losses_w, ref_losses_l = ref_losses.chunk(2)

        model_diff = model_losses_w - model_losses_l
        ref_diff = ref_losses_w - ref_losses_l
        inside_term = -0.5 * self.beta_dpo * (model_diff - ref_diff)

        pair_weight = dupfactor.to(device=self.device, dtype=model_diff.dtype)
        loss = (-pair_weight * F.logsigmoid(inside_term)).mean()

        logs = {
            "loss_simple": loss.detach(),
            "loss_model_w": model_losses_w.mean().detach(),
            "loss_model_l": model_losses_l.mean().detach(),
            "loss_ref_w": ref_losses_w.mean().detach(),
            "loss_ref_l": ref_losses_l.mean().detach(),
            "implicit_acc": (inside_term > 0).float().mean().detach(),
            "pair_weight": pair_weight.mean().detach(),
            "timestep": timesteps.float().mean().detach(),
        }
        return loss, logs

    def shared_step(self, batch: Dict, prefix: str, random_uncond: bool = False):
        videos, prompts, dupfactor = self._prepare_batch(batch, random_uncond=random_uncond)

        with torch.no_grad():
            latents = self.encode_video(videos)
            prompt_embeddings = self.encode_text(prompts)
            prompt_embeddings = prompt_embeddings.repeat(2, 1, 1)

        loss, logs = self.compute_dpo_loss(latents, prompt_embeddings, dupfactor)
        log_dict = {f"{prefix}/{key}": value for key, value in logs.items()}
        log_dict[f"{prefix}/loss"] = loss.detach()
        return loss, log_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(
            batch, prefix="train", random_uncond=self.uncond_prob > 0
        )
        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )
        if (batch_idx + 1) % self.log_every_t == 0:
            mainlogger.info(
                f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss.item():.6f}"
            )
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, prefix="val", random_uncond=False)
        self.log_dict(
            loss_dict,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        return loss

    @torch.no_grad()
    def log_images(self, batch, split="train", **kwargs):
        del split, kwargs
        videos = batch[self.first_stage_key]
        prompts = list(batch[self.cond_stage_key])

        logs = {"condition": prompts}
        if videos.dim() == 5 and videos.shape[1] % 2 == 0:
            winner, loser = videos.chunk(2, dim=1)
            logs["winner"] = winner
            logs["loser"] = loser
        else:
            logs["inputs"] = videos
        return logs

    def on_save_checkpoint(self, checkpoint):
        keys_to_remove = [
            key for key in checkpoint["state_dict"].keys() if key.startswith("ref_transformer.")
        ]
        for key in keys_to_remove:
            checkpoint["state_dict"].pop(key, None)

        if self.use_lora:
            checkpoint["wan_lora_config"] = {
                "adapter_name": self.lora_adapter_name,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "lora_scale": self.lora_scale,
                "target_modules": list(self.lora_target_modules),
            }
        return checkpoint
