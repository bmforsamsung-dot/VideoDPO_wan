import logging
import random
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from transformers import AutoTokenizer, T5EncoderModel


mainlogger = logging.getLogger("mainlogger")


_DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


class CogVideoXVideoDPO(pl.LightningModule):
    """CogVideoX-based DPO trainer that plugs into the existing VideoDPO pipeline."""

    def __init__(
        self,
        model_path,
        beta_dpo=5000.0,
        first_stage_key="video",
        cond_stage_key="caption",
        video_length=49,
        image_size=(480, 720),
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
        logdir=None,
        **unused_kwargs,
    ):
        super().__init__()
        del unused_kwargs

        self.model_path = model_path
        self.beta_dpo = beta_dpo
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.lora_args = []
        self.temporal_length = video_length
        self.image_size = list(image_size)
        self.log_every_t = log_every_t
        self.uncond_prob = uncond_prob
        self.logdir = logdir

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

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=self.weight_dtype
        )
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=self.weight_dtype
        )
        self.ref_transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=self.weight_dtype
        )
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_path, subfolder="vae", torch_dtype=self.weight_dtype
        )
        self.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        if enable_slicing:
            self.vae.enable_slicing()
        if enable_tiling:
            self.vae.enable_tiling()
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.text_encoder.requires_grad_(False)
        self.ref_transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(True)

        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_encoder.eval()
        self.ref_transformer.eval()
        self.vae.eval()
        return self

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
            max_length=self.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(self.device)
        return self.text_encoder(input_ids)[0]

    @torch.no_grad()
    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        videos = videos.to(self.device, dtype=self.vae.dtype)
        latent_dist = self.vae.encode(videos).latent_dist
        return latent_dist.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(self.device, dtype=self.vae.dtype)
        decoded = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        return decoded

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        transformer_config = self.transformer.config
        if not transformer_config.use_rotary_positional_embeddings:
            return None

        vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (
                num_frames + transformer_config.patch_size_t - 1
            ) // transformer_config.patch_size_t

        return get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

    def _predict_model_losses(
        self,
        latents: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        model: CogVideoXTransformer3DModel,
    ) -> torch.Tensor:
        transformer_config = model.config

        if transformer_config.patch_size_t is not None:
            ncopy = latents.shape[2] % transformer_config.patch_size_t
            if ncopy > 0:
                first_frame = latents[:, :, :1, :, :]
                latents = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latents], dim=2)
                noise = torch.cat([noise[:, :, :1, :, :].repeat(1, 1, ncopy, 1, 1), noise], dim=2)

        batch_size, _, num_frames, height, width = latents.shape
        prompt_embeddings = prompt_embeddings.to(dtype=latents.dtype)

        latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        noise = noise.permute(0, 2, 1, 3, 4).contiguous()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        rotary_emb = self.prepare_rotary_positional_embeddings(
            height=height * (2 ** (len(self.vae.config.block_out_channels) - 1)),
            width=width * (2 ** (len(self.vae.config.block_out_channels) - 1)),
            num_frames=num_frames,
            device=self.device,
        )

        predicted_noise = model(
            hidden_states=noisy_latents,
            encoder_hidden_states=prompt_embeddings,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]
        latent_pred = self.scheduler.get_velocity(predicted_noise, noisy_latents, timesteps)

        alphas_cumprod = torch.as_tensor(self.scheduler.alphas_cumprod, device=self.device)[
            timesteps
        ]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        return torch.mean((weights * (latent_pred - latents) ** 2).reshape(batch_size, -1), dim=1)

    def compute_dpo_loss(
        self,
        latents: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        dupfactor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = latents.shape[0]
        if batch_size % 2 != 0:
            raise ValueError(f"DPO expects an even latent batch size, got {batch_size}")
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        noise = torch.randn_like(latents)

        model_losses = self._predict_model_losses(
            latents, prompt_embeddings, timesteps, noise, self.transformer
        )
        with torch.no_grad():
            ref_losses = self._predict_model_losses(
                latents, prompt_embeddings, timesteps, noise, self.ref_transformer
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
