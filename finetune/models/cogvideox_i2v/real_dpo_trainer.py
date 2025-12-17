import hashlib
import json
import logging
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import diffusers
import torch
import transformers
import wandb
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
)
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel
from finetune.constants import LOG_LEVEL, LOG_NAME
from finetune.datasets.utils import (
    load_images,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)
from finetune.datasets.i2v_dataset import RealDPOI2VDatasetWithResize
from finetune.schemas import Args, Components, State
from finetune.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
)
from typing_extensions import override


logger = get_logger(LOG_NAME, LOG_LEVEL)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,  # FP16 is Only Support for CogVideoX-2B
    "bf16": torch.bfloat16,
}

from ..cogvideox_i2v.lora_trainer import CogVideoXI2VLoraTrainer
from ..utils import register


class CogVideoXI2VRealDPOTrainer(CogVideoXI2VLoraTrainer):

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXImageToVideoPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        components.transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer")
        
        components.ref  = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer")

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline:
        pipe = CogVideoXImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )

        ref_pipe = CogVideoXImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=self.components.ref,
            scheduler=self.components.scheduler,
        )
        return pipe, ref_pipe

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {
            "encoded_win_videos": [], "prompt_embedding": [], "images": [],
            "all_encoded_lose_videos": []
        }

        for sample in samples:
            encoded_win_video = sample["encoded_win_video"]
            encoded_lose_video = sample["all_encoded_lose_videos"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]

            ret["encoded_win_videos"].append(encoded_win_video)
            ret["all_encoded_lose_videos"].append(encoded_lose_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["images"].append(image)

        ret["encoded_win_videos"] = torch.stack(ret["encoded_win_videos"])
        ret["all_encoded_lose_videos"] = torch.stack(ret["all_encoded_lose_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["images"] = torch.stack(ret["images"])

        return ret
    
    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        win_latent = batch["encoded_win_videos"]
        lose_latent = batch["all_encoded_lose_videos"]
        # batch["all_encoded_lose_videos"].shape
        # B,N,C,F,H,W
        images = batch["images"]

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        # Shape of images: [B, C, H, W]

        patch_size_t = self.state.transformer_config.patch_size_t

        # 1.5
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = win_latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=win_latent.dtype)

        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = images.unsqueeze(2)
        # Add noise to images
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        image_latent_dist = self.components.vae.encode(noisy_images.to(dtype=self.components.vae.dtype)).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps//2, (batch_size,), device=self.accelerator.device
        )
        timesteps = timesteps.long()

        # from [B, C, F, H, W] to [B, F, C, H, W]
        win_latent = win_latent.permute(0, 2, 1, 3, 4)

        # B N C F H W => N B F C H W
        lose_latent = lose_latent.permute(1,0, 3, 2, 4, 5)
        N,B,_,C,H,W = lose_latent.shape
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        assert (win_latent.shape[0], *win_latent.shape[2:]) == (image_latents.shape[0], *image_latents.shape[2:])

        # Padding image_latents to the same frame number as latent
        padding_shape = (win_latent.shape[0], win_latent.shape[1] - 1, *win_latent.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Add noise to latent
        noise = torch.randn_like(win_latent)
        win_latent_noisy = self.components.scheduler.add_noise(win_latent, noise, timesteps)

        # Concatenate latent and image_latents in the channel dimension
        win_latent_img_noisy = torch.cat([win_latent_noisy, image_latents], dim=2)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None if self.state.transformer_config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)
        )

        negative_sample_loss_dict = []
        loss = 0
        for n_idx in range(N):
            lose_latent_noisy = self.components.scheduler.add_noise(lose_latent[n_idx], noise, timesteps)
            lose_latent_img_noisy = torch.cat([lose_latent_noisy, image_latents], dim=2)

            lose_predicted_noise = self.components.transformer(
                hidden_states=lose_latent_img_noisy,
                encoder_hidden_states=prompt_embedding,
                timestep=timesteps,
                ofs=ofs_emb,
                image_rotary_emb=rotary_emb,
                return_dict=False,
            )[0]

            ref_lose_predicted_noise = self.components.ref(
                hidden_states=lose_latent_img_noisy,
                encoder_hidden_states=prompt_embedding,
                timestep=timesteps,
                ofs=ofs_emb,
                image_rotary_emb=rotary_emb,
                return_dict=False,
            )[0]
            win_predicted_noise = self.components.transformer(
                hidden_states=win_latent_img_noisy,
                encoder_hidden_states=prompt_embedding,
                timestep=timesteps,
                ofs=ofs_emb,
                image_rotary_emb=rotary_emb,
                return_dict=False,
            )[0]

            ref_win_predicted_noise = self.components.ref(
                hidden_states=win_latent_img_noisy,
                encoder_hidden_states=prompt_embedding,
                timestep=timesteps,
                ofs=ofs_emb,
                image_rotary_emb=rotary_emb,
                return_dict=False,
            )[0]


            # Denoise
            win_latent_pred = self.components.scheduler.get_velocity(win_predicted_noise, win_latent_noisy, timesteps)
            ref_win_latent_pred = self.components.scheduler.get_velocity(ref_win_predicted_noise, win_latent_noisy, timesteps)

            lose_latent_pred = self.components.scheduler.get_velocity(lose_predicted_noise, lose_latent_noisy, timesteps)
            ref_lose_latent_pred = self.components.scheduler.get_velocity(ref_lose_predicted_noise, lose_latent_noisy, timesteps)

            alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
            weights = 1 / (1 - alphas_cumprod)
            while len(weights.shape) < len(ref_lose_latent_pred.shape):
                weights = weights.unsqueeze(-1)
            win_loss = torch.mean((weights * (win_latent_pred - win_latent) ** 2).reshape(batch_size, -1), dim=1)
            ref_win_loss = torch.mean((weights * (ref_win_latent_pred - win_latent) ** 2).reshape(batch_size, -1), dim=1)
            
            inside_term = win_loss - ref_win_loss
            lose_loss = torch.mean((weights * (lose_latent_pred - lose_latent[n_idx]) ** 2).reshape(batch_size, -1), dim=1)
            ref_lose_loss = torch.mean((weights * (ref_lose_latent_pred - lose_latent[n_idx]) ** 2).reshape(batch_size, -1), dim=1)
            inside_term-= (lose_loss-ref_lose_loss)
            beta_dpo = 5000
    
            scale_term = -0.5 * beta_dpo
            inside_term = scale_term * inside_term
            dpo_loss = -1 * F.logsigmoid(inside_term).mean()
            self.accelerator.backward(dpo_loss)
            loss += dpo_loss.item()
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                grad_norm = self.components.transformer.get_global_grad_norm()
                # In some cases the grad norm may not return a float
                if torch.is_tensor(grad_norm):
                    grad_norm = grad_norm.item()
            else:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.components.transformer.parameters(), self.args.max_grad_norm
                )
                if torch.is_tensor(grad_norm):
                    grad_norm = grad_norm.item()
            negative_sample_loss_dict.append({
                "lose_loss": lose_loss.mean().item(),
                "ref_lose_loss":ref_lose_loss.mean().item(),
            })
            free_memory()

        loss /= N
        return loss,win_loss,ref_win_loss,negative_sample_loss_dict

    @override
    def prepare_dataset(self):
        logger.info("Initializing dataset and dataloader")

        self.dataset = RealDPOI2VDatasetWithResize(
            **(self.args.model_dump()),
            device=self.accelerator.device,
            max_num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            trainer=self,
        )

        # Prepare VAE and text encoder for encoding
        self.components.vae.requires_grad_(False)
        self.components.ref.requires_grad_(False)

        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )

        # Precompute latent for video and prompt embedding
        logger.info("Precomputing latent for video and prompt embedding ...")
        tmp_data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=1,
            num_workers=0,
            pin_memory=self.args.pin_memory,
        )
        tmp_data_loader = self.accelerator.prepare_data_loader(tmp_data_loader)
        for _ in tmp_data_loader:
            ...
        self.accelerator.wait_for_everyone()
        logger.info("Precomputing latent for video and prompt embedding ... Done")

        unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        unload_model(self.components.ref)

        free_memory()

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )
    
    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )
        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]

            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    loss,win_loss,ref_win_loss,negatvie_loss_dict = self.compute_loss(batch)
                    # accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.components.transformer.parameters(), self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.update_ref(global_step)
                    self.__maybe_save_checkpoint(global_step)

                logs["loss"] = loss
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                logs['win_loss'] = win_loss.detach().mean().item()
                logs['ref_win_loss'] = ref_win_loss.detach().mean().item()
                for n_idx in range(len(negatvie_loss_dict)):
                    logs[f'{n_idx}_lose_loss'] = negatvie_loss_dict[n_idx]['lose_loss']
                    logs[f'{n_idx}_ref_lose_loss'] = negatvie_loss_dict[n_idx]['ref_lose_loss']

                progress_bar.set_postfix(logs)

                # Maybe run validation
                should_run_validation = self.args.do_validation and global_step>0 and  global_step % self.args.validation_steps == 0
                if should_run_validation:
                    del loss
                    del win_loss
                    del ref_win_loss

                    free_memory()
                    self.validate(global_step)

                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

        accelerator.wait_for_everyone()
        self.__maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()
    
    def __get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")

    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(self.components, name, component.to(self.accelerator.device, dtype=dtype))

    def __move_components_to_cpu(self, unload_list: List[str] = []):
        unload_list = set(unload_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name in unload_list:
                    setattr(self.components, name, component.to("cpu"))

    def __prepare_saving_loading_hooks(self, transformer_lora_config):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        model = unwrap_model(self.accelerator, model)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                self.components.pipeline_cls.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            if not self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        transformer_ = unwrap_model(self.accelerator, model)
                    else:
                        raise ValueError(f"Unexpected save model: {unwrap_model(self.accelerator, model).__class__}")
            else:
                transformer_ = unwrap_model(self.accelerator, self.components.transformer).__class__.from_pretrained(
                    self.args.model_path, subfolder="transformer"
                )
                transformer_.add_adapter(transformer_lora_config)

            lora_state_dict = self.components.pipeline_cls.lora_state_dict(input_dir)
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def __maybe_save_checkpoint(self, global_step: int, must_save: bool = False):
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
            if must_save or global_step % self.args.checkpointing_steps == 0:
                # for training
                save_path = get_intermediate_ckpt_path(
                    checkpointing_limit=self.args.checkpointing_limit,
                    step=global_step,
                    output_dir=self.args.output_dir,
                )
                self.accelerator.save_state(save_path, safe_serialization=True)

    def validate(self, step: int) -> None:
        logger.info("Starting validation")

        accelerator = self.accelerator
        num_validation_samples = len(self.state.validation_prompts)

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.components.transformer.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        #####  Initialize pipeline  #####
        pipe,ref_pipe = self.initialize_pipeline()

        if self.state.using_deepspeed:
            # Can't using model_cpu_offload in deepspeed,
            # so we need to move all components in pipe to device
            # pipe.to(self.accelerator.device, dtype=self.state.weight_dtype)
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=["transformer","ref"])
        else:
            # if not using deepspeed, use model_cpu_offload to further reduce memory usage
            # Or use pipe.enable_sequential_cpu_offload() to further reduce memory usage
            pipe.enable_model_cpu_offload(device=self.accelerator.device)
            ref_pipe.enable_model_cpu_offload(device=self.accelerator.device)

            # Convert all model weights to training dtype
            # Note, this will change LoRA weights in self.components.transformer to training dtype, rather than keep them in fp32
            pipe = pipe.to(dtype=self.state.weight_dtype)
            ref_pipe = ref_pipe.to(dtype=self.state.weight_dtype)


        #################################

        all_processes_artifacts = []
        for i in range(num_validation_samples):
            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
                # Skip current validation on all processes but one
                if i % accelerator.num_processes != accelerator.process_index:
                    continue

            prompt = self.state.validation_prompts[i]
            image = self.state.validation_images[i]
            video = self.state.validation_videos[i]

            if image is not None:
                image = preprocess_image_with_resize(image, self.state.train_height, self.state.train_width)
                # Convert image tensor (C, H, W) to PIL images
                image = image.to(torch.uint8)
                image = image.permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(image)

            if video is not None:
                video = preprocess_video_with_resize(
                    video, self.state.train_frames, self.state.train_height, self.state.train_width
                )
                # Convert video tensor (F, C, H, W) to list of PIL images
                video = video.round().clamp(0, 255).to(torch.uint8)
                video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]

            logger.debug(
                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
                main_process_only=False,
            )
            validation_artifacts = self.validation_step({"prompt": prompt, "image": image, "video": video}, pipe)
            # ref_validation_artifacts = self.validation_step({"prompt": prompt, "image": image, "video": video}, ref_pipe)

            if (
                self.state.using_deepspeed
                and self.accelerator.deepspeed_plugin.zero_stage == 3
                and not accelerator.is_main_process
            ):
                continue

            prompt_filename = string_to_filename(prompt)[:25]
            # Calculate hash of reversed prompt as a unique identifier
            reversed_prompt = prompt[::-1]
            hash_suffix = hashlib.md5(reversed_prompt.encode()).hexdigest()[:5]

            artifacts = {
                "video": {"type": "video", "value": video},
            }

            for i, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                artifacts.update({f"artifact_{i}": {"type": artifact_type, "value": artifact_value}})

            logger.debug(
                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
                main_process_only=False,
            )

            for key, value in list(artifacts.items()):
                artifact_type = value["type"]
                artifact_value = value["value"]
                if artifact_type not in ["video"] or artifact_value is None:
                    continue
                flag = 'policy'
                if 'ref' in key:
                    flag = 'ref'
                extension = "png" if artifact_type == "image" else "mp4"
                filename = f"validation-{step}-{accelerator.process_index}-{flag}-{prompt_filename}-{hash_suffix}.{extension}"
                validation_path = self.args.output_dir / "validation_res"
                validation_path.mkdir(parents=True, exist_ok=True)
                filename = str(validation_path / filename)

                if artifact_type == "image":
                    logger.debug(f"Saving image to {filename}")
                    artifact_value.save(filename)
                    artifact_value = wandb.Image(filename)
                elif artifact_type == "video":
                    logger.debug(f"Saving video to {filename}")
                    export_to_video(artifact_value, filename, fps=self.args.gen_fps)
                    artifact_value = wandb.Video(filename, caption=prompt)

                all_processes_artifacts.append(artifact_value)

        all_artifacts = gather_object(all_processes_artifacts)

        if accelerator.is_main_process:
            tracker_key = "validation"
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
                    video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
                    tracker.log(
                        {
                            tracker_key: {"images": image_artifacts, "videos": video_artifacts},
                        },
                        step=step,
                    )

        ##########  Clean up  ##########
        if self.state.using_deepspeed:
            del pipe
            del ref_pipe
            # Unload models except those needed for training
            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
        else:
            pipe.remove_all_hooks()
            ref_pipe.remove_all_hooks()
            del pipe
            del ref_pipe
            # Load models except those not needed for training
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST)
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)
            self.components.ref.to(self.accelerator.device, dtype=self.state.weight_dtype)

            # Change trainable weights back to fp32 to keep with dtype after prepare the model
            cast_training_params([self.components.transformer], dtype=torch.float32)

        free_memory()
        accelerator.wait_for_everyone()
        ################################

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        torch.set_grad_enabled(True)
        self.components.transformer.train()

    def ema(self,current_state,history_state,ema_rate):
        ema_dict = {}
        for key,value in current_state.items():
            new_state = history_state[key] * ema_rate + (1-ema_rate) * value
            ema_dict[key] = new_state
        
        return ema_dict

    def update_ref(self,global_step):
        if global_step!=0 and global_step % 100==0:
            ema_rate = self.args.ema_rate
            ema_state = self.ema(unwrap_model(self.accelerator, self.components.transformer).state_dict(),self.components.ref.state_dict(),ema_rate)
            self.components.ref.load_state_dict(ema_state)
    


register("cogvideox-i2v", "real_dpo", CogVideoXI2VRealDPOTrainer)
