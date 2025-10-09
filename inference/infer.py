import argparse
from typing import Literal, Optional, List
import os
import torch
import numpy as np
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDPMScheduler,
    CogVideoXDDIMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video
from decord import VideoReader, cpu
import cv2
import numpy as np
from PIL import Image
import json
import copy 
import multiprocessing as mp
import time
from pathlib import Path
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CogVideoXInference:
    def __init__(self, model_path: str = "THUDM/CogVideoX-5b-I2V", 
                 finetune_ckpt_path: Optional[str] = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        """
        Initialize CogVideoX inference pipeline
        
        Args:
            model_path: Path to pretrained model
            finetune_ckpt_path: Path to fine-tuned checkpoint (optional)
            device: Device to run on ('cuda' or 'cpu')
            dtype: Data type for model weights
        """
        self.device = device
        self.dtype = dtype
        
        # Load base model
        logger.info(f"Loading base model from {model_path}")
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_path, 
            torch_dtype=self.dtype
        ).to(self.device)
        
        # Load fine-tuned model if provided
        if finetune_ckpt_path:
            self.load_finetuned_model(finetune_ckpt_path)
        else:
            self.finetune_pipe = None
    
    def load_finetuned_model(self, ckpt_path: str):
        """Load fine-tuned model weights"""
        logger.info(f"Loading fine-tuned weights from {ckpt_path}")
        finetune_ckpt = torch.load(ckpt_path)
        
        finetune_transformer = copy.deepcopy(self.pipe.transformer)
        finetune_transformer.load_state_dict(finetune_ckpt['module'])
        
        self.finetune_pipe = CogVideoXImageToVideoPipeline(
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            transformer=finetune_transformer,
            scheduler=self.pipe.scheduler,
            tokenizer=self.pipe.tokenizer,
        ).to(self.device)
    
    def preprocess_image(self, image: Image.Image, target_width: int = 720, target_height: int = 480) -> Image.Image:
        """
        Preprocess input image to match model requirements
        
        Args:
            image: Input PIL Image
            target_width: Target width for output
            target_height: Target height for output
            
        Returns:
            Processed PIL Image
        """
        w, h = image.size
        scale = min(w / target_width, h / target_height)
        crop_width = int(target_width * scale)
        crop_height = int(target_height * scale)

        x_center, y_center = w // 2, h // 2
        x1 = max(x_center - crop_width // 2, 0)
        y1 = max(y_center - crop_height // 2, 0)
        x2 = min(x1 + crop_width, w)
        y2 = min(y1 + crop_height, h)
        
        cropped_image = image.crop((x1, y1, x2, y2))
        resized_image = cropped_image.resize((target_width, target_height))
        return resized_image
    
    def infer(
        self,
        pipeline,
        caption: str,
        image: Image.Image,
        height: int = 480,
        width: int = 720,
        num_videos_per_prompt: int = 1,
        num_inference_steps: int = 50,
        num_frames: int = 49,
        use_dynamic_cfg: bool = False,
        guidance_scale: float = 6.0,
    ) -> List[Image.Image]:
        """
        Run inference to generate video from image and caption
        
        Args:
            pipeline: The pipeline to use for inference
            caption: Text prompt describing the video
            image: Input image to condition on
            height: Output video height
            width: Output video width
            num_videos_per_prompt: Number of videos to generate per prompt
            num_inference_steps: Number of denoising steps
            num_frames: Number of frames in output video
            use_dynamic_cfg: Whether to use dynamic classifier-free guidance
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            List of PIL Images representing video frames
        """
        try:
            video = pipeline(
                height=height,
                width=width,
                prompt=caption,
                image=copy.deepcopy(image),
                num_videos_per_prompt=num_videos_per_prompt,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                use_dynamic_cfg=use_dynamic_cfg,
                guidance_scale=guidance_scale,
            ).frames[0]
            return video
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise
    
    def generate_comparison_video(
        self,
        input_path: str,
        caption: str,
        output_path: str,
        use_caption_generation: bool = False,
        **inference_kwargs
    ) -> None:
        """
        Generate comparison video between base and fine-tuned models
        
        Args:
            input_path: Path to input image
            caption: Text prompt describing the video
            output_path: Path to save output video
            use_caption_generation: Whether to generate caption automatically
            inference_kwargs: Additional arguments for inference
        """
        # Load and preprocess image
        image = Image.open(input_path)
        processed_image = self.preprocess_image(image)
        
        # Generate caption if needed
        if use_caption_generation:
            caption = self.generate_caption(processed_image)
            logger.info(f"Generated caption: {caption}")
        
        # Run inference with both models
        logger.info("Running inference with base model...")
        base_video = self.infer(self.pipe, caption, processed_image, **inference_kwargs)
        
        if self.finetune_pipe:
            logger.info("Running inference with fine-tuned model...")
            ft_video = self.infer(self.finetune_pipe, caption, processed_image, **inference_kwargs)
        else:
            ft_video = None
        
        # Save results
        self.save_results(caption, base_video, ft_video, output_path)
    
    def generate_caption(self, image: Image.Image) -> str:
        """
        Generate caption for input image (placeholder - implement with your captioning model)
        
        Args:
            image: Input image to caption
            
        Returns:
            Generated caption text
        """
        # TODO: Implement with BLIP-2 or other captioning model
        return "A descriptive caption of the image"
    
    def save_results(
        self,
        caption: str,
        base_video: List[Image.Image],
        ft_video: Optional[List[Image.Image]],
        output_path: str
    ) -> None:
        """
        Save comparison video and metadata
        
        Args:
            caption: The prompt used for generation
            base_video: Frames from base model
            ft_video: Frames from fine-tuned model (optional)
            output_path: Path to save output
        """
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison video if both models available
        if ft_video is not None:
            self.concat_and_save_videos(caption, base_video, ft_video, output_path)
        else:
            logger.info("Saving only base model output")
            export_to_video(base_video, output_path, fps=8)
        
        # Save metadata
        metadata_path = os.path.splitext(output_path)[0] + ".json"
        with open(metadata_path, "w") as f:
            json.dump({
                "caption": caption,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "CogVideoX",
                "has_finetuned": ft_video is not None
            }, f, indent=2)
    
    def concat_and_save_videos(
        self,
        caption: str,
        base_video: List[Image.Image],
        ft_video: List[Image.Image],
        output_path: str
    ) -> None:
        """
        Concatenate videos side by side and save
        
        Args:
            caption: The prompt used for generation
            base_video: Frames from base model
            ft_video: Frames from fine-tuned model
            output_path: Path to save output
        """
        logger.info(f"Saving comparison video to {output_path}")
        concat_video = []
        
        for v1, v2 in zip(base_video, ft_video):
            width, height = v1.size
            new_image = Image.new("RGB", (width * 2, height))
            new_image.paste(v1, (0, 0))
            new_image.paste(v2, (width, 0))
            concat_video.append(new_image)
        
        export_to_video(concat_video, output_path, fps=8)

def main():
    parser = argparse.ArgumentParser(description="CogVideoX Inference Script")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output/output.mp4", help="Output video path")
    parser.add_argument("--caption", type=str, help="Text prompt for video generation")
    parser.add_argument("--generate-caption", action="store_true", help="Generate caption automatically")
    parser.add_argument("--finetune-ckpt", type=str, help="Path to fine-tuned checkpoint")
    parser.add_argument("--num-frames", type=int, default=49, help="Number of frames in output video")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=6.0, help="Guidance scale")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    inference = CogVideoXInference(
        finetune_ckpt_path=args.finetune_ckpt,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Run inference
    inference.generate_comparison_video(
        input_path=args.input,
        caption=args.caption,
        output_path=args.output,
        use_caption_generation=args.generate_caption,
        num_frames=args.num_frames,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale
    )

if __name__ == "__main__":
    # Example usage:
    # python script.py --input input/test.jpg --output output/comparison.mp4 --caption "A boy running in nature"
    
    main()