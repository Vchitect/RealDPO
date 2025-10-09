import argparse
from typing import Literal
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
from decord import VideoReader,cpu
import cv2
import numpy as np
import PIL
from PIL import Image
import json
import copy 
from tqdm import tqdm

import multiprocessing as mp
import time

def open_txt(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]
    return lines

def worker(gpu,world_size):
    model_path = "THUDM/CogVideoX-5b-I2V"
    dtype = torch.bfloat16

    videos = open_txt("./demo_data/train_data/video.txt")
    prompts = open_txt("./demo_data/train_data/prompt.txt")
    data_dir = "./demo_data/train_data/"
    save_dir = "./demo_data/train_data/negative_samples"
    sampe_num = 3

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype,map_location='cpu')
    pipe = pipe.to(f'cuda:{gpu}')
    for i in tqdm(range(len(videos))):
        if (i % world_size) !=gpu:
            continue
        caption = prompts[i]
        file_name_prefix = os.path.basename(videos[i]).split('.')[0]
        vr = VideoReader(os.path.join(data_dir,videos[i]))
        frame = vr[0]
        frame = frame.asnumpy()
        image = Image.fromarray(frame)
        #img_path = os.path.join(data_dir,"first_frames",file_name_prefix+".png")
        #image = Image.open(img_path)

        os.makedirs(os.path.join(save_dir,file_name_prefix),exist_ok=True)
        for j in range(sampe_num):
            output_path = os.path.join(save_dir,file_name_prefix,str(j)+'.mp4')
            if os.path.exists(output_path):
                print("skip -> ",output_path)
                continue
            video = pipe(
                height=480,
                width=720,
                prompt=caption,
                image=copy.deepcopy(image),
                # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
                num_videos_per_prompt=1,  # Number of videos to generate per pc c crompt
                num_inference_steps=50,  # Number of inference steps
                num_frames=49,  # Number of frames to generate
                use_dynamic_cfg=False,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
                guidance_scale=6.0,
            ).frames[0]
            export_to_video(video, output_path, fps=8)


if __name__=='__main__':
    mp.set_start_method("spawn",force=True)
    world_size = 4
    processes = []
    for i in range(world_size):
        p = mp.Process(target=worker, args=(i,world_size))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    print("All workers are done.")