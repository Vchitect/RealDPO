
import importlib
import os
import torch


# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import io
import pandas as pd
import argparse
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import json
import os
import shutil
from moviepy.video.io.VideoFileClip import VideoFileClip

import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_anyres_image,tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu

from transformers import AutoConfig

import cv2
import base64
import re
from PIL import Image
import numpy as np



device = 'cuda'

model_list  = [
    "Qwen/Qwen2-VL-7B-Instruct"
]

def extract_answer(output):
    try:
        pattern = r"Answer \d: \d+"
        matches = re.findall(pattern, output)
        print(matches)
        total_score = 0
        for m in matches[:5]:
            pattern = r"Answer \d: \d+"
            answer = m.split(':')[-1]
            pattern = "\d+"
            scores = re.findall(pattern, answer)
            total_score+=int(scores[-1])
        return total_score
        # print(tmp_matches[0])
        # pattern = "\d+"
        # matches = re.findall(pattern, tmp_matches[0])
        # return int(matches[-1])
    except:
        return 0

TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def init_model(model_name):
    if model_name == "Qwen/Qwen2-VL-7B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(model_name)
        model.eval().to(device)
    elif model_name =="THUDM/cogvlm2-llama3-caption":
        MODEL_PATH = "THUDM/cogvlm2-llama3-caption"

        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=TORCH_TYPE,
            trust_remote_code=True
        ).eval().to(DEVICE)
        return model,tokenizer
    else:
        mm_spatial_pool_mode = "average"
        mm_spatial_pool_stride = 2
        mm_newline_position = "no_token"
        for_get_frames_num = 32
        overwrite_config = {}
        overwrite_config["mm_spatial_pool_mode"] = mm_spatial_pool_mode
        overwrite_config["mm_spatial_pool_stride"] = mm_spatial_pool_stride
        overwrite_config["mm_newline_position"] = mm_newline_position
        cfg_pretrained = AutoConfig.from_pretrained(model_name)

        if "qwen" not in model_name.lower():
            if "224" in cfg_pretrained.mm_vision_tower:
                # suppose the length of text tokens is around 1000, from bo's report
                least_token_number = for_get_frames_num*(16//mm_spatial_pool_stride)**2 + 1000
            else:
                least_token_number = for_get_frames_num*(24//mm_spatial_pool_stride)**2 + 1000

            scaling_factor = math.ceil(least_token_number/4096)
            if scaling_factor >= 2:
                if "vicuna" in cfg_pretrained._name_or_path.lower():
                    print(float(scaling_factor))
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

        local_model_name = get_model_name_from_path(model_name)
        tokenizer, model, image_processor, _ = load_pretrained_model(model_name, None, local_model_name)
        
        processor = {
            "tokenizer": tokenizer,
            "image_processor": image_processor
        }

    return model,processor

def qwen2_vl_chat(model,processor,template,video_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {"type": "text", "text": template},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print(output_text)
    return output_text

def open_txt(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]
    return lines

def read_lines(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

def load_video_list(output_video_dir,image_dir,dimension,image_list=None,prompt_list=None):
    if dimension =='i2v_subject':
        video_list = []
        for v in os.listdir(output_video_dir):
            image_name = v.replace('.mp4','.png')
            image_path = os.path.join(image_dir,image_name)
            video_path = os.path.join(output_video_dir,v)
            video_list.append((image_path,video_path))
        return video_list
    elif dimension == 'overall_consistency':
        video_list = []
        for img,p in zip(image_list,prompt_list):
            img = os.path.join(image_dir,os.path.basename(img))
            video_name = os.path.basename(img).replace('.png','.mp4')
            image_path = os.path.join(image_dir,img)
            video_path = os.path.join(output_video_dir,video_name)
            video_list.append({
                "prompt":p,
                "video_list":[video_path]
            })
        return video_list
    else:
        video_list = []
        for v in os.listdir(output_video_dir):
            # image_name = v.replace('.mp4','.jpg')
            # image_path = os.path.join(image_dir,image_name)
            video_path = os.path.join(output_video_dir,v)
            video_list.append(video_path)
        return video_list

caption_list = open_txt("./vbench_test/input/vbench_test/vbench_test/vbench_test.txt")

method = "realdpo"
video_dir = f"./vbench_test/input/vbench_test/vbench_test/{method}"

model,processor = init_model("Qwen/Qwen2-VL-7B-Instruct")


q_type_list = ["Q1_en","Q2_en","Q3_en","Q4_en"]
for q_type in q_type_list:
    question_file = f"./vbench_test/{q_type}.txt"
    template = "".join(read_lines(question_file))
    if q_type == "Q2_en":
        template = "Given caption of this video: {caption}"+template
    total_score = []
    for f in os.listdir(video_dir):
        video_id = f.split('.')[0]
        caption = caption_list[int(video_id)]
        video_path = os.path.join(video_dir,f)
        if q_type == "Q2_en":
            output = qwen2_vl_chat(model,processor,template.format(caption=caption),video_path)
        else:
            output = qwen2_vl_chat(model,processor,template,video_path)       
            score = extract_answer(output[0])
        total_score.append(score)

    print(F"{q_type},total_score:",sum(total_score)/len(total_score)*2)

