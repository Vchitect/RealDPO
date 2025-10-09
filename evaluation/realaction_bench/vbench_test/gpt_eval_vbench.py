
import importlib
import os
import torch


from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
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
# import openai
import re
from tqdm import tqdm

from PIL import Image



import numpy as np



device = 'cuda'

# dimmension_list = [
#     "temporal_flickering"
# ]

model_list  = [
    "Qwen/Qwen2-VL-7B-Instruct"
]

def extract_answer(output):
    try:
        # pattern = "\d+分"
        pattern = r"Answer \d: \d+"
        matches = re.findall(pattern, output)
        # print(matches)
        total_score = 0
        for m in matches[:5]:
            pattern = r"Answer \d: \d+"
            answer = m.split(':')[-1]
            pattern = "\d+"
            scores = re.findall(pattern, answer)
            total_score+=int(scores[-1])
        return total_score
    except:
        return 0

TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def init_model(model_name):
    if model_name == "Qwen/Qwen2-VL-7B-Instruct":
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name,torch_dtype=torch.bfloat16)
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
        return_tensors='pt'
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

def llava_video_chat(model,processor,template,video_path):
    mm_spatial_pool_mode = "average"
    mm_spatial_pool_stride = 2
    mm_newline_position = "no_token"
    conv_mode = "vicuna_v1"
    force_sample = False
    for_get_frames_num = 32
    import pdb;pdb.set_trace()
    def load_video(video_path):
        vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i/fps for i in frame_idx]
        if len(frame_idx) > for_get_frames_num or force_sample:
            sample_fps = for_get_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames,frame_time,video_time

    if getattr(model.config, "add_time_instruction", None) is not None:
        add_time_instruction = model.config.add_time_instruction
    else:
        add_time_instruction = False



    sample_set = {}
    question = template
    sample_set["Q"] = question
    sample_set["video_name"] = video_path
    image_processor = processor["image_processor"]
    tokenizer = processor["tokenizer"]

    # Check if the video exists
    if os.path.exists(video_path):
        video,frame_time,video_time = load_video(video_path)
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
        video = [video]

    # try:
    # Run inference on the video and add the output to the list
    qs = question
    if add_time_instruction:
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        qs = f'{time_instruciton}\n{qs}'
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    if tokenizer.pad_token_id is None:
        if "qwen" in tokenizer.name_or_path.lower():
            print("Setting pad token to bos token for qwen model.")
            tokenizer.pad_token_id = 151643
            
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    cur_prompt = question
    system_error = ""

    with torch.inference_mode():
        output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1,num_beams=1,use_cache=True, stopping_criteria=[stopping_criteria])
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # import pdb;pdb.set_trace()
    return outputs

def cogvlm2_video_chat(model,tokenizer,template,video_path):
    def load_video(video_data, strategy='chat'):
        bridge.set_bridge('torch')
        # mp4_stream = video_data
        num_frames = 24
        decord_vr = VideoReader(video_path, ctx=cpu(0))

        frame_id_list = None
        total_frames = len(decord_vr)
        if strategy == 'base':
            clip_end_sec = 60
            clip_start_sec = 0
            start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
            end_frame = min(total_frames,
                            int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
            frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        elif strategy == 'chat':
            timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
            timestamps = [i[0] for i in timestamps]
            max_second = round(max(timestamps)) + 1
            frame_id_list = []
            for second in range(max_second):
                closest_num = min(timestamps, key=lambda x: abs(x - second))
                index = timestamps.index(closest_num)
                frame_id_list.append(index)
                if len(frame_id_list) >= num_frames:
                    break

        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)
        return video_data

    def predict(prompt, video_data, temperature):
        strategy = 'chat'

        video = load_video(video_data, strategy=strategy)

        history = []
        query = prompt
        inputs = model.build_conversation_input_ids(
            tokenizer=tokenizer,
            query=query,
            images=[video],
            history=history,
            template_version=strategy
        )
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": False,
            "top_p": 0.1,
            "temperature": temperature,
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
    # print(template)
    output = predict(template,load_video(video_path),1.0)
    # print(output)
    return output

#dimmension_list = ["overall_consistency"]
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
            video_path = os.path.join(output_video_dir,v)
            video_list.append(video_path)
        return video_list


method = "realdpo"

video_dir = "/root/data/vbench_i2v_realdpo_v2"
model,processor = init_model("Qwen/Qwen2-VL-7B-Instruct")


q_type_list = ["Q3_en"]
for q_type in q_type_list:
    question_file = f"./vbench_test/{q_type}.txt"
    template = "".join(read_lines(question_file))
    if q_type == "Q2_en":
        template = "Given caption of this video: {caption}"+template
    total_score = []
    for f in tqdm(os.listdir(video_dir)):
        caption = f.split(".")[0]
        video_path = os.path.join(video_dir,f)
        if q_type == "Q2_en":
            output = qwen2_vl_chat(model,processor,template.format(caption=caption),video_path)
        else:
            output = qwen2_vl_chat(model,processor,template,video_path)
        score = extract_answer(output[0])
        total_score.append(score)

    print(F"{q_type},total_score:",sum(total_score)/len(total_score)*2)