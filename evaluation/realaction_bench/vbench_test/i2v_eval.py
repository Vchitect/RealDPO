from vbench import VBench
from vbench2_beta_i2v.utils import init_submodules
import importlib
import os
device = 'cuda'
dimmension_list = [
    "i2v_subject",
    'subject_consistency',
    'background_consistency',
    'motion_smoothness',
    'dynamic_degree',
    'aesthetic_quality',
    'imaging_quality',
    'overall_consistency',
]

# dimmension_list = [
#     "i2v_background"
# ]

i2v_dims = ["i2v_subject", "i2v_background", "camera_motion"]

#dimmension_list = ["overall_consistency"]
def open_txt(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]
    return lines

def load_video_list(output_video_dir,image_dir,dimension,image_list=None,prompt_list=None):
    if dimension  in ['i2v_subject', 'i2v_background'] :
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
    

prompt_list = open_txt('./vbench_test/input/vbench_test/vbench_test/vbench_test.txt')
image_list = open_txt('./vbench_test/input/vbench_test/vbench_test/images.txt')

output_video_dir = ['./vbench_test/input/vbench_test/vbench_test/realdpo']
image_dir = "./vbench_test/input/images"
sub_module_dict = init_submodules(dimmension_list)
for dimension in dimmension_list:
    if dimension in i2v_dims:
        dimension_module = importlib.import_module(f'vbench2_beta_i2v.{dimension}')
    else:
        dimension_module = importlib.import_module(f'vbench.{dimension}')
    try:
        submodules_list = sub_module_dict[dimension]
    except:
        from vbench.utils import init_submodules
        submodules_list = init_submodules(dimmension_list)[dimension]

    evaluate_fnc = getattr(dimension_module,f'our_compute_{dimension}')
    video_list = load_video_list(output_video_dir[0],image_dir,dimension,image_list=image_list,prompt_list=prompt_list)
    results = evaluate_fnc(video_list,device,submodules_list)
    print(f'{dimension} avg: ',round(results[0]*100,2))
