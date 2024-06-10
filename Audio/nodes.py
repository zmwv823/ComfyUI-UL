import os
import torch
import torchaudio
from einops import rearrange
import numpy as np
from .stable_audio_tools.models.factory import create_model_from_config
from .stable_audio_tools.models.utils import load_ckpt_state_dict
from .stable_audio_tools.inference.generation import generate_diffusion_cond
import folder_paths
import json
import time
from ..UL_common.common import get_device

# 获取当前文件的目录
current_directory = os.path.dirname(os.path.abspath(__file__))

# 加载模型
def load_model(device, model_path, dtype):
    # model_config = get_model_config()
    with open(os.path.join(current_directory, f'stable-audio-open-model-config\model_config.json'), encoding='utf-8') as f:
        model_config = json.load(f)
    model = create_model_from_config(model_config)
    model.load_state_dict(load_ckpt_state_dict(model_path))
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    
    model = model.to(device, dtype)
    return model,sample_rate,sample_size

def generate(model,prompt,seconds,seed,steps,cfg_scale,sample_size, sigma_min, sigma_max, sampler_type,device):
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": seconds
    }]
    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        device=device,
        seed=seed,
    )
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    return output

class UL_StableAudio:
    def __init__(self):
        self.initialized_model = None
        self.sample_rate=None
        self.sample_size=None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "prompt": ("STRING", 
                         {
                            "multiline": True, 
                            "default": "a woman's groaing by fucked_silly",
                            "dynamicPrompts": True
                          }),
                "seconds":("FLOAT", {"default": 47, "min": 1, "max": 10000,"step": 0.1}),
                "steps": ("INT", {"default": 16, "min": 1, "max": 10000}),

                "seed":  ("INT", {"default": 0, "min": 0, "max": np.iinfo(np.int32).max}), 

                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}), 
                "sigma_min": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "sigma_max": ("FLOAT", {"default": 200.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "sampler_type": (["dpmpp-3m-sde", "dpmpp-2m-sde", "k-dpm-fast", "k-lms", 'k-heun', 'k-dpmpp-2s-ancestral', 'k-dpm-2', 'k-dpm-adaptive'], {"default": "dpmpp-3m-sde"}),
                "fp16": ("BOOLEAN", {"default": True}),
                "save_name": ("STRING", {"multiline": False, "default": "stabe_audio"}),
                "force_cpu":("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "UL_StableAudio"
    CATEGORY = "ExtraModels/UL"
    TITLE = "UL StableAudio"
    
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)

    def UL_StableAudio(self, prompt,seconds,steps,seed, cfg_scale,  sigma_min, sigma_max, force_cpu, ckpt_name, fp16, sampler_type, save_name):

        if fp16 == True:
            dtype = torch.float16
        else:
            dtype = torch.float32
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        device = get_device()
        if force_cpu == False:
            device = device
        else:
            device = 'cpu'

        if self.initialized_model:
            self.initialized_model=self.initialized_model.to(device, dtype) #t5-base
        else:
            self.initialized_model,self.sample_rate,self.sample_size=load_model(device, ckpt_path, dtype)

        # 根据时长，计算size
        self.sample_size=int(self.sample_rate*seconds)
        
        output=generate(self.initialized_model,prompt,seconds,seed,steps,cfg_scale,self.sample_size, sigma_min, sigma_max, sampler_type, device)

        #生成后将模型转移到cpu，释放显存。
        self.initialized_model.to(torch.device('cpu'), dtype)

        comfy_output_dir = folder_paths.get_output_directory()
        # 添加文件名后缀
        audio_file = save_name
        now = time.strftime("%Y%m%d%H%M%S",time.localtime(time.time())) 
        audio_file = f"{audio_file}_{now}.wav"
        audio_path = os.path.join(comfy_output_dir, audio_file)

        torchaudio.save(audio_path, output, self.sample_rate)
        
        return ({
                "filename": audio_file,
                "subfolder": "",
                "type": "output",
                "prompt":prompt
                },)
        
NODE_CLASS_MAPPINGS = {
    "UL_StableAudio": UL_StableAudio
}