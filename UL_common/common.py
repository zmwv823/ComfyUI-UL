import torch
import numpy as np
import os
import shutil
import comfy.model_management as mm
import time

def is_module_imported(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True
    
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def get_device_by_name(device):
    if device == 'auto':
        try:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
                # device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = "mps"
                # device = torch.device("mps")
            elif torch.xpu.is_available():
                device = "xpu"
                # device = torch.device("xpu")
        except:
                raise AttributeError("What's your device(到底用什么设备跑的)？")
    # elif device == 'cuda':
    #     device = torch.device("cuda")
    # elif device == "mps":
    #     device = torch.device("mps")
    # elif device == "xpu":
    #     device = torch.device("xpu")
    print("\033[93mUse Device(使用设备):", device, "\033[0m")
    return device

def get_dtype_by_name(dtype):
    if dtype == 'auto':
        try:
            if mm.should_use_fp16():
                dtype = torch.float16
            elif mm.should_use_bf16():
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtypes manually.")
    elif dtype== "fp16":
         dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype = torch.float32
    print("\033[93mModel Precision(模型精度):", dtype, "\033[0m")
    return dtype

def copy_and_rename_file(source, destination, new_name):
    try:
        shutil.copy(source, destination)
        new_file_path = os.path.join(destination, new_name)
        os.rename(os.path.join(destination, os.path.basename(source)), new_file_path)
        print("\033[93mSuccess copy and rename file!-文件复制并重命名成功！\033[0m")
    except FileNotFoundError:
        print("\033[93mNo file!-文件不存在！\033[0m")
        
def is_folder_exist(folder_path):
    result = os.path.exists(folder_path)
    return result
    
def is_file_exists(file_path):
    result = os.path.exists(file_path)
    return result

def save_to_custom_folder_or_desktop(audio_path, save_to_desktop, save_to_custom_folder, save_name, custom_folder):
    now = time.strftime("%Y%m%d%H%M%S",time.localtime(time.time()))
    if save_to_desktop == True:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        new_name = f"{save_name}_{now}.wav"
        copy_and_rename_file(audio_path, desktop_path, new_name)
    if save_to_custom_folder == True:
        new_name = f"{save_name}_{now}.wav"
        copy_and_rename_file(audio_path, custom_folder, new_name)
    return

class UL_Text_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", 
                            {
                                "multiline": True, 
                                "default": "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。[laugh]本姑娘先做再说，哪怕做成屎一样，在慢慢改[laugh]，不要整天犹犹豫豫[uv_break]，一个粗糙的开始，就是最好的开始，什么也别管，先去做，然后你就会发现，用不了多久，你几十万就没了[laugh]。",
                                "dynamicPrompts": True
                            }),
                }
            }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("UL_Text_Input", )
    FUNCTION = "UL_Text_Input"
    CATEGORY = "ExtraModels"
    TITLE = "UL Text Input"

    def UL_Text_Input(self, text):
        self.text = text
        return (text, )
    
NODE_CLASS_MAPPINGS = {
    "UL_Text_Input": UL_Text_Input, 
}