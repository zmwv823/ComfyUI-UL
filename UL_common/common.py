import torch
import numpy as np
import os
import sys
import shutil
import comfy.model_management as mm

def is_module_imported(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True
    
def is_module_imported_simple(module_name):
    return module_name in sys.modules
    
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def get_device():
    # logging.info("[INFO] SDXLNodesLib get_device")
    device = "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.xpu.is_available():
        return "xpu"
    print(f"[INFO]device: {device}")
    return device

def get_device_by_name(device):
    if device == 'auto':
        try:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            elif torch.xpu.is_available():
                device = "xpu"
        except:
                raise AttributeError("What's your device(到底用什么设备跑的)？")
    print("\033[93mUse Device(使用设备):", device, "\033[0m")
    return device
    

def get_dtype(dtype):
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
