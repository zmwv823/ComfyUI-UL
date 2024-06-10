import torch
import numpy as np
import os
import shutil
import comfy.model_management as mm

def is_module_imported(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True
    
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def get_device():
    # logging.info("[INFO] SDXLNodesLib get_device")
    device = "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    print(f"[INFO]device: {device}")
    return device

def get_dtype(dtype):
    if dtype == 'auto':
        try:
            if mm.should_use_fp16():
                   print("\033[93mfp16\033[0m")
                   dtype = torch.float16
            elif mm.should_use_bf16():
                   print("\033[93mbf16\033[0m")
                   dtype = torch.bfloat16
            else:
                   print("\033[93mfp32\033[0m")
                   dtype = torch.float32
        except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtypes manually.")
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
