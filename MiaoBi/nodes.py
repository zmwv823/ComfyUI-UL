import os
import torch
from comfy.utils import load_torch_file, ProgressBar
from comfy.lora import model_lora_keys_unet
from comfy.sd import load_lora_for_models
import folder_paths
from diffusers import DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, UniPCMultistepScheduler, DDIMScheduler, TCDScheduler, LCMScheduler
from transformers import AutoTokenizer, CLIPTokenizer, CLIPTextModel
#这里可以直接导入插件目录下的社区长文本带权重区分包。
from ..UL_common.common import pil2tensor, is_module_imported, get_device_by_name, get_dtype_by_name
from huggingface_hub import hf_hub_download, snapshot_download

current_directory = os.path.dirname(os.path.abspath(__file__))
#获取当前py脚本路径，绝对路径
scheduler_list = [
    "Euler a",
    "DPM++ 2M",
    "DPM++ 2M Karras",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE Karras",
    "DPM++ SDE",
    "DPM++ SDE Karras",
    "DPM2",
    "DPM2 Karras",
    "DPM2 a",
    "DPM2 a Karras",
    "Euler",
    "DDIM",
    "Heun",
    "LMS",
    "LMS Karras",
    "UniPC",
    "TCD",
    "LCM",
    "diffusion",
]

# def get_sheduler(name):
# sdxl专用scheduler定义，通过传递过来的匹配值选择哪个调度器。
#     scheduler = False
#     if name == "DPM++ 2M":
#         scheduler = DPMSolverMultistepScheduler()
#     elif name == "DPM++ 2M Karras":
#         scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True)
#     elif name == "DPM++ 2M SDE":
#         scheduler = DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++")
#     elif name == "DPM++ 2M SDE Karras":
#         scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
#     elif name == "DPM++ SDE":
#         scheduler = DPMSolverSinglestepScheduler() 
#     elif name == "DPM++ SDE Karras":
#         scheduler = DPMSolverSinglestepScheduler(use_karras_sigmas=True)
#     elif name == "DPM2":
#         scheduler = KDPM2DiscreteScheduler() 
#     elif name == "DPM2 Karras":
#         scheduler = KDPM2DiscreteScheduler(use_karras_sigmas=True)
#     elif name == "DPM2 a":
#         scheduler = KDPM2AncestralDiscreteScheduler() 
#     elif name == "DPM2 a Karras":
#         scheduler = KDPM2AncestralDiscreteScheduler(use_karras_sigmas=True)
#     elif name == "Euler":
#         scheduler = EulerDiscreteScheduler() 
#     elif name == "Euler a":
#         scheduler = EulerAncestralDiscreteScheduler() 
#     elif name == "Heun":
#         scheduler = HeunDiscreteScheduler() 
#     elif name == "LMS":
#         scheduler = LMSDiscreteScheduler() 
#     elif name == "LMS Karras":
#         scheduler = LMSDiscreteScheduler(use_karras_sigmas=True)
#     elif name == "UniPC":
#         scheduler = UniPCMultistepScheduler() 
#     elif name == "DDIM":
#         scheduler = DDIMScheduler() 
#     elif name == "TCD":
#         scheduler = TCDScheduler() 
#     elif name == "LCM":
#         scheduler = LCMScheduler() 
#     return scheduler
#def INPUT_TYPES(s):return {required": {"scheduler": (scheduler_list,),}}RETURN_TYPES = ("xxx",) RETURN_NAMES = ("xxx",) FUNCTION = "xxx" CATEGORY = "xxx"
#def xxx(self, scheduler="", )
# 获取INPUT_TYPES(s)中输入的值，传递到32行
#scheduler_apply = get_sheduler(scheduler)
# sdxl专属scheduler定义，获取32行传递过来的值
#pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, scheduler=scheduler_apply, torch_dtype=torch.float16).to("cuda")

# base_path = os.path.dirname(os.path.realpath(__file__))获取插件根目录
# models_dir = os.path.join(base_path, "models")插件根目录下的models目录
# os.makedirs(models_dir, exist_ok=True)如果没有models目录则创建，有则跳过
ResAdapter_dir = os.path.join(folder_paths.models_dir, "upscale_models", "ResAdapter")
ResAdapter = ["resadapter_v1_sd1.5", "resadapter_v1_sd1.5_interpolation", "resadapter_v1_sd1.5_extrapolation", "resadapter_v2_sd1.5",
              "resadapter_v1_sdxl", "resadapter_v1_sdxl_interpolation", "resadapter_v1_sdxl_extrapolation", "resadapter_v2_sdxl"]
def get_model_list(ResAdapter_dir):
    return [f for f in os.listdir(ResAdapter_dir)]

class UL_MiaoBi_ResAdapterLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "resadapter_name": (ResAdapter,),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "UL_MiaoBi_ResAdapterLoader"
    CATEGORY = "ExtraModels/UL MiaoBi"
    TITLE = "UL MiaoBi ResAdapter Loader"

    def UL_MiaoBi_ResAdapterLoader(self, model, clip, resadapter_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        for MODEL in ResAdapter:
            #枚举ResAdapter列表中的名字MODEL，下面来判断是否有当前选择名字的文件-55，没有的话自动下载，MODEL是返回的名字值。
           if not os.path.exists(os.path.join(ResAdapter_dir, MODEL)):
              print("\033[93m第一次使用会自动下载全部模型\n\033[0m")
              hf_hub_download(repo_id="jiaxiangc/res-adapter", subfolder=f"{MODEL}", filename="pytorch_lora_weights.safetensors", local_dir=ResAdapter_dir,)
              if "interpolation" not in MODEL:
                 hf_hub_download(repo_id="jiaxiangc/res-adapter", subfolder=MODEL, filename="diffusion_pytorch_model.safetensors", local_dir=ResAdapter_dir,)
              #else:
                  #pass

        # load lora...
        lora_path = os.path.join(ResAdapter_dir, f"{resadapter_name}/pytorch_lora_weights.safetensors")
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        # load norm...
        norm_path = os.path.join(ResAdapter_dir, f"{resadapter_name}/diffusion_pytorch_model.safetensors")
        if os.path.exists(norm_path):
            key_map = {}
            key_map = model_lora_keys_unet(model.model, key_map)
            norm = load_torch_file(norm_path, safe_load=True)
            mapping_norm = {}

            for key in norm.keys():
                if ".weight" in key:
                    key_name_in_ori_sd = key_map[key.replace(".weight", "")]
                    mapping_norm[key_name_in_ori_sd] = norm[key]
                elif ".bias" in key:
                    key_name_in_ori_sd = key_map[key.replace(".bias", "")]
                    mapping_norm[key_name_in_ori_sd.replace(".weight", ".bias")] = norm[key]
                else:
                    print("### resadapter: unexpected key", key)
                    mapping_norm[key] = norm[key]

            for k in mapping_norm.keys():
                if k not in model.model.state_dict():
                    print("### resadapter: missing key:", k)
            model.model.load_state_dict(mapping_norm, strict=False)
        else:
            print("\033[93mFor resolution interpolation, we do not need normalization temporally.\033[0m")

        model_lora, clip_lora = load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

class UL_MiaoBi_Singlefile_SD15:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                #os.listdir列出clip文件夹下所有文件及文件夹，输出选择的项目名字非路径，类型为字符串string，使用utf-8开启非英文支持。
                "clip":(os.listdir(os.path.join(folder_paths.models_dir, "clip")), 'utf-8', ),
                #从configs获取所有文件,默认输出字符串值v1-inference_fp16.yaml，定义默认加载哪个yaml文件。
                "config_name": (folder_paths.get_filename_list("configs"),  {"default": "v1-inference_fp16.yaml"}),
                "MiaoBi_Loader": ("MiaoBi_Loader", {"forceInput": True}),
                "Language": (["中文(简体、繁体)","English"],),
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "动漫风格的风景画，有山脉、湖泊，也有繁华的小镇子，色彩鲜艳，光影效果明显，高清，高细节，高分辨率。"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "低质量，低分辨率。"
                }),
                "num_inference_steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 99,
                    "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1,
                    "max": 99,
                    "step": 0.1
                }),
                "eta": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 0,
                    "max": 9999999
                }),
                "height": ("INT", {
                    "default": 768,
                    "min": 0,
                    "max": 9999999
                }),
                "scheduler": (scheduler_list,{"default":"Euler"}),
                "clip_dtype": (["auto","fp16","bf16","fp32"],{"default":"auto"}),
                "model_dtype": (["auto","fp16","bf16","fp32"],{"default":"auto"}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "UL_MiaoBi_Singlefile_SD15"
    CATEGORY = "ExtraModels/UL MiaoBi"
    TITLE = "UL MiaoBi Singlefile SD15"
    DESCRIPTION = "text_encoder--clip(text2image)--clip_dtype--c_dtype,diffusion--model--model_dtype--m_dtype."
    
    def UL_MiaoBi_Singlefile_SD15(
                 self, 
                 MiaoBi_Loader,
                 ckpt_name, 
                 clip, 
                 config_name, 
                 device,
                 positive_prompt, 
                 negative_prompt, 
                 guidance_scale, 
                 width, 
                 height, 
                 num_inference_steps, 
                 seed,
                 scheduler,
                 Language,
                 clip_dtype,
                 model_dtype,
                 eta):
        
        device = get_device_by_name(device)
        c_dtype = get_dtype_by_name(clip_dtype)
        m_dtype = get_dtype_by_name(model_dtype)
        
        #指定生成设备类型，gpu或者cpu。
        MiaoBi_Loader = MiaoBi_Loader.split("|")
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        #模型文件夹位置，大模型-clip(text_encoder)-tokenizer   
        MiaoBi_tokenizer_path = os.path.join(os.path.dirname(current_directory), "MiaoBi", "MiaoBi_tokenizer")
        #os.path.dirname获取上一层目录，套两层则是读取上上层目录,例:os.path.dirname(os.path.dirname(current_directory))。读取插件下的模型路径。
        original_config_file = os.path.join(folder_paths.models_dir, "configs", config_name)
        #彻底离线，使用comfyui--models--configs下的yaml配置文件,拼接选择的文件返回的文件名，组成完整路径名。
        generator = torch.manual_seed(seed)
        #CPU种子生成器，添加.cuda可用gpu生成。Seed generator by cpu，"torch.manual_seed(seed)" by cpu,"torch.cuda.manual_seed(seed)"by gpu
        #scheduler_apply = get_sheduler(scheduler)
        num_hidden_layers = int(MiaoBi_Loader[1])
        if num_hidden_layers == "-1":
        #    使用模型的全部层数
           text_encoder = CLIPTextModel.from_pretrained(
               os.path.join(folder_paths.models_dir,"clip", clip, ), 
               torch_dtype = c_dtype,
            #    torch_dtype=torch.float16
               ).to(device)
        elif num_hidden_layers != "-1":
           num_hidden_layers = num_hidden_layers + 1
           text_encoder = CLIPTextModel.from_pretrained(
               os.path.join(folder_paths.models_dir,"clip", clip, ), 
               num_hidden_layers = 12 + num_hidden_layers, 
               torch_dtype = c_dtype,
            #    使用模型的前n层
            #    torch_dtype=torch.float16
               ).to(device)
        
        if not is_module_imported('StableDiffusionPipeline'):
            from diffusers import StableDiffusionPipeline
        custom_pipeline = os.path.join(current_directory, "MiaoBi_Pipelines", "lpw_stable_diffusion.py")   
        if Language == "中文(简体、繁体)":
           tokenizer = AutoTokenizer.from_pretrained(MiaoBi_tokenizer_path, trust_remote_code=True)
           pipe = StableDiffusionPipeline.from_single_file(#原为StableDiffusionPipeline.from_single_file，没77限制但是没权重控制
               ckpt_path, 
               text_encoder=text_encoder,
               #custom_pipeline='lpw-stable-diffusion_xl/lpw_stable_diffusion',
               # 获取的是"lpw-stable-diffusion_xl.py/lpw_stable_diffusion.py"文件，可以自定义到插件目录下。解除77个tokens限制,通过使用社区提供的"lpw_stable_diffusion"，我们可以解锁77个tokens限制，并通过更长的prompt生成高质量图像。
               original_config_file=original_config_file,
               #彻底离线，使用comfyui--models--configs文件夹内的yaml。Totally offline by reading yaml file from "comfyui--models--configs".
               load_safety_checker = False,
               #是否加载安全检测
               requires_safety_checker = False,
               #禁用安全检测disable safety checker for nsfw output
               safety_checker=None, 
               # #禁用安全检测disable safety checker for nsfw output
               tokenizer=tokenizer,
               torch_dtype=m_dtype,
            #    torch_dtype=torch.float16
               ).to(device)
        elif Language == "English":
        # os.path.dirname获取上一层目录，套两层则是读取上上层目录,例:os.path.dirname(os.path.dirname(current_directory))。读取插件下的模型路径。
           pipe = StableDiffusionPipeline.from_single_file(
               ckpt_path, 
               custom_pipeline=custom_pipeline,
            # 获取的是"lpw-stable-diffusion_xl.py/lpw_stable_diffusion.py"文件，可以自定义到插件目录下。解除77个tokens限制,通过使用社区提供的"lpw_stable_diffusion"，我们可以解锁77个tokens限制，并通过更长的prompt生成高质量图像。这里直接使用本地管线，而不是custom方法。
               text_encoder=text_encoder,
               original_config_file=original_config_file,
               #彻底离线，使用comfyui--models--configs文件夹内的yaml。Totally offline by reading yaml file from "comfyui--models--configs".
               load_safety_checker = False,
               #是否加载安全检测
               requires_safety_checker = False,
               #禁用安全检测disable safety checker for nsfw output
               safety_checker=None, 
               # 禁用安全检测disable safety checker for nsfw output
               tokenizer=CLIPTokenizer.from_pretrained(os.path.join(folder_paths.models_dir,"clip", clip_name, )),
               torch_dtype=m_dtype,
            #    torch_dtype=torch.float16
               ).to(device)
        
        if scheduler == "UniPC":
            scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "DDIM":
            scheduler = DDIMScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "Euler a":
            scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "DPM++ 2M":
            scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "DPM++ 2M Karras":
            scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        elif scheduler== "DPM++ 2M SDE":
            scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++")
        elif scheduler == "DPM++ 2M SDE Karras":
            scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
        elif scheduler == "DPM++ SDE":
            scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, lower_order_final=True) 
        elif scheduler == "DPM++ SDE Karras":
            scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, lower_order_final=True)
        elif scheduler == "DPM2":
            scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "DPM2 Karras":
            scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        elif scheduler == "DPM2 a":
            scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "DPM2 a Karras":
            scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        elif scheduler == "Euler":
            scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "Heun":
            scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "LMS":
            scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "LMS Karras":
            scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        elif scheduler == "TCD":
            scheduler = TCDScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "LCM":
            scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "diffusion":
            scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config) 
        pipe.scheduler = scheduler

        pipe.enable_vae_tiling()
        pipe.enable_xformers_memory_efficient_attention()
        #pipe.enable_model_cpu_offload()#如果需要启用，去除上面pipe中的.to(device)，启用可降低显存占用，但处理时间增加。if "pipe.enable_model_cpu_offload()" needed ,delete ".to(device)" in line 101.With enabled,less vram but need little more time for process.
        # pipe.enable_sequential_cpu_offload() # my graphics card VRAM is very low
        #pipe.enable_attention_slicing()
        # 节省显存
        LoRA_path1 = folder_paths.get_full_path("loras", MiaoBi_Loader[2])
        LoRA_path2 = folder_paths.get_full_path("loras", MiaoBi_Loader[3])
        LoRA_path3 = folder_paths.get_full_path("loras", MiaoBi_Loader[4])
        LoRA_switch1 = MiaoBi_Loader[8]
        LoRA_switch2 = MiaoBi_Loader[9]
        LoRA_switch3 = MiaoBi_Loader[10]
        lora_weight1 = float(MiaoBi_Loader[5])
        lora_weight2 = float(MiaoBi_Loader[6])
        lora_weight3 = float(MiaoBi_Loader[7])
        multi_loras_weight = float(MiaoBi_Loader[11])
        
        if (MiaoBi_Loader[0]) == 'True' and (MiaoBi_Loader[8]) == 'False' and (MiaoBi_Loader[9]) == 'False' and (MiaoBi_Loader[10]) == 'False':
            LoRA_switch1 = False
            LoRA_switch2 = False
            LoRA_switch3 = False
            if not is_module_imported('apply_hidiffusion'):
                from .MiaoBi_hidiffusion.hidiffusion import apply_hidiffusion
            apply_hidiffusion(pipe, True, True, "SD1.5")
        elif (MiaoBi_Loader[0]) == 'False':
            pass
        else:
            raise Exception(f"Hidiffusion is incompatible with LoRA for now(Hidiffusion高清图生成目前不支持LoRA).")
        
        if LoRA_switch1 == 'True' and LoRA_switch2 == 'True' and LoRA_switch3 == 'True':
            pipe.load_lora_weights(LoRA_path1,adapter_name="lora1")
            pipe.load_lora_weights(LoRA_path2,adapter_name="lora2")
            pipe.load_lora_weights(LoRA_path3,adapter_name="lora3")
            pipe.set_adapters(["lora1", "lora2", "lora3"], adapter_weights=[lora_weight1, lora_weight2, lora_weight3])#1-2-3,应用3个LoRA，使用peft方法，降速明显。
        elif LoRA_switch1 == 'True' and LoRA_switch2 == 'True' and LoRA_switch3 == 'False':
            pipe.load_lora_weights(LoRA_path1,adapter_name="lora1")
            pipe.load_lora_weights(LoRA_path2,adapter_name="lora2")
            pipe.set_adapters(["lora1", "lora2"], adapter_weights=[lora_weight1, lora_weight2])#1-2
        elif LoRA_switch1 == 'True' and LoRA_switch2 == 'False' and LoRA_switch3 == 'False':
            pipe.load_lora_weights(LoRA_path1,adapter_name="lora1")
            #pipe.set_adapters(["lora1"], adapter_weights=[lora_weight1])
            # 1-peft方法激活LoRA，降速
            pipe.fuse_lora(lora_scale=lora_weight1)
            #单个LoRA采用传统方法，提速
        elif LoRA_switch1 == 'False' and LoRA_switch2 == 'True' and LoRA_switch3 == 'True':
            pipe.load_lora_weights(LoRA_path2,adapter_name="lora2")
            pipe.load_lora_weights(LoRA_path3,adapter_name="lora3")
            pipe.set_adapters(["lora2", "lora3"], adapter_weights=[lora_weight2, lora_weight3])#2-3
        elif LoRA_switch1 == 'False' and LoRA_switch2 == 'True' and LoRA_switch3 == 'False':
            pipe.load_lora_weights(LoRA_path2,adapter_name="lora2")
            #pipe.set_adapters(["lora2"], adapter_weights=[lora_weight2])
            # 2-peft方法激活LoRA，降速
            pipe.fuse_lora(lora_scale=lora_weight2)
            #单个LoRA采用传统方法，提速
        elif LoRA_switch1 == 'True' and LoRA_switch2 == 'False' and LoRA_switch3 == 'True':
            pipe.load_lora_weights(LoRA_path1,adapter_name="lora1")
            pipe.load_lora_weights(LoRA_path3,adapter_name="lora3")
            pipe.set_adapters(["lora1", "lora3"], adapter_weights=[lora_weight1, lora_weight3])#1-3
        elif LoRA_switch1 == 'False' and LoRA_switch2 == 'False' and LoRA_switch3 == 'True':
            pipe.load_lora_weights(LoRA_path3,adapter_name="lora3")
            #pipe.set_adapters(["lora3"], adapter_weights=[lora_weight3])
            # 3-peft方法激活LoRA，降速
            pipe.fuse_lora(lora_scale=lora_weight3)
            #单个LoRA采用传统方法，提速
        elif LoRA_switch1 == 'False' and LoRA_switch2 == 'False 'and LoRA_switch3 == 'False':
            pass
        
        #进度条
        pbar = ProgressBar(int(num_inference_steps))
        def callback(*_):
            pbar.update(1)
        #分割输入的prompt文本再传入prompt_embeds，实现解除77个字符限制(无法识别中文)。
        def get_pipeline_embeds(prompt, negative_prompt_e, device):
            max_length = pipe.tokenizer.model_max_length
            count_prompt = len(prompt.split(" "))
            count_negative_prompt = len(negative_prompt_e.split(" "))
            if count_prompt >= count_negative_prompt:
                input_ids = pipe.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
                shape_max_length = input_ids.shape[-1]
                negative_ids = pipe.tokenizer(negative_prompt_e, truncation=False, padding="max_length",
                                                  max_length=shape_max_length, return_tensors="pt").input_ids.to(device)
            else:
                negative_ids = pipe.tokenizer(negative_prompt_e, return_tensors="pt", truncation=False).input_ids.to(device)
                shape_max_length = negative_ids.shape[-1]
                input_ids = pipe.tokenizer(prompt, return_tensors="pt", truncation=False, padding="max_length",
                                               max_length=shape_max_length).input_ids.to(device)
            concat_embeds = []
            neg_embeds = []
            for i in range(0, shape_max_length, max_length):
                concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
                neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])
            return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)
        prompt = positive_prompt
        negative_prompt_e = negative_prompt
        print("Our inputs ", prompt, negative_prompt_e, len(prompt.split(" ")), len(negative_prompt_e.split(" ")))
        prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(prompt, negative_prompt_e, "cuda")
        #如果是中文输入(使用妙笔模型的clip)则使用标准pipeline和分割后的输入prompt_embeds。如果不是，则使用社区pipeline(已解除77字符限制且带权重)和原始prompt文本.
        if Language == "中文(简体、繁体)":
            images = pipe(
                       prompt_embeds=prompt_embeds, 
                       negative_prompt_embeds=negative_prompt_embeds,
                    #    prompt=positive_prompt, 
                       num_inference_steps=num_inference_steps, 
                       guidance_scale=guidance_scale, 
                       #与进度条相关
                       callback_steps=1,
                       callback=callback,
                       #与进度条相关
                       height=height, 
                       width=width, 
                       cross_attention_kwargs={"scale": multi_loras_weight}, 
                       eta=eta,
                    #    negative_prompt=negative_prompt,
                       generator = generator
                       ).images[0]
            output_t = pil2tensor(images)
        elif Language == "English":
            images = pipe(
                    #    prompt=prompt_embeds, 
                    #    negative_prompt_embeds=negative_prompt_embeds,
                       prompt=positive_prompt, 
                       num_inference_steps=num_inference_steps, 
                       guidance_scale=guidance_scale, 
                       #与进度条相关
                       callback_steps=1,
                       callback=callback,
                       #与进度条相关
                       height=height, 
                       width=width, 
                       cross_attention_kwargs={"scale": multi_loras_weight}, 
                       eta=eta,
                       negative_prompt=negative_prompt,
                       generator = generator
                       ).images[0]
            output_t = pil2tensor(images)
        pipe.to(torch.device('cpu'), torch.float32)
        return (output_t, )

class UL_MiaoBi_diffusers_SD15:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                #必须输入，不输入则无法开始pipe。
                "MiaoBi_Loader": ("MiaoBi_Loader", {"forceInput": True}),
                # "diffusion_name": (os.listdir(os.path.join(folder_paths.models_dir, "diffusion")), 'utf-8', ),
                # os.listdir列出diffusers文件夹下的所有文件（包括文件夹），使用utf-8开启非英文支持，输出选择的文件夹或者文件的名字，类型字符串string值。
                "diffusion_name": (["ShineChen1024/MiaoBi","SG161222/Realistic_Vision_V6.0_B1_noVAE","Lykon/dreamshaper-8","emilianJR/epiCRealism","Yntec/epiCPhotoGasm","runwayml/stable-diffusion-v1-5","Niggendar/counterfeitv30_fix_fp16","stablediffusionapi/rev-animated","sam749/CyberRealistic-v4-2","Lykon/dreamshaper-8-lcm"], {"default": "ShineChen1024/MiaoBi"}),
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "动漫风格的风景画，有山脉、湖泊，也有繁华的小镇子，色彩鲜艳，光影效果明显，高清，高细节，高分辨率。"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "低质量，低分辨率。"
                }),
                "num_inference_steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 99,
                    "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1,
                    "max": 99,
                    "step": 0.1
                }),
                "eta": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 0,
                    "max": 9999999
                }),
                "height": ("INT", {
                    "default": 768,
                    "min": 0,
                    "max": 9999999
                }),
                "scheduler": (scheduler_list,{"default":"Euler"}),
                "diffusion_dtype": (["auto","fp16","bf16","fp32"],{"default":"auto"}),
                # "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "UL_MiaoBi_diffusers_SD15"
    CATEGORY = "ExtraModels/UL MiaoBi"
    TITLE = "UL MiaoBi Diffusers SD15"
    DESCRIPTION = """diffusion_dtype--text_encoder&diffusion--clip&model|diffusion的clip和model必须统一精度."""

    def UL_MiaoBi_diffusers_SD15(
                 self, 
                 device,
                 diffusion_name,
                 MiaoBi_Loader,
                 positive_prompt, 
                 negative_prompt, 
                 guidance_scale, 
                 width, 
                 height, 
                 num_inference_steps, 
                 seed,
                 scheduler,
                 diffusion_dtype,
                #  batch_size,
                 eta):
        
        MiaoBi_Loader = MiaoBi_Loader.split("|")
        # diffusion_name_local = diffusion_name.rsplit('/', 1)[1]
        # 获取最后一个/后面的所有内容
        name_pre = diffusion_name.replace("/","--")
        #替换输入名字中的/为--来适配下载到本地的路径，输出为xxx--xxx
        diffusion_name_local = ("models--" + name_pre)
        #拼接models--到前面修改的名字前，来适配snapshot方法下载到本地的路径，输出为models--xxx--xxx
        
        device = get_device_by_name(device)
        u_dtype = get_dtype_by_name(diffusion_dtype)
        
        pretrain_model = os.path.join(folder_paths.models_dir, "diffusers", diffusion_name_local)
        #拼接254行选择的文件夹返回的文件夹名字符串，组成完整的文件路径

        generator = torch.manual_seed(seed)
        #CPU种子生成器，添加.cuda可用gpu生成。Seed generator by cpu，"torch.manual_seed(seed)" by cpu,"torch.cuda.manual_seed(seed)"by gpu
    
        #检查diffusers文件夹下是否有输入的的diffusion名字本地文件，并且判断下载是否成功(模型文件夹下的unet下是否有.safetensors或者.bin文件)
        if os.path.exists(os.path.join(folder_paths.models_dir, "diffusers", diffusion_name_local,)) and os.access(os.path.join(folder_paths.models_dir, "diffusers", diffusion_name_local,"unet/diffusion_pytorch_model.safetensors", ), os.F_OK) or os.access(os.path.join(folder_paths.models_dir, "diffusers", diffusion_name_local,"unet/diffusion_pytorch_model.bin", ), os.F_OK):
            # print("离线下如果某个文件没下载成功，会报错Error no file named xxx.bin或者xxx.safetensors。\n如果某个文件下载不完整，会报错Unable to load weights from checkpoint file 'xxx.bin或者xxx.safetensors',需要删除该文件后重新下载。\n")
            pass
        else:
            print("\n\033[93m如果下载没完成或者缺失diffusion模型，再次运行会将全部文件重新下载一遍!\n大陆环境需要魔法或者设置镜像地址！\033[0m\n")
            #定义从hg下载模型到本地diffusers文件夹内的"diffusion_name_local"下。
            #hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json",local_dir=xxx)#下载单个文件，例子里面是config.json),非blob编码保存到Local_dir下面.
          
            snapshot_download(
                  diffusion_name, 
                  #huggingface上的repo_id例如"THUDM/chatglm-6b",
                  local_dir=os.path.join(folder_paths.models_dir, "diffusers",diffusion_name_local),
                  local_dir_use_symlinks=False,
                  #使用非blob编码路径下载到本地
                  #revision=revision#指定版本
                  )
            
        diffusion_scheduler = UniPCMultistepScheduler.from_pretrained(pretrain_model, subfolder="scheduler")
        #定义diffusers模型自带调度器.调用diffusers模型文件scheduler子目录下的调度器
        #pipe = DiffusionPipeline.from_pretrained(pretrain_model, scheduler = diffusion_scheduler, torch_dtype=torch.float16).to("cuda")
        #Diffusion管线，应用定义的调度器，使用路径为pretrain_model文件夹下的模型。
        num_hidden_layers = MiaoBi_Loader[1]
        if num_hidden_layers == "-1":
            #    使用全部12层数
               text_encoder = CLIPTextModel.from_pretrained(
                   os.path.join(folder_paths.models_dir, "diffusers", diffusion_name_local,"text_encoder", ), 
                   torch_dtype=u_dtype,
                #    torch_dtype=torch.float16
                   ).to(device)
        elif num_hidden_layers != "-1":
                 print("\033[93m如果某个文件下载不完整，会报错Unable to load weights from checkpoint file 'xxx.bin或者xxx.safetensors',需要删除该文件后重新下载。\n\033[91m离线运行下如果某个文件没下载成功，会报错Error no file named xxx.bin或者xxx.safetensors，需要联网进行下载。\033[0m\n")
                 #"\033[93mxxxxxxxx\033[0m"powershell黄色字体开始显示(93m)，结束后默认颜色字体显示(0m)后续。
                 #"\033[91mxxxxxxxx\033[93m"powershell红色字体显示(91m)，结束后黄色(93m)渲染后续。
                 num_hidden_layers = num_hidden_layers + 1
                 text_encoder = CLIPTextModel.from_pretrained(
                     os.path.join(folder_paths.models_dir, "diffusers", diffusion_name_local,"text_encoder", ), 
                     num_hidden_layers = 12 + num_hidden_layers, 
                    #  使用模型的前n层，sd15默认12
                     torch_dtype=u_dtype,
                    #  torch_dtype=torch.float16
                     ).to(device)
        cache_dir = os.path.join(folder_paths.models_dir, "diffusers", diffusion_name_local, )
        lpw_pipe = os.path.join(current_directory, "MiaoBi_Pipelines", "lpw_stable_diffusion.py")
        #自定义第三方pipeline路径，传递到custom_pipeline.
        #判断是否MiaoBi模型，如果是，采用AutoTokenizer。如果不是，采用CLIPTokenizer，并启用自定义社区pipeline，解除字数限制。
        if not is_module_imported('StableDiffusionPipeline'):
            from diffusers import StableDiffusionPipeline
        if diffusion_name == "ShineChen1024/MiaoBi":
               tokenizer = AutoTokenizer.from_pretrained(os.path.join(folder_paths.models_dir, "diffusers", diffusion_name_local,"tokenizer"), trust_remote_code=True)
               pipe = StableDiffusionPipeline.from_pretrained(
                                                        cache_dir,
                                                        tokenizer=tokenizer,
                                                        text_encoder = text_encoder,
                                                        # custom_pipeline=lpw_pipe,#获取的是"lpw-stable-diffusion_xl.py/lpw_stable_diffusion.py"文件，可以自定义到插件目录下。解除77个tokens限制,通过使用社区提供的"lpw_stable_diffusion"，我们可以解锁77个tokens限制，并可以使用权重,并通过更长的prompt生成高质量图像。该pipeline不支持妙笔模型。
                                                        requires_safety_checker = False,
                                                        #不需要安全检测,disable safety_checker
                                                        safety_checker = None,
                                                        #禁用安全检测,disable safety_checker
                                                        torch_dtype=u_dtype,
                                                        # torch_dtype=torch.float16
                                                      ).to(device)
        
        elif diffusion_name != "ShineChen1024/MiaoBi":
                tokenizer = CLIPTokenizer.from_pretrained(os.path.join(folder_paths.models_dir, "diffusers", diffusion_name_local),subfolder="tokenizer", trust_remote_code=True)
                pipe = StableDiffusionPipeline.from_pretrained(
                                                        cache_dir,
                                                        text_encoder = text_encoder,
                                                        tokenizer=tokenizer,
                                                        custom_pipeline=lpw_pipe,
                                                        #获取的是"lpw-stable-diffusion_xl.py/lpw_stable_diffusion.py"文件，可以自定义到插件目录下。解除77个tokens限制,并可以使用权重,通过使用社区提供的"lpw_stable_diffusion"，我们可以解锁77个tokens限制，并通过更长的prompt生成高质量图像。
                                                        requires_safety_checker = False,
                                                        #不需要安全检测,disable safety_checker
                                                        safety_checker = None,
                                                        # 禁用安全检测,disable safety_checker
                                                        torch_dtype=u_dtype,
                                                        # torch_dtype=torch.float16
                                                      ).to(device)
        # sd1.5专用调度器配置，跟sdxl等不兼容。
        if scheduler == "UniPC":
            scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "DDIM":
            scheduler = DDIMScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "Euler a":
            scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "DPM++ 2M":
            scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "DPM++ 2M Karras":
            scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        elif scheduler== "DPM++ 2M SDE":
            scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++")
        elif scheduler == "DPM++ 2M SDE Karras":
            scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
        elif scheduler == "DPM++ SDE":
            scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, lower_order_final=True) 
        elif scheduler == "DPM++ SDE Karras":
            scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, lower_order_final=True)
        elif scheduler == "DPM2":
            scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "DPM2 Karras":
            scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        elif scheduler == "DPM2 a":
            scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "DPM2 a Karras":
            scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        elif scheduler == "Euler":
            scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "Heun":
            scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "LMS":
            scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "LMS Karras":
            scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        elif scheduler == "TCD":
            scheduler = TCDScheduler.from_config(pipe.scheduler.config) 
        elif scheduler == "LCM":
            scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "diffusion":
            scheduler = diffusion_scheduler
        pipe.scheduler = scheduler
        # 应用调度器
        pipe.enable_vae_tiling()#启用分块解码
        pipe.enable_xformers_memory_efficient_attention()
        #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        #diffusers自带调度器，单个调度器写死。
        #pipe.enable_model_cpu_offload()#如果需要启用，去除上面pipe中的.to(device)，启用可降低显存占用，但处理时间增加。if "pipe.enable_model_cpu_offload()" needed ,delete ".to(device)" in line 101.With enabled,less vram but need little more time for process.
        #pipe.diffusion.set_default_attn_processor()#大显存可提速>6GB
        #pipe.vae.set_default_attn_processor()#大显存可提速>6GB
        #pipe.enable_attention_slicing()#额
        # pipe.enable_sequential_cpu_offload() # my graphics card VRAM is very low
        
        LoRA_path1 = folder_paths.get_full_path("loras", MiaoBi_Loader[2])
        LoRA_path2 = folder_paths.get_full_path("loras", MiaoBi_Loader[3])
        LoRA_path3 = folder_paths.get_full_path("loras", MiaoBi_Loader[4])
        LoRA_switch1 = MiaoBi_Loader[8]
        LoRA_switch2 = MiaoBi_Loader[9]
        LoRA_switch3 = MiaoBi_Loader[10]
        lora_weight1 = float(MiaoBi_Loader[5])
        lora_weight2 = float(MiaoBi_Loader[6])
        lora_weight3 = float(MiaoBi_Loader[7])
        multi_loras_weight = float(MiaoBi_Loader[11])
            
        if (MiaoBi_Loader[0]) == 'True' and (MiaoBi_Loader[8]) == 'False' and (MiaoBi_Loader[9]) == 'False' and (MiaoBi_Loader[10]) == 'False':
            LoRA_switch1 = False
            LoRA_switch2 = False
            LoRA_switch3 = False
            if not is_module_imported('apply_hidiffusion'):
                from .MiaoBi_hidiffusion.hidiffusion import apply_hidiffusion
            apply_hidiffusion(pipe, True, True, "SD1.5")
        elif (MiaoBi_Loader[0]) == 'False':
            pass
        else:
            raise Exception(f"Hidiffusion is incompatible with LoRA for now(Hidiffusion高清图生成目前不支持LoRA).")
        
        if LoRA_switch1 == 'True' and LoRA_switch2 == 'True' and LoRA_switch3 == 'True':
            pipe.load_lora_weights(LoRA_path1,adapter_name="lora1")
            pipe.load_lora_weights(LoRA_path2,adapter_name="lora2")
            pipe.load_lora_weights(LoRA_path3,adapter_name="lora3")
            pipe.set_adapters(["lora1", "lora2", "lora3"], adapter_weights=[lora_weight1, lora_weight2, lora_weight3])#1-2-3,应用3个LoRA，使用peft方法，降速明显。
        elif LoRA_switch1 == 'True' and LoRA_switch2 == 'True' and LoRA_switch3 == 'False':
            pipe.load_lora_weights(LoRA_path1,adapter_name="lora1")
            pipe.load_lora_weights(LoRA_path2,adapter_name="lora2")
            pipe.set_adapters(["lora1", "lora2"], adapter_weights=[lora_weight1, lora_weight2])#1-2
        elif LoRA_switch1 == 'True' and LoRA_switch2 == 'False' and LoRA_switch3 == 'False':
            pipe.load_lora_weights(LoRA_path1,adapter_name="lora1")
            #pipe.set_adapters(["lora1"], adapter_weights=[lora_weight1])
            # 1-peft方法激活LoRA，降速
            pipe.fuse_lora(lora_scale=lora_weight1)
            #单个LoRA采用传统方法，提速
        elif LoRA_switch1 == 'False' and LoRA_switch2 == 'True' and LoRA_switch3 == 'True':
            pipe.load_lora_weights(LoRA_path2,adapter_name="lora2")
            pipe.load_lora_weights(LoRA_path3,adapter_name="lora3")
            pipe.set_adapters(["lora2", "lora3"], adapter_weights=[lora_weight2, lora_weight3])#2-3
        elif LoRA_switch1 == 'False' and LoRA_switch2 == 'True' and LoRA_switch3 == 'False':
            pipe.load_lora_weights(LoRA_path2,adapter_name="lora2")
            #pipe.set_adapters(["lora2"], adapter_weights=[lora_weight2])
            # 2-peft方法激活LoRA，降速
            pipe.fuse_lora(lora_scale=lora_weight2)
            #单个LoRA采用传统方法，提速
        elif LoRA_switch1 == 'True' and LoRA_switch2 == 'False' and LoRA_switch3 == 'True':
            pipe.load_lora_weights(LoRA_path1,adapter_name="lora1")
            pipe.load_lora_weights(LoRA_path3,adapter_name="lora3")
            pipe.set_adapters(["lora1", "lora3"], adapter_weights=[lora_weight1, lora_weight3])#1-3
        elif LoRA_switch1 == 'False' and LoRA_switch2 == 'False' and LoRA_switch3 == 'True':
            pipe.load_lora_weights(LoRA_path3,adapter_name="lora3")
            #pipe.set_adapters(["lora3"], adapter_weights=[lora_weight3])
            # 3-peft方法激活LoRA，降速
            pipe.fuse_lora(lora_scale=lora_weight3)
            #单个LoRA采用传统方法，提速
        elif LoRA_switch1 == 'False' and LoRA_switch2 == 'False 'and LoRA_switch3 == 'False':
            pass
        
        #进度条
        pbar = ProgressBar(int(num_inference_steps))
        def callback(*_):
            pbar.update(1)
        
        #分割输入的prompt文本再传入prompt_embeds，实现解除77个字符限制(无法识别中文)。
        def get_pipeline_embeds(prompt, negative_prompt_e, device):
            max_length = pipe.tokenizer.model_max_length
            count_prompt = len(prompt.split(" "))
            count_negative_prompt = len(negative_prompt_e.split(" "))
            if count_prompt >= count_negative_prompt:
                input_ids = pipe.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
                shape_max_length = input_ids.shape[-1]
                negative_ids = pipe.tokenizer(negative_prompt_e, truncation=False, padding="max_length",
                                                  max_length=shape_max_length, return_tensors="pt").input_ids.to(device)
            else:
                negative_ids = pipe.tokenizer(negative_prompt_e, return_tensors="pt", truncation=False).input_ids.to(device)
                shape_max_length = negative_ids.shape[-1]
                input_ids = pipe.tokenizer(prompt, return_tensors="pt", truncation=False, padding="max_length",
                                               max_length=shape_max_length).input_ids.to(device)
            concat_embeds = []
            neg_embeds = []
            for i in range(0, shape_max_length, max_length):
                concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
                neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])
            return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)
        prompt = positive_prompt
        negative_prompt_e = negative_prompt
        print("Our inputs ", prompt, negative_prompt_e, len(prompt.split(" ")), len(negative_prompt_e.split(" ")))
        prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(prompt, negative_prompt_e, "cuda")
        # batch = batch_size
        #如果是中文输入(使用妙笔模型的clip)则使用标准pipeline和分割后的输入prompt_embeds。如果不是，则使用社区pipeline(已解除77字符限制且带权重)和原始prompt文本.
        if diffusion_name == "ShineChen1024/MiaoBi":
            images = pipe(
                       prompt_embeds=prompt_embeds, 
                       negative_prompt_embeds=negative_prompt_embeds,
                    #    prompt=positive_prompt, 
                       num_inference_steps=num_inference_steps, 
                       guidance_scale=guidance_scale, 
                       cross_attention_kwargs={"scale": multi_loras_weight}, 
                       height=height, 
                       width=width, 
                       eta=eta, 
                       #与进度条相关
                       callback_steps=1,
                       callback=callback,
                       #与进度条相关
                       generator=generator,
                    #    num_images_per_prompt=batch,
                    #    negative_prompt=negative_prompt
                       ).images[0]
            output_t = pil2tensor(images)
        else:
        #最终生图控制
            images = pipe(
                       prompt=positive_prompt, 
                       negative_prompt=negative_prompt,
                       num_inference_steps=num_inference_steps, 
                       guidance_scale=guidance_scale, 
                       cross_attention_kwargs={"scale": multi_loras_weight}, 
                       height=height, 
                       width=width, 
                       eta=eta, 
                       #与进度条相关
                       callback_steps=1,
                       callback=callback,
                       #与进度条相关
                       generator=generator,
                       #    num_images_per_prompt=batch,
                       ).images[0]
            output_t = pil2tensor(images)
        pipe.to(torch.device('cpu'), torch.float32)
        return (output_t,)
        
NODE_CLASS_MAPPINGS = {
    "UL_MiaoBi_Singlefile_SD15": UL_MiaoBi_Singlefile_SD15,
    "UL_MiaoBi_ResAdapterLoader": UL_MiaoBi_ResAdapterLoader,
    "UL_MiaoBi_diffusers_SD15": UL_MiaoBi_diffusers_SD15,
}