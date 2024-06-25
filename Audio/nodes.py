import os
import torch
import torchaudio
import numpy as np
import folder_paths
import tempfile
# import time
from ..UL_common.common import is_module_imported, get_device_by_name, get_dtype_by_name
# from huggingface_hub import snapshot_download as hg_snapshot_download
from huggingface_hub import hf_hub_download
from .utils import stable_audio_open_generate, stable_audio_open_load_model, Run_ChatTTS, OpenVoiceV2_clone, uvr5_split, noise_suppression, get_audio_from_video

# 获取当前文件的目录，stable-audio-open模型配置文件
current_directory = os.path.dirname(os.path.abspath(__file__))
comfy_temp_dir = folder_paths.get_temp_directory()
output_dir = folder_paths.get_output_directory()
sys_temp_dir = tempfile.gettempdir()

# https://huggingface.co/stabilityai/stable-audio-open-1.0

class UL_Audio_stable_audio_open:
    def __init__(self):
        self.initialized_model = None
        self.sample_rate=None
        self.sample_size=None

    @classmethod
    def INPUT_TYPES(s):
        checkpoints_list = folder_paths.get_filename_list("checkpoints")
        checkpoints_list.insert(0, "Auto_DownLoad")
        return {
            "required": {
                "advance_preview_only": ("BOOLEAN",{"default": False}),
                "ckpt_name": (checkpoints_list, ),
                "ref_audio": ("AUDIO_PATH",),
                "Stable_Audio_mask_args": ("mask_args",),
                "prompt": ("STRING", 
                         {
                            "multiline": True, 
                            "default": "a woman's groaing by fucked_silly.",
                            "dynamicPrompts": True
                          }),
                "seconds":("FLOAT", {"default": 47, "min": 1, "max": 10000,"step": 0.1}),
                "steps": ("INT", {"default": 16, "min": 1, "max": 10000,"step": 1}),
                "seed":  ("INT", {"default": 0, "min": 0, "max": np.iinfo(np.int32).max}), 
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 25.0, "step": 0.1}), 
                "sigma_min": ("FLOAT", {"default": 0.3, "min": 0.00, "max": 2.00, "step": 0.01}),
                "sigma_max": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "init_noise_level": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 100.0, "step": 0.01}), 
                "sampler_type": (["dpmpp-3m-sde", "dpmpp-2m-sde", "k-dpm-fast", "k-lms", 'k-heun', 'k-dpmpp-2s-ancestral', 'k-dpm-2', 'k-dpm-adaptive'], {"default": "dpmpp-3m-sde"}),
                "dtype": (["auto", "fp16", "bf16", "fp32"],{"default": "auto"}), 
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
                "use_init_audio": ("BOOLEAN",{"default": False}),
                "apply_mask_cropfrom": ("BOOLEAN",{"default": True}),
            }
        }
    
    RETURN_TYPES = ("AUDIO_PATH",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "UL_stable_audio_open"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL Audio Stable-Audio-Open-声音生成"
    
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)

    def UL_stable_audio_open(self, prompt,seconds,steps,seed, cfg_scale,  sigma_min, sigma_max, ckpt_name, dtype, sampler_type, device, ref_audio, use_init_audio, init_noise_level, Stable_Audio_mask_args, apply_mask_cropfrom, advance_preview_only):
        ref_audio = get_audio_from_video(ref_audio)
        #如果ckpt_name为‘Auto_Download’，混合进ckpt_path之后，ckpt_path会被写为None值。
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if ckpt_path == None:
            if not os.access(os.path.join(folder_paths.models_dir, "checkpoints\ex_ExtraModels\stable-audio-open-1.0.safetensors"), os.F_OK):
                hf_hub_download(
                            repo_id="stabilityai/stable-audio-open-1.0", 
                            filename="model.safetensors",
                            local_dir=os.path.join(folder_paths.models_dir, "checkpoints\ex_ExtraModels")
                            )
                old_file = os.path.join(folder_paths.models_dir, "checkpoints\ex_ExtraModels\model.safetensors")
                new_file = os.path.join(folder_paths.models_dir, "checkpoints\ex_ExtraModels\stable-audio-open-1.0.safetensors")
                os.rename(old_file, new_file)
                ckpt_path = new_file
            else:
                ckpt_path = os.path.join(folder_paths.models_dir, "checkpoints\ex_ExtraModels\stable-audio-open-1.0.safetensors")
                
        dtype = get_dtype_by_name(dtype)
        device = get_device_by_name(device)

        if self.initialized_model:
            self.initialized_model = self.initialized_model.to(device, dtype) #t5-base
        else:
            self.initialized_model, self.sample_rate, self.sample_size = stable_audio_open_load_model(device, ckpt_path, dtype)

        # 根据时长，计算size
        self.sample_size = int(self.sample_rate*seconds)
        sample_size = self.sample_size
        if use_init_audio == True:
            #使用soundfile加载音频，数据类型为np.ndarray
            from soundfile import read
            init_audio, in_sr = read(ref_audio)
            #使用torchaudio加载参考音频，数据类型为torch tensor
            # init_audio, in_sr = torchaudio.load(ref_audio)
            print("\033[93m音频数据类型：", type(init_audio),  "    音频采样率：",  in_sr, "\033[0m")
            init_audio = torch.from_numpy(init_audio).float().div(32767)
            init_audio = init_audio.to(device, dtype)
            if init_audio.dim() == 1:
                init_audio = init_audio.unsqueeze(0) # [1, n]
            elif init_audio.dim() == 2:
                init_audio = init_audio.transpose(0, 1) # [n, 2] -> [2, n]
            if in_sr != self.sample_rate:
                from torchaudio import transforms as T
                resample_tf = T.Resample(in_sr, self.sample_rate).to(device, dtype)
                init_audio = resample_tf(init_audio)
            audio_length = init_audio.shape[-1]
            if audio_length >self. sample_size:
                input_sample_size = audio_length + (self.initialized_model.min_input_length - (audio_length % self.initialized_model.min_input_length)) % self.initialized_model.min_input_length
                sample_size = input_sample_size
            init_audio = (self.sample_rate, init_audio)
        else:
            init_audio = None
        if init_noise_level == 0:
            init_noise_level = None
        
        mask_args_loader = Stable_Audio_mask_args.split('|')
        if apply_mask_cropfrom == True: 
            mask_args = {
                "cropfrom": float(mask_args_loader[0]),
                "pastefrom": float(mask_args_loader[1]),
                "pasteto": float(mask_args_loader[2]),
                "maskstart": float(mask_args_loader[3]),
                "maskend": float(mask_args_loader[4]),
                "softnessL": float(mask_args_loader[5]),
                "softnessR": float(mask_args_loader[6]),
                "marination": float(mask_args_loader[7]),
            }
        else:
            mask_args = None 
            
        output=stable_audio_open_generate(self.initialized_model, prompt, seconds, seed, steps, cfg_scale, sample_size,  sigma_min, sigma_max, sampler_type, device, init_audio, init_noise_level, self.sample_rate, mask_args)

        #生成后将模型转移到cpu，释放显存。
        self.initialized_model.to(torch.device('cpu'), torch.float32)

        comfy_output_dir = folder_paths.get_output_directory()
        # 添加文件名后缀
        audio_file = 'UL_audio'
        audio_file = f"{audio_file}.wav"
        audio_path = os.path.join(comfy_output_dir, audio_file)

        torchaudio.save(audio_path, output, self.sample_rate)
        
        # if save_to_desktop == True:
        #     desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        #     new_name = f"{save_name}_{now}.wav"
        #     copy_and_rename_file(audio_path, desktop_path, new_name)
        # if save_to_custom_folder == True:
        #     new_name = f"{save_name}_{now}.wav"
        #     copy_and_rename_file(audio_path, custom_folder, new_name)
            
        if advance_preview_only == True:
            result = {
                "filename": audio_file,
                "subfolder": "",
                "type": "output",
                "prompt":prompt
                }
        else:
            result = audio_path
        return (result,)
        
# https://huggingface.co/facebook/musicgen-small
# Implementation with audiocraft

class UL_Audio_facebook_musicgen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "advance_preview_only": ("BOOLEAN",{"default": False}),
                "model": (["facebook/musicgen-stereo-small", "facebook/musicgen-stereo-medium", "facebook/musicgen-stereo-large","facebook-- musicgen-stereo-melody", "facebook/musicgen-stereo-melody-large", "nateraw/musicgen-songstarter-v0.2"],{"default": "facebook/musicgen-stereo-small"}), 
                "ref_audio_for_melody": ("AUDIO_PATH",),
                "trim_ref_audio": ("BOOLEAN",{"default": False}),
                "start_time": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
                "duration": ("FLOAT" , {"default": 9, "min": 0, "max": 10000000, "step": 0.01}),
                "musicgen_type": (["musicgen", "musicgen_melody"],{"default": "musicgen"}), 
                "prompt": ("STRING", 
                            {
                                "multiline": True, 
                                "default": 'Chinese traditional music.\nor\nacoustic, guitar, melody, trap, d minor, 90 bpm',
                                "dynamicPrompts": True
                            }),
                "seconds":("FLOAT", {
                            "default": 30, 
                            "min": 1, #Minimum value
                            "max": 9999.0, #Maximum value
                            "step": 0.1, #Slider's step
                            "display": "number" # Cosmetic only: display as "number" or "slider"
                        }),
                "guidance_scale":("INT", {
                            "default": 3, 
                            "min": 0, #Minimum value
                            "max": 9999, #Maximum value
                        }),
                "top_k":("INT", {
                            "default": 250, 
                            "min": 0, #Minimum value
                            "max": 9999, #Maximum value
                        }),
                "top_p":("FLOAT", {
                            "default": 0.0, 
                            "min": 0.0, #Minimum value
                            "max": 9999.0, #Maximum value
                            "step": 0.1, #Slider's step
                            # "display": "number" # Cosmetic only: display as "number" or "slider"
                        }),
                "temperature":("FLOAT", {"default": 1.0, "min": 0.0, "max": 999.0, "step": 0.1, }),
                "seed":  ("INT", {"default": 0, "min": 0, "max": np.iinfo(np.int32).max}), 
                # "dtype": (["auto", "fp16", "bf16", "fp32"],{"default": "auto"}), 
                "two_step_cfg": ("BOOLEAN", {"default":False}, ),
                "use_sampling": ("BOOLEAN", {"default":True}, ),
                "extend_stride": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999.0, "step": 0.1, }),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
                    },
                }
    
    RETURN_TYPES = ("AUDIO_PATH",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "UL_facebook_musicgen"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL Audio facebook-musicge-音乐、旋律生成"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)
  
    def UL_facebook_musicgen(self, model, prompt, seconds, device, seed, guidance_scale, top_k, top_p, temperature, two_step_cfg, use_sampling, extend_stride, ref_audio_for_melody, musicgen_type, trim_ref_audio, start_time, duration, advance_preview_only):
        seed = torch.manual_seed(seed)
        ref_audio_for_melody = get_audio_from_video(ref_audio_for_melody)
        if trim_ref_audio == True:
            # 获取输入音频文件的路径和名字
            # dirname, filename = os.path.split(ref_audio)
            # 去除音频名字后缀，后续统一改为.wav
            # new_name = str(filename).replace(".mp3", "").replace(".wav", "").replace(".ogg", "").replace(".m4a", "").replace(".flac", "")
            temp_dir = tempfile.gettempdir()
            trim_audio_path = os.path.join(temp_dir,f'ChatTTS.wav')
            os.system(
                f'ffmpeg -i "{ref_audio}" -ss "{start_time}" -t "{duration}" "{trim_audio_path}" -y'
            )
            ref_audio = trim_audio_path
        
        device = get_device_by_name(device)
        # musicgen_modelpath = os.path.join(folder_paths.models_dir, "audio_checkpoints", "modesl--facebook--musicgen-small")
        # if model == 'Auto_DownLoad':
        #     if not os.access(os.path.join(musicgen_modelpath, "model.safetensors"), os.F_OK):
        #         hg_snapshot_download(
        #             #huggingface上的repo_id例如"THUDM/chatglm-6b",
        #             "facebook/musicgen-small", 
        #             #指定下载路径
        #             local_dir=os.path.join(folder_paths.models_dir, "checkpoints\ex_ExtraModels"),
        #             local_dir_use_symlinks=False,
        #             #使用非blob编码路径下载到本地
        #             #revision=revision#指定版本
        #             )
        if model == 'facebook/musicgen-stereo-small':
            musicgen_modelpath = os.path.join(folder_paths.models_dir, "audio_checkpoints", "models--facebook--musicgen-stereo-small")
            if not os.access(os.path.join(musicgen_modelpath, "model.safetensors"), os.F_OK):
                musicgen_modelpath = 'facebook/musicgen-stereo-small'
        elif model == 'facebook/musicgen-stereo-medium':
            musicgen_modelpath = os.path.join(folder_paths.models_dir, "audio_checkpoints", "models--facebook--musicgen-stereo-medium")
            if not os.access(os.path.join(musicgen_modelpath, "model.safetensors"), os.F_OK):
                musicgen_modelpath = 'facebook/musicgen-stereo-medium'
        elif model == 'facebook/musicgen-stereo-large':
            musicgen_modelpath = os.path.join(folder_paths.models_dir, "audio_checkpoints", "models--facebook--musicgen-stereo-large")
            if not os.access(os.path.join(musicgen_modelpath, "model-00001-of-00002.safetensors"), os.F_OK):
                musicgen_modelpath = 'facebook/musicgen-stereo-large'
        elif model == 'facebook/musicgen-stereo-melody':
            musicgen_modelpath = os.path.join(folder_paths.models_dir, "audio_checkpoints", "models--facebook--musicgen-stereo-melody")
            if not os.access(os.path.join(musicgen_modelpath, "model-00001-of-00002.safetensors"), os.F_OK):
                musicgen_modelpath = 'facebook/musicgen-stereo-melody'
        elif model == 'facebook/musicgen-stereo-melody-large':
            musicgen_modelpath = os.path.join(folder_paths.models_dir, "audio_checkpoints", "models--facebook--musicgen-stereo-melody-large")
            if not os.access(os.path.join(musicgen_modelpath, "model-00002-of-00003.safetensors"), os.F_OK):
                musicgen_modelpath = 'facebook/musicgen-stereo-melody-large'
        elif model == 'nateraw/musicgen-songstarter-v0.2':
            musicgen_modelpath = os.path.join(folder_paths.models_dir, "audio_checkpoints", "models--nateraw--musicgen-songstarter-v0.2")
            if not os.access(os.path.join(musicgen_modelpath, "state_dict.bin"), os.F_OK):
                musicgen_modelpath = 'nateraw/musicgen-songstarter-v0.2'
            
        # audiocraft方法
        if not is_module_imported('MusicGen'):
            from .music_gen_audiocraft.models import MusicGen
        if not is_module_imported('audio_write'):
            from .music_gen_audiocraft.data.audio import audio_write
        self.model = MusicGen.get_pretrained(musicgen_modelpath, device)
        if extend_stride == '0.0':
            extend_stride =None
        self.model.set_generation_params(use_sampling=use_sampling, duration=seconds, top_k=top_k, top_p=top_p, temperature=temperature, cfg_coef=guidance_scale, two_step_cfg=two_step_cfg, extend_stride=extend_stride)
        descriptions = [prompt]
        if musicgen_type == "musicgen_melody":
            melody, sr = torchaudio.load(ref_audio_for_melody)
            audio = self.model.generate_with_chroma(descriptions, melody[None].expand(1, -1, -1), sr, progress=True)
        else:
            audio = self.model.generate(descriptions, progress=True)
        sampling_rate = self.model.sample_rate
        del self.model
        output_dir = folder_paths.get_output_directory()
        audio_file = "UL_audio.wav"
        audio_path = os.path.join(output_dir, "UL_audio")
        for idx, one_wav in enumerate(audio):
            #保存带wav后缀的音频文件
            audio_write(audio_path, one_wav.cpu(), sampling_rate, strategy="loudness")

        if advance_preview_only == True:
            result = {
                "filename": audio_file,
                "subfolder": "",
                "type": "output",
                "prompt":prompt
                }
        else:
            result = f'{audio_path}.wav'
        return (result,)
        
# https://huggingface.co/2Noise/ChatTTS

class UL_Audio_ChatTTS:
    def __init__(self):
        self.speaker = None
    def INPUT_TYPES():
        return {
            "required": {
                "advance_preview_only": ("BOOLEAN",{"default": False}),
                "ChatTTS_Loader": ("UL_ChatTTS_Loader", {"forceInput": True}),
                "text": ("STRING", {"forceInput": True}),
                "random_speaker":("BOOLEAN",{"default": False}),
                "name_for_save_speaker": ("STRING", {"multiline": False, "default": "ChatTTS"}),
                "save_speaker":("BOOLEAN",{"default": False}),
                "oral":("INT", {"default": 2, "min": 0, "max": 10, "display": "slider", }),
                "laugh":("INT", {"default": 0, "min": 0, "max": 10, "display": "slider", }),
                "uv_break":("INT", {"default": 6, "min": 0, "max": 10, "display": "slider", }),
                "speed":("INT", {"default": 5, "min": 0, "max": 9999, }),
                "top_K":("INT", {"default": 20, "min": 0, "max": 9999, }),
                "top_P":("FLOAT", {"default": 0.70, "min": 0.00, "max": 9999.0, "step": 0.01, #Slider's step
                            # "display": "number" # Cosmetic only: display as "number" or "slider"
                        }),
                "temperature": ("FLOAT", {"default": 0.30, "min": 0.01, "max": 9999.00, "step": 0.01,}),
                "refine_temperature":("FLOAT",{"default":0.70,"min": 0.01,"max":9999.00, "step": 0.01,}),
                "infer_max_new_token":("INT", {"default": 2048, "min": 0, "max": 999999, }),
                "refine_max_new_token":("INT", {"default": 384, "min": 0, "max": 999999, }),
                "repetition_penalty":("FLOAT",{"default":1.05, "step": 0.01,}),
                "skip_refine_text":("BOOLEAN",{"default": False}),
                "mono2stereo": ("BOOLEAN",{"default": False}),
                "use_decoder":("BOOLEAN",{"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "do_text_normalization": ("BOOLEAN",{"default": False}),
                    },
                }
    
    RETURN_TYPES = ("AUDIO_PATH", )
    RETURN_NAMES = ("audio_path", )
    FUNCTION = "UL_ChatTTS"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL Audio ChatTTS-文本转语音"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)
  
    def UL_ChatTTS(self, text, seed, temperature, top_P, top_K, speed, refine_max_new_token, infer_max_new_token, name_for_save_speaker, repetition_penalty, refine_temperature, use_decoder, random_speaker, skip_refine_text, save_speaker, mono2stereo, do_text_normalization, ChatTTS_Loader, laugh, uv_break, oral, advance_preview_only):
        
        # sample_rate = 24000  # Assuming 24kHz sample rate
        # if save_speaker ==True:
        #     if random_speaker == False:
        #         raise Exception(f"Must with random_speaker checked if want to save speaker, specify a name first to avoid rewrite exited speaker(必须开启随机音色，才能进行保存，记得改保存名字防止覆盖已有音色).")
            
        ChatTTS_Loader_split = ChatTTS_Loader.split('|')
        model = ChatTTS_Loader_split[0]
        speakers = ChatTTS_Loader_split[1]
        device = get_device_by_name(ChatTTS_Loader_split[2])
        fix_saved_speaker_temperature = ChatTTS_Loader_split[3]
        prompt = f'[oral_{oral}][laugh_{laugh}][break_{uv_break}]'
        print("\033[93mPrompt(连接词、笑声、停顿)：", prompt, "\033[0m")
            
        ChatTTS_model_path = os.path.join(folder_paths.models_dir, "audio_checkpoints", "models--Dzkaka--ChatTTS")
        if model == 'Auto_DownLoad':
            if not os.access(os.path.join(ChatTTS_model_path, "asset\GPT.pt"), os.F_OK):
                ChatTTS_model_path = 'Dzkaka/ChatTTS'
            else:
                ChatTTS_model_path = ChatTTS_model_path
        else:
            ChatTTS_model_path = os.path.join(folder_paths.models_dir, "audio_checkpoints", model)
            
        speakers = os.path.join(current_directory, "ChatTTS_Speakers", speakers)
        
        if text == "":
            raise Exception(f"text params lost.")
        
        if random_speaker:
            self.speaker = None
        
        from .uilib import utils
        text_ori = text.strip()
        # 输入中英按语言换行，方便后续检测语言类别再替换数字为对应中英文，并输出为一个lsit列表。
        text_list=[t.strip() for t in text_ori.split("\n") if t.strip()]
        new_text=utils.split_text(text_list)
        # 合并list列表中所有值为字符串。
        final_text = ''.join(new_text)
        
        result, rand_spk = Run_ChatTTS(final_text, prompt, self.speaker, ChatTTS_model_path, device, temperature, top_P, top_K, use_decoder, refine_temperature, repetition_penalty, infer_max_new_token, refine_max_new_token, speed,name_for_save_speaker, skip_refine_text, speakers, save_speaker, mono2stereo, do_text_normalization, fix_saved_speaker_temperature, advance_preview_only)
        self.speaker = rand_spk
        return (result, )
    
class UL_Audio_OpenVoiceV2:
    def INPUT_TYPES():
        models_list = os.listdir(os.path.join(folder_paths.models_dir, r"audio_checkpoints"))
        models_list.insert(0, "Auto_DownLoad")
        return {
            "required": {
                "advance_preview_only": ("BOOLEAN",{"default": False}),
                "model": (models_list,),
                "ori_audio": ("AUDIO_PATH",),
                "ref_audio": ("AUDIO_PATH",),
                "trim_ref_audio": ("BOOLEAN",{"default": False}),
                "start_time": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
                "duration": ("FLOAT" , {"default": 6, "min": 0, "max": 10000000, "step": 0.01}),
                "mono2stereo": ("BOOLEAN",{"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "OpenVoiceV2_tau": ("FLOAT", {"default": 0.30, "min": 0.01, "max": 9999.00, "step": 0.01,}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
                    },
                }
    
    RETURN_TYPES = ("AUDIO_PATH",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "UL_OpenVoiceV2"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL Audio OpenVoiceV2-音色克隆"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)
  
    def UL_OpenVoiceV2(self, model, ori_audio, ref_audio, mono2stereo, seed, OpenVoiceV2_tau, device, trim_ref_audio, start_time, duration, advance_preview_only):
        
        if trim_ref_audio == True:
            # dirname, filename = os.path.split(ref_audio)
            # new_name = str(filename).replace(".mp3", "").replace(".wav", "").replace(".ogg", "").replace(".m4a", "").replace(".flac", "")
            temp_dir = tempfile.gettempdir()
            trim_audio_path = os.path.join(temp_dir,f'XTTS.wav')
            os.system(
                f'ffmpeg -i "{ref_audio}" -ss "{start_time}" -t "{duration}" "{trim_audio_path}" -y'
            )
            ref_audio = trim_audio_path
            
        print(ori_audio)
        model_path = os.path.join(folder_paths.models_dir, 'audio_checkpoints', model)
        OpenVoiceV2_model_path = os.path.join(folder_paths.models_dir, "audio_checkpoints\models--myshell-ai--OpenVoiceV2")
        if model == 'Auto_DownLoad':
            if not os.access(os.path.join(OpenVoiceV2_model_path, "checkpoints\converter\checkpoint.pth"), os.F_OK):
                model_path = 'myshell-ai/OpenVoiceV2'
            else:
                model_path = OpenVoiceV2_model_path
        device = get_device_by_name(device)
        audio_file = 'UL_audio_clone'
        audio_path = os.path.join(output_dir, f'{audio_file}.wav')
        stereo_audio_path = os.path.join(output_dir, f'{audio_file}_stereo.wav')
        print("\033[93m原声：", ori_audio, "\n目标声", ref_audio, "\033[0m")
        OpenVoiceV2_clone(model_path, device, ori_audio, ref_audio, comfy_temp_dir, audio_path, OpenVoiceV2_tau)
        if mono2stereo == True:
            os.system(
                f'ffmpeg -i "{audio_path}" -ac 2 "{stereo_audio_path}" -y'
            )
            audio_file = f'{audio_file}_stereo'
            audio_path = stereo_audio_path
            
        if advance_preview_only == True:
            result = {
                "filename": f'{audio_file}.wav',
                "subfolder": "",
                "type": "output",
                "prompt":"声音克隆",
                }
        else:
            result = audio_path
            
        return (result,)

#XTTS-v2 supports 17 languages: English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko) Hindi (hi).

class UL_Audio_XTTS:
    def INPUT_TYPES():
        models_list = os.listdir(os.path.join(folder_paths.models_dir, r"audio_checkpoints"))
        models_list.insert(0, "Auto_DownLoad")
        return {
            "required": {
                "advance_preview_only": ("BOOLEAN",{"default": False}),
                "model": (models_list,),
                "ref_audio": ("AUDIO_PATH",),
                "prompt": ("STRING", 
                         {
                            "multiline": True, 
                            "default": "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
                            "dynamicPrompts": True
                          }),
                "language": (["en", "zh-cn", "ru", "es", "ja", "ko", "fr", "de", "it", "pt", "tr", "nl", "cs", "ar", "hu", "hi", "pl"],{"default": "zh-cn"}), 
                "trim_ref_audio": ("BOOLEAN",{"default": False}),
                "start_time": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
                "duration": ("FLOAT" , {"default": 6, "min": 0, "max": 10000000, "step": 0.01}),
                "gpt_cond_len": ("INT", {"default": 3, "min": 0, "max": 999999}),
                "speed": ("FLOAT" , {"default": 1, "min": 0, "max": 10000000, "step": 0.01}),
                # "dtype": (["auto", "fp16", "bf16", "fp32"],{"default": "auto"}), 
                "apply_tts_api": ("BOOLEAN",{"default": False}),
                "tts_api_split_sentences": ("BOOLEAN",{"default": True}),
                "tts_api_emotion": ("STRING", {"multiline": False, "default": "angry, peace, expressiveness, aggressiveness, pace",}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "mono2stereo": ("BOOLEAN",{"default": False}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
                "use_deepspeed": ("BOOLEAN",{"default": False}),
                    },
                }
    
    RETURN_TYPES = ("AUDIO_PATH",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "UL_XTTS"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL Audio XTTS-文本转语音加音色克隆"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)
  
    def UL_XTTS(self, model, prompt, mono2stereo, seed, device, use_deepspeed, language, gpt_cond_len, ref_audio, trim_ref_audio, start_time, duration, apply_tts_api, speed, tts_api_split_sentences, tts_api_emotion, advance_preview_only):
        
        ref_audio = get_audio_from_video(ref_audio)
        
        if trim_ref_audio == True:
            # dirname, filename = os.path.split(ref_audio)
            # new_name = str(filename).replace(".mp3", "").replace(".wav", "").replace(".ogg", "").replace(".m4a", "").replace(".flac", "")
            temp_dir = tempfile.gettempdir()
            trim_audio_path = os.path.join(temp_dir,f'XTTS.wav')
            os.system(
                f'ffmpeg -i "{ref_audio}" -ss "{start_time}" -t "{duration}" "{trim_audio_path}" -y'
            )
            ref_audio = trim_audio_path
            
        model_path = os.path.join(folder_paths.models_dir, 'audio_checkpoints', model)
        XTTS_v2_model_path = os.path.join(folder_paths.models_dir, "audio_checkpoints\models--coqui--XTTS-v2")
        if model == 'Auto_DownLoad':
            if not os.access(os.path.join(XTTS_v2_model_path, "model.pth"), os.F_OK):
                model_path = 'coqui/XTTS-v2'
            else:
                model_path = XTTS_v2_model_path
                
        device = get_device_by_name(device)
        # dtype = get_dtype_by_name(dtype)
        
        audio_file = 'UL_audio'
        audio_path = os.path.join(output_dir, f'{audio_file}.wav')
        stereo_audio_path = os.path.join(output_dir, f'{audio_file}_stereo.wav')
        if apply_tts_api == False:
            if not is_module_imported('XttsConfig'):
                from TTS.tts.configs.xtts_config import XttsConfig
            if not is_module_imported('Xtts'):
                from TTS.tts.models.xtts import Xtts
            config = XttsConfig()
            config.load_json(os.path.join(model_path, 'config.json'))
            model = Xtts.init_from_config(config)
            model.load_checkpoint(
                                config, 
                                checkpoint_dir=model_path, 
                                eval=True, 
                                use_deepspeed=use_deepspeed,
                                )
            model.to(device)
            outputs = model.synthesize(
                    prompt,
                    config,
                    speaker_wav=ref_audio,
                    gpt_cond_len=gpt_cond_len,
                    language=language,
                    speed=speed,
                )
            torchaudio.save(audio_path, torch.tensor(outputs["wav"]).unsqueeze(0), 24000)
            model.to('cpu')
        else:
            from TTS.api import TTS
            config = os.path.join(model_path, 'config.json')
            tts = TTS(model_path=model_path, config_path=config, progress_bar=True)
            tts.to(device)
            # generate speech by cloning a voice using default settings
            tts.tts_to_file(text=prompt,
                            file_path=audio_path,
                            speaker_wav=ref_audio,
                            language=language,
                            emotion=tts_api_emotion,
                            speed=speed,
                            split_sentences=tts_api_split_sentences,
                            gpt_cond_len=gpt_cond_len,
                            )
            tts.to('cpu', torch.float32)
        
        # print("\033[93m原声：", ori_audio, "\n目标声", ref_audio, "\033[0m")
        if mono2stereo == True:
            os.system(
                f'ffmpeg -i "{audio_path}" -ac 2 "{stereo_audio_path}" -y'
            )
            audio_file = f'{audio_file}_stereo'
            audio_path = stereo_audio_path
        if advance_preview_only == True:
            result = {
                "filename": f'{audio_file}.wav',
                "subfolder": "",
                "type": "output",
                "prompt":"XTTS声音克隆",
                }
        else:
            result = audio_path
            
        return (result,)
        
class UL_Audio_UVR5:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        model_list = ["HP5_only_main_vocal.pth","HP5-主旋律人声vocals+其他instrumentals.pth","HP5_only_main_vocal.pth",
                      "HP2_all_vocals.pth","HP2-人声vocals+非人声instrumentals.pth","HP3_all_vocals.pth",
                      "VR-DeEchoAggressive.pth","VR-DeEchoDeReverb.pth","VR-DeEchoNormal.pth","onnx_dereverb_By_FoxJoy"]
        return {
            "required": {
                "advance_preview_only": ("BOOLEAN",{"default": False}),
                "audio": ("AUDIO_PATH",),
                "model": (model_list,{
                    "default": "HP5-主旋律人声vocals+其他instrumentals.pth"
                }),
                "agg":("INT",{
                    "default": 10, 
                    "min": 0, #Minimum value
                    "max": 20, #Maximum value
                    "step": 1, #Slider's step
                    "display": "slider"
                }),
                "tta": ("BOOLEAN",{"default": False}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
                "is_half": ("BOOLEAN",{"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO_PATH","AUDIO_PATH", )
    RETURN_NAMES = ("vocal_audio","bgm_audio", )
    FUNCTION = "UL_Audio_UVR5"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL Audio UVR5-人声、乐器声分离"
    
    def UL_Audio_UVR5(self, audio, model, agg, device, is_half, advance_preview_only, tta):
        device =get_device_by_name(device)
        
        audio = get_audio_from_video(audio)
        
        vocal_AUDIO, bgm_AUDIO = uvr5_split(self, audio, model, agg, device, is_half, advance_preview_only, tta)
        return (vocal_AUDIO,bgm_AUDIO, )
        
# FRCRN语音降噪-单麦-16k--https://www.modelscope.cn/models/iic/speech_frcrn_ans_cirm_16k
# 
# 模型局限性以及可能的偏差
# 模型在存在多说话人干扰声的场景噪声抑制性能有不同程度的下降。    
# DFSMN语音降噪-单麦-48k-实时近场--https://www.modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal 
# DFSMN语音降噪模型首次支持音视频会议48k采样率的语音降噪，目标是抑制目标说话人语音以外的其他各类环境噪声（比如啸叫，手机铃声，键盘声，背景人声，吃零食声，关门声等）。
# 模型局限性
# 本模型主要针对近距离麦克风采集语音的场景（小于0.5米）具有较理想的噪声抑制表现，并可以较好地保障较低的语音失真，如果进行远距离麦克风的语音降噪测试（大于0.5米），由于混响情况加重，本模型对混响语音处理能力有限，可能会产生一定程度的语音失真或小音量语音误消除情况。
# 本模型支持流式处理音频，需要在初始化pipeline时增加参数 stream_mode=True。 示例如下：
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
# ans = pipeline(
#     Tasks.acoustic_noise_suppression,
#     model='damo/speech_dfsmn_ans_psm_48k_causal',
#     stream_mode=True)
# with open(os.path.join(os.getcwd(), NOISE_SPEECH_FILE_48K_PCM), 'rb') as f:
#     block_size = 3840  # 模型每次处理的音频片段不能小于block_size
#     audio = f.read(block_size)
#     with open('output.pcm', 'wb') as w:
#         while len(audio) >= block_size:
#             result = ans(audio)
#             pcm = result[OutputKeys.OUTPUT_PCM]
#             w.write(pcm)
#             audio = f.read(block_size)
# 本模型也支持导出为ONNX格式：
# from modelscope.models import Model
# model = Model.from_pretrained('damo/speech_dfsmn_ans_psm_48k_causal')
# Exporter.from_model(model).export_onnx(output_dir=/your/output_dir)
class UL_Audio_noise_suppression:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        model_list = ["damo/speech_frcrn_ans_cirm_16k", "damo/speech_dfsmn_ans_psm_48k_causal"]
        return {
            "required": {
                "advance_preview_only": ("BOOLEAN",{"default": False}),
                "audio": ("AUDIO_PATH",),
                "model": (model_list,{
                    "default": "damo/speech_frcrn_ans_cirm_16k"
                }),
                "mono2stereo": ("BOOLEAN",{"default": False}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
            },
        }

    RETURN_TYPES = ("AUDIO_PATH",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "UL_Audio_noise_suppression"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL Audio noise_suppression-单麦16k/48k人声降噪"
    
    def UL_Audio_noise_suppression(self, audio, model, device, advance_preview_only, mono2stereo): 
        audio = get_audio_from_video(audio)
        device =get_device_by_name(device)
        audio_file = 'UL_audio.wav'
        output_audio_path = os.path.join(output_dir, audio_file)
        stereo_audio_path = os.path.join(output_dir, 'UL_audio_stereo.wav')
        noise_suppression(audio, output_audio_path, device, model)
        if model == 'damo/speech_frcrn_ans_cirm_16k':
            model_args = 'FRCRN语音降噪-单麦-16k'
        else:
            model_args = 'DFSMN语音降噪-单麦-48k-实时近场'
            
        if mono2stereo == True:
            os.system(
                f'ffmpeg -i "{output_audio_path}" -ac 2 "{stereo_audio_path}" -y'
            )
            audio_file = 'UL_audio_stereo.wav'
            output_audio_path = stereo_audio_path
            
        if advance_preview_only == True:
            result = {
                "filename": audio_file,
                "subfolder": "",
                "type": "output",
                "prompt":model_args,
                }
        else:
            result = output_audio_path
            
        return (result,)

NODE_CLASS_MAPPINGS = {
    "UL_Audio_Stable_Audio_Open": UL_Audio_stable_audio_open,
    "UL_Audio_facebook_musicgen": UL_Audio_facebook_musicgen,
    "UL_Audio_ChatTTS": UL_Audio_ChatTTS,
    "UL_Audio_OpenVoiceV2": UL_Audio_OpenVoiceV2, 
    "UL_Audio_XTTS": UL_Audio_XTTS, 
    "UL_Audio_UVR5": UL_Audio_UVR5, 
    "UL_Audio_noise_suppression": UL_Audio_noise_suppression, 
}