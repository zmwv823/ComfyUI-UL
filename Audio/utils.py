import os
import json
import time
import ffmpeg
import torch
import torchaudio
import shutil
import time
import folder_paths
import hashlib
from einops import rearrange
from ..UL_common.common import is_module_imported

current_directory = os.path.dirname(os.path.abspath(__file__))
input_path = folder_paths.get_input_directory()
comfy_temp_dir = folder_paths.get_temp_directory()
output_dir = folder_paths.get_output_directory()
# import tempfile
# sys_temp_dir = tempfile.gettempdir()

class UL_Audio_ChatTTS_Loader:
    @classmethod
    def INPUT_TYPES(s):
        models_list = os.listdir(os.path.join(folder_paths.models_dir, r"audio_checkpoints"))
        speakers_list = os.listdir(os.path.join(current_directory, "ChatTTS_Speakers"))
        models_list.insert(0, "Auto_DownLoad")
        speakers_list.insert(0, "None")
        return {"required": {
                "ChatTTS_model": (models_list, ),
                "speakers": (speakers_list, ),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
                "fix_saved_speaker_temperature":("BOOLEAN",{"default": False}),
                        }, 
            }
    
    RETURN_TYPES = ("UL_ChatTTS_Loader", )
    RETURN_NAMES = ("UL_ChatTTS_Loader", )
    FUNCTION = "UL_ChatTTS_Loader"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL ChatTTS Loader"
  
    def UL_ChatTTS_Loader(self, 
                          ChatTTS_model, #0
                          speakers, #1
                          device, #2
                          fix_saved_speaker_temperature, #3
                          ):
        self.ChatTTS_Loader = ChatTTS_model + '|' + speakers + '|' + device + '|' + str(fix_saved_speaker_temperature)
        return (self.ChatTTS_Loader, )

class UL_Audio_Preview_AutoPlay:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio_preview": ("AUDIO_PREVIEW",),
              }, 
                }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "UL_Preview_AutoPlay"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL Audio_Preview_AutoPlay"
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = ()
    OUTPUT_NODE = True
  
    def UL_Preview_AutoPlay(self,audio_preview):
        return {"ui": {"audio":[audio_preview]}}

# class UL_Audio_Preview_AutoPlay:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {
#                 "select_input": (["audio_preview", "audio", "vhs_audio"], {"default": "audio_preview"}),
#               }, 
#                  "optional": {
#                             "audio_preview": ("AUDIO_PREVIEW",),
#                             "audio": ("AUDIO", ),
#                             "vhs_audio": ("VHS_AUDIO", )
#                         },
#                 "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
#                 }
    
#     RETURN_TYPES = ()
#     RETURN_NAMES = ()
#     FUNCTION = "UL_Audio_Preview_AutoPlay"
#     CATEGORY = "ExtraModels/UL Audio"
#     TITLE = "UL Audio_Preview_AutoPlay"
#     INPUT_IS_LIST = False
#     OUTPUT_IS_LIST = ()
#     OUTPUT_NODE = True
  
#     def UL_Audio_Preview_AutoPlay(self, select_input, audio_preview="", audio="", vhs_audio="", prompt=None, extra_pnginfo=None):
#         audio_file = 'UL_audio_preview.wav'
#         audio_path = os.path.join(output_dir,  'audio', audio_file)
        
#         result = list()
#         result.append({
#                 "filename": audio_file,
#                 "subfolder": "audio",
#                 "type": "output",
#                 "prompt":"comfy_audio",
#                 })
#         result = result[0]
        
#         if select_input == 'audio_preview':
#             result = audio_preview
#         elif select_input == 'audio':
#             metadata = {}
#             disable_metadata = True
#             if not disable_metadata:
#                 if prompt is not None:
#                     metadata["prompt"] = json.dumps(prompt)
#                 if extra_pnginfo is not None:
#                     for x in extra_pnginfo:
#                         metadata[x] = json.dumps(extra_pnginfo[x])
#             for waveform in enumerate(audio["waveform"]):
#                 import io
#                 from comfy_extras.nodes_audio import insert_or_replace_vorbis_comment
#                 buff = io.BytesIO()
#                 buff = insert_or_replace_vorbis_comment(buff, metadata)
#                 torchaudio.save(buff, waveform[1], audio["sample_rate"], format="WAV")
#                 with open(audio_path, 'wb') as f:
#                     f.write(buff.getbuffer())
#         else:
#             with open(audio_path, 'wb') as f:
#                 f.write(vhs_audio())
#         # print(result)
#         return {"ui": {"audio":[result]}}

# class UL_Audio_Preview_noAutoPlay:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {
#                 "select_input": (["audio_preview", "audio", "vhs_audio"], {"default": "audio_preview"}),
#               }, 
#                  "optional": {
#                             "audio_preview": ("AUDIO_PREVIEW",),
#                             "audio": ("AUDIO", ),
#                             "vhs_audio": ("VHS_AUDIO", )
#                         },
#                 "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
#                 }
    
#     RETURN_TYPES = ()
#     RETURN_NAMES = ()
#     FUNCTION = "UL_Audio_Preview_noAutoPlay"
#     CATEGORY = "ExtraModels/UL Audio"
#     TITLE = "UL Audio_Preview_noAutoPlay"
#     INPUT_IS_LIST = False
#     OUTPUT_IS_LIST = ()
#     OUTPUT_NODE = True
  
#     def UL_Audio_Preview_noAutoPlay(self, select_input, audio_preview="", audio="", vhs_audio="", prompt=None, extra_pnginfo=None):
#         audio_file = 'UL_audio_preview.wav'
#         audio_path = os.path.join(output_dir,  'audio', audio_file)
        
#         result = list()
#         if select_input == 'audio':
#             prompt = 'Audio from comfy audio.'
#         if select_input == 'vhs_audio':
#             prompt = 'Audio from comfy vhs_audio.'
#         result.append({
#                 "filename": audio_file,
#                 "subfolder": "audio",
#                 "type": "output",
#                 "prompt":prompt,
#                 })
#         result = result[0]
        
#         if select_input == 'audio_preview':
#             result = audio_preview
#         elif select_input == 'audio':
#             # save audio from comfy_audio
#             metadata = {}
#             disable_metadata = True
#             if not disable_metadata:
#                 if prompt is not None:
#                     metadata["prompt"] = json.dumps(prompt)
#                 if extra_pnginfo is not None:
#                     for x in extra_pnginfo:
#                         metadata[x] = json.dumps(extra_pnginfo[x])
#             for waveform in enumerate(audio["waveform"]):
#                 import io
#                 from comfy_extras.nodes_audio import insert_or_replace_vorbis_comment
#                 buff = io.BytesIO()
#                 buff = insert_or_replace_vorbis_comment(buff, metadata)
#                 torchaudio.save(buff, waveform[1], audio["sample_rate"], format="WAV")
#                 with open(audio_path, 'wb') as f:
#                     f.write(buff.getbuffer())
#         else:
#             # save audio from vhs_audio
#             with open(audio_path, 'wb') as f:
#                 f.write(vhs_audio())
#         # print(result)
#         return {"ui": {"audio":[result]}}
        
class UL_Audio_Preview_noAutoPlay:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio_preview": ("AUDIO_PREVIEW",),
              }, 
                }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "UL_Preview_noAutoPlay"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL Audio_Preview_noAutoPlay"
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = ()
    OUTPUT_NODE = True
  
    def UL_Preview_noAutoPlay(self,audio_preview):
        return {"ui": {"audio":[audio_preview]}}

class UL_Load_Audio_or_Video:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["wav", "mp3", "flac", "m4a", "ogg",  "wma", "mp4", "mkv", "avi", "ts", "rm", "rmvb", "flv"]]
        return {"required":
                    {"audio": (sorted(files),)},
                    # "trim_audio": ("BOOLEAN",{"default": False}),
                    # "start_time": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
                    # "duration": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
                }

    CATEGORY = "ExtraModels/UL Audio"
    RETURN_NAMES = ("audio_path",)
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "UL_Load_Audio_or_Video"
    TITLE = "UL Load Audio_or_Video(Not comfy_tensor!!!)"

    def UL_Load_Audio_or_Video(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        return (audio_path,)

    @classmethod
    def IS_CHANGED(s, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(audio_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()
 
class UL_Audio_Stable_Audio_mask_args:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_cropfrom":("FLOAT", {"default": 0, "min": 0, "max": 100.0,"step": 0.1}),
                "mask_pastefrom":("FLOAT", {"default": 0, "min": 0, "max": 100.0,"step": 0.1}),
                "mask_pasteto":("FLOAT", {"default": 100, "min": 0, "max": 100.0,"step": 0.1}),
                "mask_maskstart":("FLOAT", {"default": 50, "min": 0, "max": 100.0,"step": 0.1}),
                "mask_maskend":("FLOAT", {"default": 100, "min": 0, "max": 100.0,"step": 0.1}),
                "mask_softnessL":("FLOAT", {"default": 0, "min": 0, "max": 100.0,"step": 0.1}),
                "mask_softnessR":("FLOAT", {"default": 0, "min": 0, "max": 100.0,"step": 0.1}),
                "mask_marination":("FLOAT", {"default": 0, "min": 0, "max": 1,"step": 0.001}),
            }
        }

    RETURN_TYPES = ("mask_args", )
    RETURN_NAMES = ("Stable_Audio_mask_args", )
    FUNCTION = "UL_Audio_Stable_Audio_mask_args"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL Audio Stable_Audio_mask_args"

    def UL_Audio_Stable_Audio_mask_args(self, mask_cropfrom, mask_pastefrom, mask_pasteto, mask_maskstart, mask_maskend, mask_softnessL, mask_softnessR, mask_marination):
        self.mask_args = str(mask_cropfrom) + '|' + str(mask_pastefrom) + '|' + str(mask_pasteto) + '|' + str(mask_maskstart) + '|' + str(mask_maskend) + '|' + str(mask_softnessL) + '|' + str(mask_softnessR) + '|' + str(mask_marination)
        return (self.mask_args, )
        
class UL_Audio_Convert_Audio2Wav:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "audio": ("AUDIO",),
                "mono2stereo": ("BOOLEAN",{"default": False}),
                "acodec": (["pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_f64le", ],{"default": "pcm_f32le"}), 
                "channels":("INT",{
                    "default": 2, 
                    "min": 1, #Minimum value
                    "max": 8, #Maximum value
                    "step": 1, #Slider's step
                    "display": "slider"
                }),
                "sample_rate": (["16000", "24000", "36000", "44100", "48000"],{"default": "48000"}), 
              }, 
                }
    
    RETURN_TYPES = ("AUDIO_PREVIEW", "AUDIO", )
    RETURN_NAMES = ("audio_preview", "audio", )
    FUNCTION = "UL_Audio_Convert_Audio2Wav"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL Audio_Convert_Audio2Wav"
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False, False, )
    OUTPUT_NODE = True
  
    def UL_Audio_Convert_Audio2Wav(self,audio, mono2stereo, acodec, channels, sample_rate):
        
        audio_path = audio_file_or_audio_tensor(audio)
        
        converted_audio_temp_path = os.path.join(comfy_temp_dir, f'UL_audio_converted_audio_{time.time()}.wav')
        preview_audio_path = os.path.join(output_dir, 'audio', 'UL_audio_converted_audio.wav')
        
        audio_path = convert_audio(mono2stereo, audio_path, acodec, converted_audio_temp_path, channels, sample_rate,)
        
        shutil.copy(converted_audio_temp_path, preview_audio_path)
        
        audio_tensor = audio_file2audio_tensor(converted_audio_temp_path)
        
        advance_preview = {
                "filename": 'UL_audio_converted_audio.wav',
                "subfolder": "audio",
                "type": "output",
                "prompt":'Convert audio or video into audio.',
                }
        
        return (advance_preview, audio_tensor, )
        
class UL_Audio_Trim_Audio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "audio": ("AUDIO",),
                "start_time": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
                "duration": ("FLOAT" , {"default": 9, "min": 0, "max": 10000000, "step": 0.01}),
              }, 
                }
    
    RETURN_TYPES = ("AUDIO_PREVIEW", "AUDIO", )
    RETURN_NAMES = ("audio_preview", "audio", )
    FUNCTION = "UL_Audio_Trim_Audio"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL Audio_Trim_Audio"
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False, False, )
    OUTPUT_NODE = True
  
    def UL_Audio_Trim_Audio(self,audio, start_time, duration):
        
        audio_path = audio_file_or_audio_tensor(audio)
        if not ('waveform' in audio and 'sample_rate' in audio):
            audio_path = convert_audio_or_video2wav(keep_info=True, acodec=None, input_audio=audio_path, channels=None, sample_rate=None)
        
        trimmed_audio_temp_path = os.path.join(comfy_temp_dir, f'UL_audio_trimmed_audio_{time.time()}.wav')
        preview_audio_path = os.path.join(output_dir, 'audio', 'UL_audio_trimmed_audio.wav')
        
        audio_path = trim_audio(audio_path, trimmed_audio_temp_path, start_time, duration)
        
        shutil.copy(audio_path, preview_audio_path)
        
        audio_tensor = audio_file2audio_tensor(trimmed_audio_temp_path)
        
        advance_preview = {
                "filename": 'UL_audio_trimmed_audio.wav',
                "subfolder": "audio",
                "type": "output",
                "prompt":'Convert audio or video into audio.',
                }
        
        return (advance_preview, audio_tensor, )
 
NODE_CLASS_MAPPINGS = {
    "UL_Audio_Preview_AutoPlay": UL_Audio_Preview_AutoPlay,
    "UL_Audio_Preview_noAutoPlay": UL_Audio_Preview_noAutoPlay,
    "UL_Audio_ChatTTS_Loader": UL_Audio_ChatTTS_Loader, 
    "UL_Load_Audio_or_Video": UL_Load_Audio_or_Video, 
    "UL_Audio_Stable_Audio_mask_args": UL_Audio_Stable_Audio_mask_args, 
    "UL_Audio_Convert_Audio2Wav": UL_Audio_Convert_Audio2Wav, 
    "UL_Audio_Trim_Audio": UL_Audio_Trim_Audio, 
}

# 加载模型
def stable_audio_open_load_model(device, model_path, dtype):
    stable_audio_open_model_config_file = os.path.join(current_directory, 'stable-audio-open-model-config\model_config.json')
    with open(stable_audio_open_model_config_file, encoding='utf-8') as f:
        model_config = json.load(f)
        # print(f'\033[93mModel config file(模型配置文件):', stable_audio_open_model_config_file, '\033[0m')
    if not is_module_imported('create_model_from_config'):
        from .stable_audio_tools.models.factory import create_model_from_config
    if not is_module_imported('load_ckpt_state_dict'):
        from .stable_audio_tools.models.utils import load_ckpt_state_dict
    model = create_model_from_config(model_config)
    model = model.to(device, dtype)
    model.load_state_dict(load_ckpt_state_dict(model_path))
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    
    return model,sample_rate,sample_size

def stable_audio_open_generate(model,prompt,seconds,seed,steps,cfg_scale,sample_size, sigma_min, sigma_max, sampler_type,device, init_audio, init_noise_level, sample_rate, mask_args):
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": seconds
    }]
    if not is_module_imported('generate_diffusion_cond'):
        from .stable_audio_tools.inference.generation import generate_diffusion_cond
    output = generate_diffusion_cond(
        model,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sample_rate=sample_rate,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        device=device,
        seed=seed,
        mask_args=mask_args,
    )
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    return output

def OpenVoiceV2_clone(converter_model_path, device, ori_voice_path, ref_voice_path, temp_dir, ouput_path, tau):
    if not is_module_imported('se_extractor'):
        from .OpenVoiceV2 import se_extractor
    if not is_module_imported('ToneColorConverter'):
        from .OpenVoiceV2.api import ToneColorConverter
    converter_model_path = converter_model_path
    tone_color_converter = ToneColorConverter(os.path.join(converter_model_path, 'checkpoints\converter\config.json'), device=device)
    model = tone_color_converter.load_ckpt(os.path.join(converter_model_path, 'checkpoints\converter\checkpoint.pth'))
    #input_voice_path from chattts
    source_se, audio_name = se_extractor.get_se(ori_voice_path, tone_color_converter, target_dir=temp_dir, vad=True)
    reference_speaker = ref_voice_path
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir=temp_dir, vad=True)
    tone_color_converter.convert(
        audio_src_path=ori_voice_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=ouput_path,
        tau=tau
        )
    del tone_color_converter
    del model

def Run_ChatTTS(text, prompt, rand_spk, model_local_path, device, temperature, top_P, top_K, use_decoder, refine_temperature, repetition_penalty, infer_max_new_token, refine_max_new_token, speed, save_name, skip_refine_text, speakers, save_speaker, do_text_normalization, fix_saved_speaker_temperature):
    if not is_module_imported('ChatTTS'):
        from . import ChatTTS
    chat = ChatTTS.Chat()
    chat.load_models(source='local', local_path=model_local_path,compile=False,device=device) # 设置为True以获得更快速度(其实刚好相反。。。)

    if rand_spk == None:
        rand_spk = chat.sample_random_speaker()
    else:
        if "None" not in speakers:
            rand_spk = torch.load(speakers)
            if fix_saved_speaker_temperature == 'True':
                temperature = 0.000000000001
                refine_temperature = 0.000000000001
            print("\033[93mUse Speaker(使用音色): ", speakers, "\033[0m")
    
    texts = [text,]

    params_refine_text = {
        'prompt': prompt,
        'temperature':refine_temperature,
        'top_P': top_P, # top P decode
        'top_K': top_K, 
        'repetition_penalty': repetition_penalty,
        'max_new_token':refine_max_new_token,
        }

    params_infer_code = {
    'spk_emb': rand_spk, # add sampled speaker 
    'temperature': temperature, # using custom temperature
    'top_P': top_P, # top P decode
    'top_K': top_K, # top K decode
    'repetition_penalty': repetition_penalty,
    'max_new_token':infer_max_new_token,
    'prompt':f'[speed_{speed}]',
    }

   
    # ChatTTS使用pynini对中英文进行处理，目前在window上安装报错，需要编译环境,
    # 暂时把do_text_normalization关掉
    wavs = chat.infer(texts, use_decoder=use_decoder,do_text_normalization=do_text_normalization,params_refine_text=params_refine_text,params_infer_code=params_infer_code, skip_refine_text=skip_refine_text)

    # print('#audio_path',folder_paths, )
    # 添加文件名后缀
    audio_file = f"UL_audio_ChatTTS_{time.time()}"
    audio_path = os.path.join(comfy_temp_dir, f"{audio_file}.wav")
    #保存音频文件，默认不带后缀，得手动添加。
    torchaudio.save(audio_path, torch.from_numpy(wavs[0]), 24000)
    
    preview_audio_path = os.path.join(output_dir, 'audio', 'UL_audio_ChatTTS.wav')
    shutil.copy(audio_path, preview_audio_path)
    audio_tensor = audio_file2audio_tensor(audio_path)
    
    if save_speaker == True:
        torch.save(rand_spk, os.path.join(current_directory, f'ChatTTS_Speakers\{save_name}.pt'))

    result = {
                "filename": 'UL_audio_ChatTTS.wav',
                "subfolder": "audio",
                "type": "output",
                "prompt":text,
                }
        
    return (result, rand_spk, audio_tensor, )

# uvr5
uvr5_weights = os.path.join(folder_paths.models_dir, r'audio_checkpoints\ExtraModels\uvr5')

def uvr5_split(self, audio, model,agg, device, is_half, tta):
        
        if not is_module_imported('hf_hub_download'):
            from huggingface_hub import hf_hub_download
        if model == "onnx_dereverb_By_FoxJoy":
            if not os.path.isfile(os.path.join(uvr5_weights,"uvr5_weights/onnx_dereverb_By_FoxJoy", "vocals.onnx")):
                hf_hub_download(
                    repo_id="lj1995/VoiceConversionWebUI",
                    filename="vocals.onnx",
                    subfolder= "uvr5_weights/onnx_dereverb_By_FoxJoy",
                    local_dir= uvr5_weights
                )
        else:
            if not os.path.isfile(os.path.join(uvr5_weights,"uvr5_weights", model)):
                hf_hub_download(
                    repo_id="lj1995/VoiceConversionWebUI",
                    filename=model,
                    subfolder= "uvr5_weights",
                    local_dir= uvr5_weights
                )
        
        # old_name = audio
        # new_name = os.path.join(comfy_temp_dir, f'UL_audio_{time.time()}.wav')
        # shutil.copy(old_name, new_name)
        # new_audio = new_name
        # save_root_vocal = comfy_temp_dir
        # save_root_ins = comfy_temp_dir
        # vocal_AUDIO,bgm_AUDIO, vocal_path, bgm_path = uvr5(model, new_audio, save_root_vocal,save_root_ins,agg, 'wav', device, is_half, tta)
        vocal_AUDIO,bgm_AUDIO, vocal_path, bgm_path = uvr5(model, audio, agg, 'wav', device, is_half, tta)
        return (vocal_AUDIO,bgm_AUDIO, vocal_path, bgm_path)

def uvr5(model_name, inp_root, agg, format0, device, is_half, tta):
    if not is_module_imported('MDXNetDereverb'):
        from .uvr5.mdxnet import MDXNetDereverb
    if not is_module_imported('AudioPre'):
        from .uvr5.vr import AudioPre
    if not is_module_imported('AudioPreDeEcho'):
        from .uvr5.vr import AudioPreDeEcho
        
    old_name = inp_root
    audio_file = f'UL_audio_{time.time()}'
    new_name = os.path.join(comfy_temp_dir, f'{audio_file}.wav')
    shutil.copy(old_name, new_name)
    inp_root = new_name
    save_root_vocal, save_root_ins = comfy_temp_dir, comfy_temp_dir
    
        
    vocal_AUDIO,bgm_AUDIO = "", ""
    inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    save_root_vocal = (
        save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    )
    save_root_ins = (
        save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    )
    is_hp3 = "HP3" in model_name
    if model_name == "onnx_dereverb_By_FoxJoy":
        pre_fun = MDXNetDereverb(15)
    else:
        func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
        pre_fun = func(
            agg=int(agg),
            model_path=os.path.join(uvr5_weights, "uvr5_weights",model_name),
            device=device,
            is_half=is_half,
            tta=tta, 
        )
    inp_path = inp_root
    need_reformat = 1
    done = 0
    
    info = ffmpeg.probe(inp_path, cmd="ffprobe")
    if (
        info["streams"][0]["channels"] == 2
        and info["streams"][0]["sample_rate"] == "44100"
    ):
        need_reformat = 0
        vocal_AUDIO,bgm_AUDIO = pre_fun._path_audio_(
            inp_path, save_root_ins, save_root_vocal, format0,is_hp3
        )
        done = 1
    else:
        need_reformat = 1
        
    if need_reformat == 1:
        src_audio = inp_path
        dsc_audio = os.path.join(comfy_temp_dir, 'input_audio_uvr5.wav')
        shutil.copy(src_audio, dsc_audio)
        
        tmp_audio_file = f'UL_audio_{time.time()}'
        tmp_path = os.path.join(comfy_temp_dir, f'{tmp_audio_file}.wav')
        (
        ffmpeg
        .input(dsc_audio)
        .output(tmp_path, acodec='pcm_s16le', ac=2, ar='44100')
        .overwrite_output()
        .run()
        )
        inp_path = tmp_path
    
    if done == 0:
        vocal_AUDIO,bgm_AUDIO = pre_fun._path_audio_(
            inp_path, save_root_ins, save_root_vocal, format0,is_hp3
        )
        print("%s->Success" % (os.path.basename(inp_path)))
    
    try:
        if model_name == "onnx_dereverb_By_FoxJoy":
            del pre_fun.pred.model
            del pre_fun.pred.model_
        else:
            del pre_fun.model
            del pre_fun
    except:
        pass
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    if need_reformat == 1:
        vocal_name = f'vocal_{tmp_audio_file}.wav_10.wav'
        bgm_name = f'instrument_{tmp_audio_file}.wav_10.wav'
    else:
        vocal_name = f'vocal_{audio_file}.wav_10.wav'
        bgm_name = f'instrument_{audio_file}.wav_10.wav'
    
    preview_vocal_path = os.path.join(output_dir, 'audio', vocal_name)
    preview_bgm_path = os.path.join(output_dir, 'audio', bgm_name)
    
    vocal_path = os.path.join(comfy_temp_dir, vocal_name)
    bgm_path = os.path.join(comfy_temp_dir, bgm_name)
    
    vocal_tensor = audio_file2audio_tensor(vocal_path)
    bgm_tensor = audio_file2audio_tensor(bgm_path)
    
    shutil.copy(vocal_path, preview_vocal_path)
    shutil.copy(bgm_path, preview_bgm_path)
    
    result_a = {
                "filename": vocal_name,
                "subfolder": "audio",
                "type": "output",
                "prompt":"人声",
                }
    result_b = {
                "filename": bgm_name,
                "subfolder": "audio",
                "type": "output",
                "prompt":"背景音",
                }
    return (result_a, result_b, vocal_tensor, bgm_tensor)

def noise_suppression(input_audio_path, output_audio_path, device, model):
    if not is_module_imported('pipeline_ali'):
        from modelscope.pipelines import pipeline as pipeline_ali
    if not is_module_imported('Tasks'):
        from modelscope.utils.constant import Tasks
    if model == 'damo/speech_frcrn_ans_cirm_16k':
        noise_suppression_model_path = os.path.join(folder_paths.models_dir, 'audio_checkpoints\ExtraModels\modelscope--damo--speech_frcrn_ans_cirm_16k')
        if not os.access(os.path.join(noise_suppression_model_path, 'pytorch_model.bin'), os.F_OK):
            noise_suppression_model_path = 'damo/speech_frcrn_ans_cirm_16k'
    elif model == 'damo/speech_dfsmn_ans_psm_48k_causal':
        noise_suppression_model_path = os.path.join(folder_paths.models_dir, 'audio_checkpoints\ExtraModels\modelscope--damo--speech_dfsmn_ans_psm_48k_causal')
        if not os.access(os.path.join(noise_suppression_model_path, 'pytorch_model.bin'), os.F_OK):
            noise_suppression_model_path = 'damo/speech_dfsmn_ans_psm_48k_causal'
            
        info = ffmpeg.probe(input_audio_path, cmd="ffprobe")
        tmp_path = os.path.join(comfy_temp_dir, f'UL_audio_denoised_48k_preprocess_{time.time()}.wav')
        if info["streams"][0]["sample_rate"] != "48000":
            # -i输入input， -vn表示vedio not，即输出不包含视频，-acodec重新音频编码，-ac 1单声道, -ac 2双声道, ar 48000采样率48khz, -y操作自动确认.
            # os.system(f'ffmpeg -i "{input_audio_path}" -vn -acodec pcm_s16le -ac 1 -ar 48000 "{tmp_path}" -y')
            tmp_path = convert_audio_or_video2wav(keep_info=False, acodec='pcm_s16le', input_audio=input_audio_path, channels=1, sample_rate='48000')
            input_audio_path = tmp_path
            
    ans = pipeline_ali(Tasks.acoustic_noise_suppression,model=noise_suppression_model_path, device=device)
    ans(input_audio_path,output_path=output_audio_path)
    
    del ans
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_audio_path

def get_audio_from_video(input_video_path):
    if not is_module_imported('VideoFileClip'):
        from moviepy.editor import VideoFileClip
    match = ['.mp3','.wav','.m4a','.ogg','.flac','wma']
    dirname, filename = os.path.split(input_video_path)
    file_name = str(filename).replace(".wav", "").replace(".mp3", "").replace(".m4a", "").replace(".ogg", "").replace(".flac", "").replace(".wma", "").replace(".mp4", "").replace(".mkv", "").replace(".flv", "").replace(".ts", "").replace(".rmvb", "").replace(".rm", "").replace(".avi", "")
    temp_audio = os.path.join(comfy_temp_dir, f'{file_name}_{time.time()}.wav')
    
    if not any(c in input_video_path for c in match):
        if '.avi' in input_video_path:
            # -i 表示input，即输入文件, -f 表示format，即输出格式, -vn表示video not，即输出不包含视频，注：输出位置不能覆盖原始文件(输入文件)。
            # os.system(f'ffmpeg -i "{input_video_path}" -f wav -vn {temp_audio} -y')
            temp_audio = convert_audio_or_video2wav(keep_info=True, acodec=None, input_audio=input_video_path, channels=None, sample_rate=None)
        else:
            # 读取视频文件
            video = VideoFileClip(input_video_path)
            # 提取视频文件中的声音
            audio = video.audio
            audio.write_audiofile(temp_audio)
        audio = temp_audio
    else:
        audio = input_video_path
    return audio
            
def audio_file2audio_tensor(audio_path):
    # convert audio file to audio_tensor
    print(audio_path)
    waveform, sample_rate = torchaudio.load(audio_path)
    multiplier = 1.0
    audio_tensor = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    return audio_tensor

def audio_file_or_audio_tensor(audio):
    if ('waveform' in audio and 'sample_rate' in audio):
        audio = audio_tensor2audio_file(audio)
    return audio

def audio_tensor2audio_file(audio_tensor):
    # save audio from comfy_audio
    save_audio_temp_dir = os.path.join(comfy_temp_dir, f'UL_audio_{time.time()}.wav')
    metadata = {}
    disable_metadata = True
    prompt, extra_pnginfo = None, None
    if not disable_metadata:
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])
    for waveform in enumerate(audio_tensor["waveform"]):
        import io
        from comfy_extras.nodes_audio import insert_or_replace_vorbis_comment
        buff = io.BytesIO()
        buff = insert_or_replace_vorbis_comment(buff, metadata)
        torchaudio.save(buff, waveform[1], audio_tensor["sample_rate"], format="WAV")
        with open(save_audio_temp_dir, 'wb') as f:
            f.write(buff.getbuffer())
    return(save_audio_temp_dir)

def convert_audio_or_video2wav(keep_info=None, acodec=None, input_audio=None, channels=None, sample_rate=None):
    output_audio = os.path.join(comfy_temp_dir, f'ffmpeg_converted_audio_{time.time()}.wav')
    if keep_info:
        ffmpeg.input(input_audio).output(output_audio).run()
    else:
        (
        ffmpeg
        .input(input_audio)
        .output(output_audio, acodec=acodec, ac=channels, ar=sample_rate)
        .overwrite_output()
        .run()
        )
    return output_audio

def convert_audio(mono2stereo, audio_input, acodec, audio_output, channels, sample_rate):
    """
    acodec:
        pcm_alaw ---> PCM A-law
        
        pcm_f32be ---> PCM 32-bit floating-point big-endian
        
        pcm_f32le ---> PCM 32-bit floating-point little-endian
        
        pcm_f64be ---> PCM 64-bit floating-point big-endian
        
        pcm_f64le ---> PCM 64-bit floating-point little-endian
        
        pcm_mulaw ---> PCM mu-law
        
        pcm_s16be ---> PCM signed 16-bit big-endian
        
        pcm_s16le ---> PCM signed 16-bit little-endian
        
        pcm_s24be ---> PCM signed 24-bit big-endian
        
        pcm_s24le ---> PCM signed 24-bit little-endian
        
        pcm_s32be ---> PCM signed 32-bit big-endian
        
        pcm_s32le ---> PCM signed 32-bit little-endian
        
        pcm_s8 ---> PCM signed 8-bit
        
        pcm_u16be ---> PCM unsigned 16-bit big-endian
        
        pcm_u16le ---> PCM unsigned 16-bit little-endian
        
        pcm_u24be ---> PCM unsigned 24-bit big-endian
        
        pcm_u24le ---> PCM unsigned 24-bit little-endian
        
        pcm_u32be ---> PCM unsigned 32-bit big-endian
        
        pcm_u32le ---> PCM unsigned 32-bit little-endian
        
        pcm_u8 ---> PCM unsigned 8-bit
    sample_rate:
        '16000'
        
        '24000'
        
        '32000'
        
        '44100'
        
        '48000'
    channels:
        1
        
        2
    FFmpeg can take input of raw audio types by specifying the type on the command line. For instance, to convert a "raw" audio type to a ".wav" file:
        ffmpeg -f s32le input_filename.raw output.wav
    You can specify number of channels, etc. as well, ex:
        ffmpeg -f u16le -ar 44100 -ac 1 -i input.raw output.wav
    The default for muxing into WAV files is pcm_s16le. You can change it by specifying the audio codec and using the WAV file extension:
        ffmpeg -i input -c:a pcm_s32le output.wav
    which will create a WAV file containing audio with that codec (not a raw file). There are also other containers that can contain raw audio packets, like pcm_bluray.

    If you want to create a raw file, don't use the WAV format, but the raw one (as seen in the table above), e.g. s16le, and the appropriate audio codec:
        ffmpeg -i input -f s16le -c:a pcm_s16le output.raw
    You can determine the format of a file, ex
        $ ffmpeg -i Downloads/BabyElephantWalk60.wav 
        ffmpeg version ...
        ...
        Input #0, wav, from 'Downloads/BabyElephantWalk60.wav':
            Duration: 00:01:00.00, bitrate: 352 kb/s
            Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 22050 Hz, mono, s16, 352 kb/s
    The pcm_s16le tells you what format your audio is in. And that happens to be a common format.
    """
    if mono2stereo == True:
        ffmpeg.input(audio_input).output(audio_output, ac=2).run()
    else:
        (
        ffmpeg
        .input(audio_input)
        .output(audio_output, acodec=acodec, ac=channels, ar=sample_rate)
        .overwrite_output()
        .run()
        )
        
def trim_audio(audio_input_path, audio_output_path, start_time, duration):
    if not is_module_imported('AudioSegment'):
        from pydub import AudioSegment
    sound = AudioSegment.from_wav(audio_input_path)
    start = start_time*1000
    end = (start_time + duration)*1000
    extract = sound[start:end]
    extract.export(audio_output_path, format="wav")
    return audio_output_path