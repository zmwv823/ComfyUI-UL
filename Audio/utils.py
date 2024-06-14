import os
import json
import torch
import torchaudio
import folder_paths
from einops import rearrange
from ..UL_common.common import is_module_imported, save_to_custom_folder_or_desktop

current_directory = os.path.dirname(os.path.abspath(__file__))

class UL_AudioPlay_AutoPlay:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio": ("AUDIO",),
              }, 
                }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "UL_AudioPlay_AutoPlay"
    CATEGORY = "ExtraModels/UL"
    TITLE = "UL AudioPlay_AutoPlay"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = ()

    OUTPUT_NODE = True
  
    def UL_AudioPlay_AutoPlay(self,audio):
        return {"ui": {"audio":[audio]}}
    
class UL_AudioPlay_noAutoPlay:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio": ("AUDIO",),
              }, 
                }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "UL_AudioPlay_noAutoPlay"
    CATEGORY = "ExtraModels/UL"
    TITLE = "UL AudioPlay noAutoPlay"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = ()

    OUTPUT_NODE = True
  
    def UL_AudioPlay_noAutoPlay(self,audio):
        return {"ui": {"audio":[audio]}}
    
NODE_CLASS_MAPPINGS = {
    "UL_AudioPlay_AutoPlay": UL_AudioPlay_AutoPlay,
    "UL_AudioPlay_noAutoPlay": UL_AudioPlay_noAutoPlay,
}

class UL_ChatTTS_Loader:
    @classmethod
    def INPUT_TYPES(s):
        models_list = os.listdir(os.path.join(folder_paths.models_dir, r"checkpoints\ex_ExtraModels"))
        speakers_list = os.listdir(os.path.join(current_directory, "ChatTTS_Speakers"))
        ref_audio_list = os.listdir(os.path.join(os.path.expanduser("~"), "Desktop", "ref_audio\OpenVoiceV2_ref_audio"))
        models_list.insert(0, "Auto_DownLoad")
        speakers_list.insert(0, "None")
        return {"required": {
                "ChatTTS_model": (models_list, ),
                "speakers": (speakers_list, ),
                "ref_audio": (ref_audio_list, ),
                "custom_folder": ("STRING", {"multiline": False, "default": r"C:\Users\pc\Desktop"}),
                "save_audio_to_custom_folder":("BOOLEAN", {"default": False}),
                "OpenVoice_model": (models_list, ),
                        }, 
            }
    
    RETURN_TYPES = ("UL_ChatTTS_Loader", )
    RETURN_NAMES = ("UL_ChatTTS_Loader", )
    FUNCTION = "UL_ChatTTS_Loader"
    CATEGORY = "ExtraModels/Audio"
    TITLE = "UL ChatTTS Loader"
  
    def UL_ChatTTS_Loader(self, 
                          ChatTTS_model, #0
                          speakers, #1
                          ref_audio, #2
                          custom_folder, #3
                          save_audio_to_custom_folder, #4
                          OpenVoice_model, #5
                          ):
        self.ChatTTS_Loader = ChatTTS_model + '|' + speakers + '|' + ref_audio + '|' + custom_folder + '|' + str(save_audio_to_custom_folder) + '|' + OpenVoice_model
        # text = self.ChatTTS_Loader
        return (self.ChatTTS_Loader, )
    
NODE_CLASS_MAPPINGS = {
    "UL_AudioPlay_AutoPlay": UL_AudioPlay_AutoPlay,
    "UL_AudioPlay_noAutoPlay": UL_AudioPlay_noAutoPlay,
    "UL_ChatTTS_Loader": UL_ChatTTS_Loader, 
}

# 加载模型
def stable_audio_open_load_model(device, model_path, dtype):
    stable_audio_open_model_config_file = os.path.join(current_directory, 'stable-audio-open-model-config\model_config.json')
    with open(stable_audio_open_model_config_file, encoding='utf-8') as f:
        model_config = json.load(f)
        print(f'\033[93mModel config file(模型配置文件):', stable_audio_open_model_config_file, '\033[0m')
    if not is_module_imported('create_model_from_config'):
        from .stable_audio_tools.models.factory import create_model_from_config
    if not is_module_imported('load_ckpt_state_dict'):
        from .stable_audio_tools.models.utils import load_ckpt_state_dict
    model = create_model_from_config(model_config)
    model.load_state_dict(load_ckpt_state_dict(model_path))
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    
    model = model.to(device, dtype)
    return model,sample_rate,sample_size

def stable_audio_open_generate(model,prompt,seconds,seed,steps,cfg_scale,sample_size, sigma_min, sigma_max, sampler_type,device):
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": seconds
    }]
    if not is_module_imported('generate_diffusion_cond'):
        from .stable_audio_tools.inference.generation import generate_diffusion_cond
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

def OpenVoiceV2_clone(converter_model_path, device, ori_voice_path, ref_voice_path, temp_dir, ouput_path, tau):
    from .OpenVoiceV2 import se_extractor
    from .OpenVoiceV2.api import ToneColorConverter
    converter_model_path = converter_model_path
    tone_color_converter = ToneColorConverter(os.path.join(converter_model_path, 'checkpoints\converter\config.json'), device=device)
    tone_color_converter.load_ckpt(os.path.join(converter_model_path, 'checkpoints\converter\checkpoint.pth'))
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
    print("test")
    return

def Run_ChatTTS(text, prompt, rand_spk, model_local_path, device, temperature, top_P, top_K, use_decoder, refine_temperature, repetition_penalty, infer_max_new_token, refine_max_new_token, speed, save_to_desktop, save_to_custom_folder, save_name, custom_folder, skip_refine_text, speakers, save_speaker, mono2stereo, do_text_normalization, OpenVoice_clone, ref_audio_path, tau, OpenVoice_model):
    import ChatTTS
    chat = ChatTTS.Chat()
    chat.load_models(local_path=model_local_path,compile=False,device=device) # 设置为True以获得更快速度(其实刚好相反。。。)

    if rand_spk == None:
        rand_spk = chat.sample_random_speaker()
    else:
        if "None" not in speakers:
            rand_spk = torch.load(speakers)
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

    
    output_dir = folder_paths.get_output_directory()
    # print('#audio_path',folder_paths, )
    # 添加文件名后缀
    audio_file = "ChatTTS.wav"
    audio_path = os.path.join(output_dir, "ChatTTS")
    #保存音频文件，默认不带后缀，得手动添加。
    torchaudio.save(f'{audio_path}.wav', torch.from_numpy(wavs[0]), 24000)
    if mono2stereo == True:
        import subprocess
        subprocess.run(["powershell", f'ffmpeg -i {audio_path}.wav -ac 2 {audio_path}_stereo.wav -y'], shell=True) # 使用子模块调用powershell运行ffmpeg将单声道转为双声道，并保存为ChatTTS_stereo.wav。
        audio_file = "ChatTTS_stereo.wav"
        audio_path = os.path.join(output_dir, f'{audio_path}_stereo')
    
    if save_speaker == True:
        torch.save(rand_spk, os.path.join(current_directory, f'ChatTTS_Speakers\{save_name}.pth'))
        
    if OpenVoice_clone == True:
        # convertor_path = os.path.join(folder_paths.models_dir, "checkpoints\ex_ExtraModels\models--myshell-ai--OpenVoiceV2")
        ori_voice_path = audio_path
        ref_voice_path = ref_audio_path
        temp_dir = folder_paths.get_temp_directory()
        output_path = f'{audio_path}_clone.wav'
        audio_file = f'{audio_path}_clone.wav'
        OpenVoiceV2_clone(OpenVoice_model, device, f'{ori_voice_path}.wav', ref_voice_path, temp_dir, output_path, tau)
        audio_path = f'{audio_path}_clone'
        
    save_to_custom_folder_or_desktop(f'{audio_path}.wav', save_to_desktop, save_to_custom_folder, save_name, custom_folder)

    return ({
                "filename": audio_file,
                "subfolder": "",
                "type": "output",
                "prompt":text,
                "audio_path":audio_path
                },rand_spk)