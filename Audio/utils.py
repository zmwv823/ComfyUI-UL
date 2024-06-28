import os
import json
import torch
import torchaudio
import shutil
import tempfile
import folder_paths
import hashlib
from einops import rearrange
from ..UL_common.common import is_module_imported

current_directory = os.path.dirname(os.path.abspath(__file__))
input_path = folder_paths.get_input_directory()
temp_dir = folder_paths.get_temp_directory()
output_dir = folder_paths.get_output_directory()
sys_temp_dir = tempfile.gettempdir()

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

class UL_Advance_AutoPlay:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio_preview": ("AUDIO_PREVIEW",),
              }, 
                }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "UL_Advance_AutoPlay"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL Advance_AutoPlay"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = ()

    OUTPUT_NODE = True
  
    def UL_Advance_AutoPlay(self,audio_preview):
        return {"ui": {"audio":[audio_preview]}}
        
class UL_VAEDecodeAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "advance_preview_only": ("BOOLEAN",{"default": False}),
            "samples": ("LATENT", ), 
            "vae": ("VAE", ),
            }}
    
    RETURN_TYPES = ("AUDIO_PATH",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "UL_VAEDecodeAudio"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL VAEDecodeAudio"

    def UL_VAEDecodeAudio(self, vae, samples, advance_preview_only):
        audio_file = 'UL_audio.wav'
        audio_path = os.path.join(output_dir, audio_file)
        audio = vae.decode(samples["samples"]).movedim(-1, 1)
        sample_rate = 44100
        torchaudio.save(audio_path, audio[0], sample_rate)
        if advance_preview_only == True:
            result = {
                "filename": audio_file,
                "subfolder": "",
                "type": "output",
                "prompt":"comfy_audio",
                }
        else:
            result = audio_path
            
        return (result,)
        
    
class UL_Advance_noAutoPlay:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio_preview": ("AUDIO_PREVIEW",),
              }, 
                }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "UL_Advance_noAutoPlay"
    CATEGORY = "ExtraModels/UL Audio"
    TITLE = "UL Advance_noAutoPlay"
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = ()
    OUTPUT_NODE = True
  
    def UL_Advance_noAutoPlay(self,audio_preview):
        return {"ui": {"audio":[audio_preview]}}

class UL_Load_Audio:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["wav", "mp3", "flac", "m4a", "ogg", "mp4", "mkv", "avi", "ts", "rm", "rmvb", "flv"]]
        return {"required":
                    {"audio": (sorted(files),)},
                    # "part_load": ("BOOLEAN",{"default": False}),
                    # "start_time": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
                    # "duration": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
                }

    CATEGORY = "ExtraModels/UL Audio"
    RETURN_NAMES = ("audio_path",)
    RETURN_TYPES = ("AUDIO_PATH",)
    FUNCTION = "UL_load_audio"
    TITLE = "UL Load Audio"

    def UL_load_audio(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        return (audio_path,)

    @classmethod
    def IS_CHANGED(s, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(audio_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()
    

# class UL_PreView_Audio:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required":
#                     {"audio_path": ("AUDIO_PATH",),}
#                 }

#     CATEGORY = "ExtraModels/UL Audio"
#     TITLE = "UL PreView Audio"
#     FUNCTION = "UL_PreView_Audio"
#     RETURN_TYPES = ()
#     RETURN_NAMES = ()
#     OUTPUT_NODE = True

#     def UL_PreView_Audio(self, audio_path):
#         audio_name = os.path.basename(audio_path)
#         tmp_path = os.path.dirname(audio_path)
#         audio_root = os.path.basename(tmp_path)
#         return {"ui": {"audio":[audio_name,audio_root]}}
 
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
 
NODE_CLASS_MAPPINGS = {
    "UL_Advance_AutoPlay": UL_Advance_AutoPlay,
    "UL_Advance_noAutoPlay": UL_Advance_noAutoPlay,
    "UL_Audio_ChatTTS_Loader": UL_Audio_ChatTTS_Loader, 
    "UL_Load_Audio": UL_Load_Audio, 
    # "UL_PreView_Audio": UL_PreView_Audio, 
    "UL_Audio_Stable_Audio_mask_args": UL_Audio_Stable_Audio_mask_args, 
    "UL_VAEDecodeAudio": UL_VAEDecodeAudio, 
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

    
    output_dir = folder_paths.get_output_directory()
    # print('#audio_path',folder_paths, )
    # 添加文件名后缀
    audio_file = "UL_audio"
    audio_path = os.path.join(output_dir, "UL_audio.wav")
    #保存音频文件，默认不带后缀，得手动添加。
    torchaudio.save(audio_path, torch.from_numpy(wavs[0]), 24000)
    
    if save_speaker == True:
        torch.save(rand_spk, os.path.join(current_directory, f'ChatTTS_Speakers\{save_name}.pt'))

    result = {
                "filename": f'{audio_file}.wav',
                "subfolder": "",
                "type": "output",
                "prompt":text,
                }
        
    return (result, rand_spk, audio_path)

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
        
        old_name = audio
        new_name = os.path.join(sys_temp_dir, 'UL_audio.wav')
        shutil.copy(old_name, new_name)
        new_audio = new_name
        save_root_vocal = output_dir
        save_root_ins = output_dir
        vocal_AUDIO,bgm_AUDIO, vocal_path, bgm_path = uvr5(model, new_audio, save_root_vocal,save_root_ins,agg, 'wav', device, is_half, tta)
        return (vocal_AUDIO,bgm_AUDIO, vocal_path, bgm_path)

def uvr5(model_name, inp_root, save_root_vocal,save_root_ins, agg, format0, device, is_half, tta):
    import ffmpeg
    if not is_module_imported('MDXNetDereverb'):
        from .uvr5.mdxnet import MDXNetDereverb
    if not is_module_imported('AudioPre'):
        from .uvr5.vr import AudioPre
    if not is_module_imported('AudioPreDeEcho'):
        from .uvr5.vr import AudioPreDeEcho
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
        dsc_audio = os.path.join(sys_temp_dir, 'input_audio.wav')
        shutil.copy(src_audio, dsc_audio)
        # tmp_path = "%s/%s.reformatted.wav" % (
        #     input_path,
        #     os.path.basename(inp_path),
        # )
        tmp_path = os.path.join(sys_temp_dir, 'inputUL_audio.wav')
        os.system(
            f'ffmpeg -i "{dsc_audio}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y'
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
    print("clean_empty_cache")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if need_reformat == 1:
        vocal_name = 'vocal_inputUL_audio.wav_10.wav'
        bgm_name = 'instrument_inputUL_audio.wav_10.wav'
    else:
        vocal_name = 'vocal_UL_audio.wav_10.wav'
        bgm_name = 'instrument_UL_audio.wav_10.wav'
    vocal_path = os.path.join(output_dir,  vocal_name)
    bgm_path = os.path.join(output_dir,  bgm_name)
    result_a = {
                "filename": vocal_name,
                "subfolder": "",
                "type": "output",
                "prompt":"人声",
                }
    result_b = {
                "filename": bgm_name,
                "subfolder": "",
                "type": "output",
                "prompt":"背景音",
                }
    return (result_a, result_b, vocal_path, bgm_path)

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
            
        import ffmpeg
        info = ffmpeg.probe(input_audio_path, cmd="ffprobe")
        tmp_path = os.path.join(sys_temp_dir, 'UL_audio_denoised_48k_preprocess.wav')
        if info["streams"][0]["sample_rate"] != "48000":
            # -i输入input， -vn表示vedio not，即输出不包含视频，-acodec重新音频编码，-ac 1单声道, -ac 2双声道, ar 48000采样率48khz, -y操作自动确认.
            os.system(f'ffmpeg -i "{input_audio_path}" -vn -acodec pcm_s16le -ac 1 -ar 48000 "{tmp_path}" -y')
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
    match = ['.mp3','.wav','.m4a','.ogg','.flac']
    dirname, filename = os.path.split(input_video_path)
    file_name = str(filename).replace(".wav", "").replace(".mp3", "").replace(".m4a", "").replace(".ogg", "").replace(".flac", "").replace(".mp4", "").replace(".mkv", "").replace(".flv", "").replace(".ts", "").replace(".rmvb", "").replace(".rm", "").replace(".avi", "")
    temp_audio = os.path.join(sys_temp_dir, f'{file_name}.wav')
    
    if not any(c in input_video_path for c in match):
        if '.avi' in input_video_path:
            # -i 表示input，即输入文件, -f 表示format，即输出格式, -vn表示video not，即输出不包含视频，注：输出位置不能覆盖原始文件(输入文件)。
            os.system(f'ffmpeg -i "{input_video_path}" -f wav -vn {temp_audio} -y')
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