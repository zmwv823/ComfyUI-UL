import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import folder_paths
import time
import torch
from ..UL_common.common import get_device_by_name, get_dtype_by_name, is_module_imported
from .utils import nlp_csanmt_translation_zh2en, SavedModel_Translator, t5_translate_en_ru_zh, nllb_200_translator
# , write_to_result
from ..Audio.utils import get_audio_from_video, audio_file_or_audio_tensor

current_directory = os.path.dirname(os.path.abspath(__file__))
temp_txt_path = os.path.join(current_directory, "temp_dir", "Translation.txt")
temp_Summarization_path = os.path.join(current_directory, "temp_dir", "Summarization.txt")

class UL_DataProcess_t5_translate_en_ru_zh:
  
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "translator": (["utrobinmv/t5_translate_en_ru_zh_small_1024", "utrobinmv/t5_translate_en_ru_zh_base_200", "utrobinmv/t5_translate_en_ru_zh_large_1024"],{"default": "utrobinmv/t5_translate_en_ru_zh_base_200"}), 
                "prompt": ("STRING", {"default": "Цель разработки — предоставить пользователям личного синхронного переводчика.", "multiline": True}),
                "Batch_prompt": ("STRING", {"default": "这里是多批次翻译文本输入，使用换行进行分割。\n天上掉馅饼啦，快去看超人！！！\n飞流直下三千尺，疑似银河落九天。\n启用Batch_Newline表示输出的翻译会按换行输入进行二次换行,否则是用空格合并起来的整篇文本。", "multiline": True}),
                "Batch_Newline" :("BOOLEAN", {"default": True}),
                "if_Batch": ("BOOLEAN", {"default": False}),
                "Target_Language": (["en", "zh", "ru", ],{"default": "en"}), 
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    CATEGORY = "ExtraModels/UL DataProcess"
    FUNCTION = "UL_DataProcess_t5_translate_en_ru_zh"
    TITLE = "UL DataProcess t5_translate_en_ru_zh"

    def UL_DataProcess_t5_translate_en_ru_zh(self, translator, Target_Language, prompt, device, Batch_prompt, Batch_Newline, if_Batch):
        device = get_device_by_name(device)
        # 使用换行(\n)作为分隔符
        Batch_prompt = Batch_prompt.split("\n")  
        if if_Batch == True:
            input_sequence = Batch_prompt
            # 用特定的连接符<SENT_SPLIT>，将多个句子进行串联
            input_sequence = '|'.join(input_sequence)
        else:
            input_sequence = prompt
        if translator == 'utrobinmv/t5_translate_en_ru_zh_large_1024':
            zh2en_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_large_1024")
            if not os.access(os.path.join(zh2en_path, "model.safetensors"), os.F_OK):
                zh2en_path = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
            outputs = t5_translate_en_ru_zh(Target_Language, input_sequence, zh2en_path, device)[0]
            
        elif translator == 'utrobinmv/t5_translate_en_ru_zh_base_200':
            zh2en_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_base_200")
            if not os.access(os.path.join(zh2en_path, "model.safetensors"), os.F_OK):
                zh2en_path = 'utrobinmv/t5_translate_en_ru_zh_base_200'
            outputs = t5_translate_en_ru_zh(Target_Language, input_sequence, zh2en_path, device)[0]
            
        elif translator == 'utrobinmv/t5_translate_en_ru_zh_small_1024':
            zh2en_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_small_1024")
            if not os.access(os.path.join(zh2en_path, "model.safetensors"), os.F_OK):
                zh2en_path = 'utrobinmv/t5_translate_en_ru_zh_small_1024'
            outputs = t5_translate_en_ru_zh(Target_Language, input_sequence, zh2en_path, device)[0]
            
        if if_Batch == True:
            results = outputs.split('| ')
            if Batch_Newline == True:
                results = '\n\n'.join(results)
                if translator == 'utrobinmv/t5_translate_en_ru_zh_large_1024':
                    results = results.replace('# ', '\n\n')
            else:
                results = ' '.join(results)
        else:
            results = outputs
        with open(temp_txt_path, "w", encoding="UTF-8") as text_file:
            text_file.write(results)
        # model.to('cpu', torch.float32)
        return (results, )

class UL_DataProcess_nlp_csanmt_translation_zh2en_translator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (['Saved_Model_zh2en', 'damo/nlp_csanmt_translation_zh2en', 'damo/nlp_csanmt_translation_en2zh_base', 'Saved_Model_en2zh_base', 'Saved_Model_en2zh'] , {"default": "damo/nlp_csanmt_translation_zh2en"}),
                "prompt": ("STRING", {"default": "这里是单批次翻译文本输入。\n声明补充说，沃伦的同事都深感震惊，并且希望他能够投案自首。\n尽量输入单句文本，如果是多句长文本建议人工分句，否则可能出现漏译或未译等情况！！！\n建议换行，效果更佳。", "multiline": True}),
                "Batch_prompt": ("STRING", {"default": "这里是多批次翻译文本输入，使用换行进行分割。\n天上掉馅饼啦，快去看超人！！！\n飞流直下三千尺，疑似银河落九天。\n启用Batch_Newline表示输出的翻译会按换行输入进行二次换行,否则是用空格合并起来的整篇文本。", "multiline": True}),
                "Batch_Newline" :("BOOLEAN", {"default": True}),
                "if_Batch": ("BOOLEAN", {"default": False}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    CATEGORY = "ExtraModels/UL DataProcess"
    FUNCTION = "UL_DataProcess_nlp_csanmt_translation_zh2en_translator"
    TITLE = "UL DataProcess-nlp阿里达摩院中英互译"

    def UL_DataProcess_nlp_csanmt_translation_zh2en_translator(self, prompt, Batch_prompt, if_Batch, device, Batch_Newline, model):
        device = get_device_by_name(device)
        # 使用换行(\n)作为分隔符
        Batch_prompt = Batch_prompt.split("\n")  
        if if_Batch == True:
            input_sequence = Batch_prompt
            # 用特定的连接符<SENT_SPLIT>，将多个句子进行串联
            input_sequence = '<SENT_SPLIT>'.join(input_sequence)
        else:
            input_sequence = prompt
            
        if 'Saved' not in model:
            if model == 'damo/nlp_csanmt_translation_zh2en':
                nlp_path = os.path.join(folder_paths.models_dir, r'prompt_generator\modelscope--damo--nlp_csanmt_translation_zh2en')
                if not os.access(os.path.join(nlp_path, "tf_ckpts", "ckpt-0.data-00000-of-00001"), os.F_OK):
                    if not is_module_imported('snapshot_download'):
                        from modelscope.hub.snapshot_download import snapshot_download
                    snapshot_download('damo/nlp_csanmt_translation_zh2en')
                    nlp_path = 'damo/nlp_csanmt_translation_zh2en'
                    
            if model == 'damo/nlp_csanmt_translation_en2zh':
                nlp_path = os.path.join(folder_paths.models_dir, r'prompt_generator\modelscope--damo--nlp_csanmt_translation_en2zh')
                if not os.access(os.path.join(nlp_path, "tf_ckpts", "ckpt-0.data-00000-of-00001"), os.F_OK):
                    if not is_module_imported('snapshot_download'):
                        from modelscope.hub.snapshot_download import snapshot_download
                    snapshot_download('damo/nlp_csanmt_translation_en2zh')
                    nlp_path = 'damo/nlp_csanmt_translation_en2zh'
                    
            if model == 'damo/nlp_csanmt_translation_en2zh_base':
                nlp_path = os.path.join(folder_paths.models_dir, r'prompt_generator\modelscope--damo--nlp_csanmt_translation_en2zh_base')
                if not os.access(os.path.join(nlp_path, "tf_ckpts", "ckpt-0.data-00000-of-00001"), os.F_OK):
                    if not is_module_imported('snapshot_download'):
                        from modelscope.hub.snapshot_download import snapshot_download
                    snapshot_download('damo/nlp_csanmt_translation_en2zh_base')
                    nlp_path = 'damo/nlp_csanmt_translation_en2zh_base'
                    
            outputs = nlp_csanmt_translation_zh2en(device, input_sequence, nlp_path)['translation']
        else:
            outputs = SavedModel_Translator(input_sequence, model)['translation']
            
        if if_Batch == True:
            results = outputs.split('<SENT_SPLIT>')
            if Batch_Newline == True:
                results = '\n\n'.join(results)
            else:
                results = ' '.join(results)
        else:
            results = outputs
        with open(temp_txt_path, "w", encoding="UTF-8") as text_file:
            text_file.write(results)
        return (results, )
        
class UL_DataProcess_nllb_200_translator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (['facebook/nllb-200-distilled-1.3B', 'facebook/nllb-200-distilled-600M', 'facebook/nllb-200-3.3B'] , {"default": "facebook/nllb-200-distilled-600M"}),
                "prompt": ("STRING", {"default": "这里是单批次翻译文本输入。\n声明补充说，沃伦的同事都深感震惊，并且希望他能够投案自首。\n尽量输入单句文本，如果是多句长文本建议人工分句，否则可能出现漏译或未译等情况！！！\n建议换行，效果更佳。", "multiline": True}),
                "Batch_prompt": ("STRING", {"default": "这里是多批次翻译文本输入，使用换行进行分割。\n天上掉馅饼啦，快去看超人！！！\n飞流直下三千尺，疑似银河落九天。\n启用Batch_Newline表示输出的翻译会按换行输入进行二次换行,否则是用空格合并起来的整篇文本。", "multiline": True}),
                "Source_Language": (["Chinese (Simplified)", "English", "Chinese (Traditional)", "Spanish", "French", "German", "Italian", "Portuguese", "Polish", "Turkish", "Russian", "Dutch", "Czech", "Modern Standard Arabic", "Japanese", "Hungarian", "Korean", "Hindi", "Yue Chinese", "Standard Tibetan", "Halh Mongolian", "Uyghur", ],{"default": "Chinese (Simplified)"}), 
                "Target_Language": (["Chinese (Simplified)", "English", "Chinese (Traditional)", "Spanish", "French", "German", "Italian", "Portuguese", "Polish", "Turkish", "Russian", "Dutch", "Czech", "Modern Standard Arabic", "Japanese", "Hungarian", "Korean", "Hindi", "Yue Chinese", "Standard Tibetan", "Halh Mongolian", "Uyghur", ],{"default": "English"}), 
                "if_Batch": ("BOOLEAN", {"default": False}),
                "Batch_Newline" :("BOOLEAN", {"default": True}),
                "max_length": ("INT", {"default": 400, "min": 0, "max": 0xffffffffffffffff}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
                "dtype": (["auto","fp16","bf16","fp32"],{"default":"auto"}),
                "use_fast": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    CATEGORY = "ExtraModels/UL DataProcess"
    FUNCTION = "UL_DataProcess_nllb_200_translator"
    TITLE = "UL DataProcess nllb_200_translator"

    def UL_DataProcess_nllb_200_translator(self, prompt, Batch_prompt, if_Batch, device, Batch_Newline, model, dtype, Source_Language, Target_Language, use_fast, max_length):
        device = get_device_by_name(device)
        dtype = get_dtype_by_name(dtype)
        
        # 使用换行(\n)作为分隔符
        Batch_prompt = Batch_prompt.split("\n")  
        if if_Batch == True:
            input_sequence = Batch_prompt
            # 用特定的连接符<SENT_SPLIT>，将多个句子进行串联
            input_sequence = '<SENT_SPLIT>'.join(input_sequence)
        else:
            input_sequence = prompt
            
        outputs = nllb_200_translator(device, dtype, input_sequence, Source_Language, Target_Language, use_fast, model, max_length)
            
        if if_Batch == True:
            results = outputs.split('<SENT_SPLIT> ')
            if Batch_Newline == True:
                results = '\n\n'.join(results)
            else:
                results = ' '.join(results)
        else:
            results = outputs
        with open(temp_txt_path, "w", encoding="UTF-8") as text_file:
            text_file.write(results)
        return (results, )

class UL_DataProcess_Summarization:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data_path": ("DATA_PATH", ),
                "use_data": ("BOOLEAN", {"default": False}),
                "model": (['utrobinmv/t5_summary_en_ru_zh_base_2048', 'csebuetnlp/mT5_multilingual_XLSum', 'csebuetnlp/mT5_m2o_chinese_simplified_crossSum'] , {"default": "csebuetnlp/mT5_multilingual_XLSum"}),
                "prompt": ("STRING", {"default": "Videos that say approved vaccines are dangerous and cause autism, cancer or infertility are among those that will be taken down, the company said.  The policy includes the termination of accounts of anti-vaccine influencers.  Tech giants have been criticised for not doing more to counter false health information on their sites.  In July, US President Joe Biden said social media platforms were largely responsible for people's scepticism in getting vaccinated by spreading misinformation, and appealed for them to address the issue.  YouTube, which is owned by Google, said 130,000 videos were removed from its platform since last year, when it implemented a ban on content spreading misinformation about Covid vaccines.  In a blog post, the company said it had seen false claims about Covid jabs 'spill over into misinformation about vaccines in general'. The new policy covers long-approved vaccines, such as those against measles or hepatitis B.  'We're expanding our medical misinformation policies on YouTube with new guidelines on currently administered vaccines that are approved and confirmed to be safe and effective by local health authorities and the WHO,' the post said, referring to the World Health Organization.", "multiline": True}),
                "summary_type": (["short", "brief", "big", ],{"default": "short"}), 
                "target_language": (["source", "zh", "zh-cn", "en", "ru", "es", "ja", "ko", "fr", "de", "it", "pt", "tr", "nl", "cs", "ar", "hu", "hi", "pl"],{"default": "zh"}), 
                "max_length": ("INT", {"default": 512, "min": 512, "max": 0xffffffffffffffff, "step": 1}),
                "output_max_length": ("INT", {"default": 84, "min": 1, "max": 0xffffffffffffffff, "step": 1}),
                "num_beams": ("INT", {"default": 4, "min": 1, "max": 999, "step": 1}),
                "no_repeat_ngram_size": ("INT", {"default": 2, "min": 1, "max": 999, "step": 1}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
                "use_T5": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    CATEGORY = "ExtraModels/UL DataProcess"
    FUNCTION = "UL_DataProcess_Summarization"
    TITLE = "UL DataProcess Summarization"

    def UL_DataProcess_Summarization(self, prompt, device, model, summary_type, target_language, max_length, num_beams, no_repeat_ngram_size, output_max_length, use_T5, data_path, use_data):
        device = get_device_by_name(device)
        if model == "utrobinmv/t5_summary_en_ru_zh_base_2048":
            Summarizer_path = os.path.join(folder_paths.models_dir, r'prompt_generator\models--utrobinmv--t5_summary_en_ru_zh_base_2048')
            if not os.access(os.path.join(Summarizer_path, "model.safetensors"), os.F_OK):
                Summarizer_path = model
                

        if model == "csebuetnlp/mT5_multilingual_XLSum":
            # Summarizer_path = os.path.join(os.path.expanduser("~"), "Desktop", "models--csebuetnlp--mT5_multilingual_XLSum")
            Summarizer_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--csebuetnlp--mT5_multilingual_XLSum")
            if not os.access(os.path.join(Summarizer_path, "pytorch_model.bin"), os.F_OK):
                Summarizer_path = model

        if model == "csebuetnlp/mT5_m2o_chinese_simplified_crossSum":
            Summarizer_path = os.path.join(folder_paths.models_dir, r'prompt_generator\models--csebuetnlp--mT5_m2o_chinese_simplified_crossSum')
            if not os.access(os.path.join(Summarizer_path, "pytorch_model.bin"), os.F_OK):
                Summarizer_path = model
        
        if use_data == True:
            if "txt" in data_path:
                with open(data_path, "r", encoding="UTF-8") as f:  
                    prompt = f.read()
            elif "pdf" in data_path:
                import fitz
                with fitz.open(data_path) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()
                        prompt = text
            elif "srt" in data_path:
                with open(data_path, "r", encoding="UTF-8") as f:  
                    prompt = f.read()
            elif "ass" in data_path:
                with open(data_path, "r", encoding="UTF-8") as f:  
                    prompt = f.read()
                
        if use_T5 == False:
            import re
            if not is_module_imported('AutoTokenizer'):
                from transformers import AutoTokenizer
            if not is_module_imported('AutoModelForSeq2SeqLM'):
                from transformers import AutoModelForSeq2SeqLM
            WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
            if summary_type == "short":
                s_type = ''
            else:
                s_type = summary_type
            prefix = f'summary {s_type} to {target_language}: '
            if target_language == "source":
                prefix = f'summary {s_type}: '
            print("\033[93m",  prefix, "\033[0m")
            article_text = prefix + prompt
            self.tokenizer = AutoTokenizer.from_pretrained(Summarizer_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(Summarizer_path)
            input_ids = self.tokenizer(
                [WHITESPACE_HANDLER(article_text)],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=output_max_length,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_beams=num_beams, 
            )[0].to(device)
            summary = self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            results = summary
            self.model.to('cpu')
        else:
            if not is_module_imported('T5ForConditionalGeneration'):
                from transformers import T5ForConditionalGeneration, T5Tokenizer
            if not is_module_imported('T5Tokenizer'):
                from transformers import T5Tokenizer
            model = T5ForConditionalGeneration.from_pretrained(Summarizer_path)
            tokenizer = T5Tokenizer.from_pretrained(Summarizer_path)
            if summary_type == "short":
                s_type = ''
            else:
                s_type = summary_type
            prefix = f'summary {s_type} to {target_language}: '
            if target_language == "source":
                prefix = f'summary {s_type}: '
            print("\033[93m",  prefix, "\033[0m")
            src_text = prefix + prompt
            input_ids = tokenizer(src_text, return_tensors="pt")
            generated_tokens = model.generate(**input_ids).to(device)
            results = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            model.to("cpu")
        with open(temp_Summarization_path, "w", encoding="UTF-8") as text_file:
            if use_T5 == True:
                results = ' '.join(results)
            text_file.write(results)
        return (results, )

class UL_DataProcess_Faster_Whisper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "audio" : ("AUDIO",),
                "whisper_type": (["stable_whisper","faster_whisper","faster_whisper_whisperX"],{"default":"faster_whisper_whisperX"}),
                "faster_whisper_model": (["Systran/faster-whisper-medium","Systran/faster-whisper-large-v3", "Systran/faster-whisper-large-v2", "Systran/faster-whisper-large-v1"],{"default":"Systran/faster-whisper-large-v3"}),
                "stable_whisper_model": (["small","medium","large-v2","large-v3"],{"default":"medium"}),
                "target_lanuage": (["source", "zh",  "yue", "en", "ru", "ja", "ko", "es", "fr", "it", "uk", "hi", "ar", "af", "am", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da", "de", "el", "et", "eu", "fa", "fi", "fo", "gl", "gu", "ha", "haw", "he", "hr", "ht", "hu", "hy", "id", "is", "jw", "ka", "kk", "km", "kn", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "ur", "uz", "vi", "yi", "yo"],{"default": "source"}), 
                "word_timestamps": ("BOOLEAN", {"default": True},),
                "vad_filter": ("BOOLEAN", {"default": True},),
                "whisperX_batch_size": ("INT", {"default": 16}),
                "whisperX_task": (["transcribe", "translate"],{"default": "transcribe"}), 
                "whisperX_diarize_max_speakers": ("INT", {"default": 5, "min": 1,"max": 9999999999999,"step": 1}),
                "whisperX_highlight_words": ("BOOLEAN", {"default": False},),
                "whisperX_max_line_count": ("INT", {"default": 2, "min": 1,"max": 9999999999999,"step": 1}),
                "stable_whisper_cpu_preload": ("BOOLEAN", {"default": True},),
                "stable_whisper_demucs_denoiser": ("BOOLEAN", {"default": False},),
                "faster_whisper_min_silence_duration_ms": ("INT", {"default": 500,"min": 0,"max": 9999999999999,"step": 1},),
                "faster_whisper_beam_size": ("INT", {"default": 5}),
                "faster_whisper_chunk_length": ("INT", {"default": 30,"min": 0,"max": 30,"step": 1}),
                "save_subtitles_to_folder": ("BOOLEAN", {"default": False},),
                "folder": ("STRING", {"default": r"C:\Users\pc\Desktop\ref_audio"}),
                "save_other_formats": ("BOOLEAN", {"default": False},),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
                "faster_whisper_dtype": (["float16","bfloat16","float32", "int8_float16", "int8"],{"default":"float16"}),
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("text", )
    FUNCTION = "UL_DataProcess_Faster_Whisper"
    CATEGORY = "ExtraModels/UL DataProcess"
    TITLE = "UL DataProcess Faster Whisper"
    
    def UL_DataProcess_Faster_Whisper(self,audio, faster_whisper_model, device, faster_whisper_dtype, target_lanuage, word_timestamps, whisper_type, faster_whisper_beam_size, vad_filter, faster_whisper_min_silence_duration_ms, stable_whisper_model, stable_whisper_cpu_preload, save_subtitles_to_folder, folder, save_other_formats, stable_whisper_demucs_denoiser, faster_whisper_chunk_length, whisperX_task, whisperX_highlight_words, whisperX_max_line_count, whisperX_batch_size, whisperX_diarize_max_speakers):
        if whisper_type != 'stable_whisper':
            if faster_whisper_model == 'Systran/faster-whisper-medium':
                model_path = os.path.join(folder_paths.models_dir, 'audio_checkpoints\ExtraModels\models--Systran--faster-whisper-medium')
                # model_path = r"C:\Users\pc\Desktop\models--Systran--faster-whisper-medium"
                if not os.access(os.path.join(model_path, "model.bin"), os.F_OK):
                    model_path = faster_whisper_model
                    
            elif faster_whisper_model == 'Systran/faster-whisper-large-v2':
                model_path = os.path.join(folder_paths.models_dir, 'audio_checkpoints\ExtraModels\models--Systran--faster-whisper-faster-whisper-large-v2')
                # model_path = r"C:\Users\pc\Desktop\models--Systran--faster-whisper-large-v2"
                if not os.access(os.path.join(model_path, "model.bin"), os.F_OK):
                    model_path = faster_whisper_model
                    
            elif faster_whisper_model == 'Systran/faster-whisper-large-v1':
                model_path = os.path.join(folder_paths.models_dir, 'audio_checkpoints\ExtraModels\models--Systran--faster-whisper-faster-whisper-large-v1')
                # model_path = r"C:\Users\pc\Desktop\models--Systran--faster-whisper-large-v1"
                if not os.access(os.path.join(model_path, "model.bin"), os.F_OK):
                    model_path = faster_whisper_model
                    
            else:
                model_path = os.path.join(folder_paths.models_dir, 'audio_checkpoints\ExtraModels\models--Systran--faster-whisper-large-v3')
                # model_path = r"C:\Users\pc\Desktop\models--Systran--faster-whisper-large-v3"
                if not os.access(os.path.join(model_path, "model.bin"), os.F_OK):
                    model_path = faster_whisper_model
                    
                    
        else:
            if stable_whisper_model == "small":
                model_path = os.path.join(folder_paths.models_dir, 'audio_checkpoints\ExtraModels\stable_whisper_model\small.pt')
                # model_path = r"C:\Users\pc\Desktop\stable_whisper_model\small.pt"
                if not os.access(model_path, os.F_OK):
                    model_path = stable_whisper_model
                    
            if stable_whisper_model == "medium":
                model_path = os.path.join(folder_paths.models_dir, 'audio_checkpoints\ExtraModels\stable_whisper_model\medium.pt')
                # model_path = r"C:\Users\pc\Desktop\stable_whisper_model\medium.pt"
                if not os.access(model_path, os.F_OK):
                    model_path = stable_whisper_model
                    
            if stable_whisper_model == "large-v2":
                model_path = os.path.join(folder_paths.models_dir, 'audio_checkpoints\ExtraModels\stable_whisper_model\large-v2.pt')
                # model_path = r"C:\Users\pc\Desktop\stable_whisper_model\large-v2.pt"
                if not os.access(model_path, os.F_OK):
                    model_path = stable_whisper_model
                    
            if stable_whisper_model == "large-v3":
                model_path = os.path.join(folder_paths.models_dir, 'audio_checkpoints\ExtraModels\stable_whisper_model\large-v3.pt')
                # model_path = r"C:\Users\pc\Desktop\stable_whisper_model\large-v3.pt"
                if not os.access(model_path, os.F_OK):
                    model_path = stable_whisper_model

        model_device = get_device_by_name(device)
        audio_save_path = audio_file_or_audio_tensor(audio)
        
        if ('waveform' in audio and 'sample_rate' in audio):
            srt_name = f'whisper_subtitle_{time.time()}'
        else:
            dirname, filename = os.path.split(audio_save_path)
            srt_name = str(filename).replace(".wav", "").replace(".mp3", "").replace(".m4a", "").replace(".ogg", "").replace(".flac", "").replace(".mp4", "").replace(".mkv", "").replace(".flv", "").replace(".ts", "").replace(".rmvb", "").replace(".rm", "").replace(".avi", "")
            # srt_path = os.path.join(os.path.expanduser("~"), r"Desktop\ref_audio", srt_name)
            # sys_temp_dir = tempfile.gettempdir()
            # temp_audio = os.path.join(sys_temp_dir, srt_name)
        
        audio_save_path = get_audio_from_video(audio_save_path)
            
        # save audio bytes from VHS to file
        # temp_dir = tempfile.gettempdir()
        # audio_save_path = os.path.join(temp_dir,f"{uuid.uuid1()}.wav")
        # with open(audio_save_path, 'wb') as f:
        #     f.write(audio())

        if whisper_type == 'stable_whisper':
            if not is_module_imported('load_model'):
                from .stable_whisper import load_model
            # transribe using whisper
            self.model = load_model(model_path, cpu_preload=stable_whisper_cpu_preload, device=model_device)
            
            if target_lanuage == "source":
                target_lanuage = None
                extra_task = "transcribe"
            else:
                extra_task = 'translate'
                
            if stable_whisper_demucs_denoiser == True:
                denoiser = "demucs"
            else:
                denoiser = None
                
            result = self.model.transcribe(audio_save_path, word_timestamps=word_timestamps, vad=vad_filter, denoiser=denoiser, task=extra_task, language=target_lanuage)
            if save_subtitles_to_folder == True:
                result.to_srt_vtt(folder+f'\{srt_name}.srt') #SRT
                result.to_ass(folder+f'\{srt_name}.ass') #ASS
                if save_other_formats == True:
                    result.to_srt_vtt(folder+f'\{srt_name}.vtt') #VTT
                    result.to_tsv(folder+f'\{srt_name}.tsv') #TSV
                    result.to_txt(folder+f'\{srt_name}.txt') #txt
                    result.save_as_json(folder+f'\{srt_name}.json') #json
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            result = result.to_dict()
            segments = result['segments']
            
            segments_alignment = []
            words_alignment = []

            for segment in segments:
                # create segment alignments
                segment_dict = {
                    'value': segment['text'].strip(),
                    'start': segment['start'],
                    'end': segment['end']
                }
                segments_alignment.append(segment_dict)
                # create word alignments
                for word in segment["words"]:
                    word_dict = {
                        'value': word["word"].strip(),
                        'start': word["start"],
                        'end': word['end']
                    }
                    words_alignment.append(word_dict)
                    
            # self.model.to('cpu', torch.float32)
            return (result["text"].strip(), segments_alignment, words_alignment)
        
        elif whisper_type == 'faster_whisper_whisperX':
            if target_lanuage == 'source':
                target_lanuage = None
            
            PYANNOTE_CONFIG_PATH = os.path.join(current_directory, 'whisperX_models\pyannote_config.yaml')
            if not is_module_imported('whisperx'):
                from . import whisperx
            if not is_module_imported('WriteTXT'):
                from .whisperx.utils import WriteTXT
            if not is_module_imported('WriteSRT'):
                from .whisperx.utils import WriteSRT
            import gc
            # No language specified--Depending on your exact usage, you can pass it to load_model() or the transcribe() function. Simply specify language=
            model = whisperx.load_model(model_path, model_device, compute_type=faster_whisper_dtype, language=target_lanuage, task=whisperX_task)
            audio = whisperx.load_audio(audio_save_path)
            result = model.transcribe(audio, batch_size=whisperX_batch_size, language=target_lanuage, chunk_size=faster_whisper_chunk_length, print_progress=True, combined_progress=True)
            gc.collect
            torch.cuda.empty_cache()
            del model
            if result["language"] == 'en':
                # ALIGN_MODEL_DIR = r"C:\Users\pc\Desktop\models--facebook--wav2vec2-base-960h"
                ALIGN_MODEL_DIR = os.path.join(folder_paths.models_dir, "audio_checkpoints\ExtraModels\models--facebook--wav2vec2-base-960h")
                if not os.access(os.path.join(ALIGN_MODEL_DIR, "wav2vec2_fairseq_base_ls960_asr_ls960.pth"), os.F_OK):
                    ALIGN_MODEL_DIR = "facebook/wav2vec2-base-960h"
            elif result["language"] == 'zh':
                # ALIGN_MODEL_DIR = r"C:\Users\pc\Desktop\models--jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn"
                ALIGN_MODEL_DIR = os.path.join(folder_paths.models_dir, "audio_checkpoints\ExtraModels\models--jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn")
                if not os.access(os.path.join(ALIGN_MODEL_DIR, "pytorch_model.bin"), os.F_OK):
                    ALIGN_MODEL_DIR = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
            elif result["language"] == 'ru':
                ALIGN_MODEL_DIR = os.path.join(folder_paths.models_dir, "audio_checkpoints\ExtraModels\models--jonatasgrosman--wav2vec2-large-xlsr-53-russian")
                if not os.access(os.path.join(ALIGN_MODEL_DIR, "pytorch_model.bin"), os.F_OK):
                    ALIGN_MODEL_DIR = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
            elif result["language"] == 'ja':
                ALIGN_MODEL_DIR = os.path.join(folder_paths.models_dir, "audio_checkpoints\ExtraModels\models--jonatasgrosman--wav2vec2-large-xlsr-53-japanese")
                if not os.access(os.path.join(ALIGN_MODEL_DIR, "pytorch_model.bin"), os.F_OK):
                    ALIGN_MODEL_DIR = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
            elif result["language"] == 'ko':
                ALIGN_MODEL_DIR = os.path.join(folder_paths.models_dir, "audio_checkpoints\ExtraModels\models--jonatasgrosman--wav2vec2-large-xlsr-korean")
                if not os.access(os.path.join(ALIGN_MODEL_DIR, "pytorch_model.bin"), os.F_OK):
                    ALIGN_MODEL_DIR = "kresnik/wav2vec2-large-xlsr-korean"
            elif result["language"] == 'ar':
                ALIGN_MODEL_DIR = os.path.join(folder_paths.models_dir, "audio_checkpoints\ExtraModels\models--jonatasgrosman--wav2vec2-large-xlsr-53-arabic")
                if not os.access(os.path.join(ALIGN_MODEL_DIR, "pytorch_model.bin"), os.F_OK):
                    ALIGN_MODEL_DIR = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
            elif result["language"] == 'es':
                ALIGN_MODEL_DIR = os.path.join(folder_paths.models_dir, "audio_checkpoints\ExtraModels\wav2vec2_voxpopuli_base_10k_asr_es")
                if not os.access(os.path.join(ALIGN_MODEL_DIR,'wav2vec2_voxpopuli_base_10k_asr_es.pt'), os.F_OK):
                    ALIGN_MODEL_DIR = None
            
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=model_device, model_dir=ALIGN_MODEL_DIR)
            aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, model_device, return_char_alignments=False)
            
            if save_subtitles_to_folder == True:
                aligned_result["language"] = result["language"]
                srt_path = os.path.join(folder, f'{srt_name}_noDiarization.srt')
                with open(srt_path, "w", encoding="utf-8") as srt:
                    writesrt = WriteSRT(srt_path)
                    writesrt.write_result(aligned_result, srt, {"max_line_width": None, "max_line_count": whisperX_max_line_count, "highlight_words": whisperX_highlight_words, "preserve_segments": True})
                if save_other_formats == True:
                    from .whisperx.utils import WriteTSV, WriteJSON, WriteVTT
                    tsv_noDiarization_path = os.path.join(folder, f'{srt_name}_noDiarization.tsv')
                    json_noDiarization_path = os.path.join(folder, f'{srt_name}_noDiarization.json')
                    vtt_noDiarization_path = os.path.join(folder, f'{srt_name}_noDiarization.vtt')
                    txt_noDiarization_path = os.path.join(folder, f'{srt_name}_noDiarization.txt')
                    with open(tsv_noDiarization_path, "w", encoding="utf-8") as tsv_noDiarization:
                        writetsv = WriteTSV(tsv_noDiarization_path)
                        writetsv.write_result(aligned_result, tsv_noDiarization, {"max_line_width": None, "max_line_count": whisperX_max_line_count, "highlight_words": whisperX_highlight_words, "preserve_segments": True})
                    with open(json_noDiarization_path, "w", encoding="utf-8") as json_noDiarization:
                        writejson = WriteJSON(json_noDiarization_path)
                        writejson.write_result(aligned_result, json_noDiarization, {"max_line_width": None, "max_line_count": whisperX_max_line_count, "highlight_words": whisperX_highlight_words, "preserve_segments": True})
                    with open(vtt_noDiarization_path, "w", encoding="utf-8") as vtt_noDiarization:
                        writevtt = WriteVTT(vtt_noDiarization_path)
                        writevtt.write_result(aligned_result, vtt_noDiarization, {"max_line_width": None, "max_line_count": whisperX_max_line_count, "highlight_words": whisperX_highlight_words, "preserve_segments": True})
                    with open(txt_noDiarization_path, "w", encoding="utf-8") as txt_noDiarization:
                        writetxt = WriteTXT(txt_noDiarization_path)
                        writetxt.write_result(aligned_result, txt_noDiarization, {"max_line_width": None, "max_line_count": whisperX_max_line_count, "highlight_words": whisperX_highlight_words, "preserve_segments": True})
            gc.collect
            torch.cuda.empty_cache()
            del model_a
            
            diarize_model = whisperx.DiarizationPipeline(model_name=PYANNOTE_CONFIG_PATH, device=model_device)
            diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=whisperX_diarize_max_speakers)
            diarize_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
            
            if save_subtitles_to_folder == True:
                diarize_result["language"] = result["language"]
                srt_path = os.path.join(folder, f'{srt_name}.srt')
                # write_to_result(diarize_result, srt_path, whisperX_keep_speaker)
                with open(srt_path, 'w', encoding='utf-8') as srt:
                    writesrt = WriteSRT(srt_path)
                    writesrt.write_result(diarize_result, file=srt, options={"max_line_width":None,"max_line_count": whisperX_max_line_count,"highlight_words":whisperX_highlight_words})
                if save_other_formats == True:
                    tsv_path = os.path.join(folder, f'{srt_name}.tsv')
                    json_path = os.path.join(folder, f'{srt_name}.json')
                    vtt_path = os.path.join(folder, f'{srt_name}.vtt')
                    txt_path = os.path.join(folder, f'{srt_name}.txt')
                    # ori_srt_path = os.path.join(folder, f'{srt_name}_ori.srt')
                    with open(tsv_path, "w", encoding="utf-8") as tsv:
                        writetsv = WriteTSV(tsv_path)
                        writetsv.write_result(diarize_result, tsv, {"max_line_width": None, "max_line_count": whisperX_max_line_count, "highlight_words": whisperX_highlight_words, "preserve_segments": True})
                    with open(json_path, "w", encoding="utf-8") as json:
                        writejson = WriteJSON(json_path)
                        writejson.write_result(diarize_result, json, {"max_line_width": None, "max_line_count": whisperX_max_line_count, "highlight_words": whisperX_highlight_words, "preserve_segments": True})
                    with open(vtt_path, "w", encoding="utf-8") as vtt:
                        writevtt = WriteVTT(vtt_path)
                        writevtt.write_result(diarize_result, vtt, {"max_line_width": None, "max_line_count": whisperX_max_line_count, "highlight_words": whisperX_highlight_words, "preserve_segments": True})
                    with open(txt_path, "w", encoding="utf-8") as txt:
                        writetxt = WriteTXT(txt_path)
                        writetxt.write_result(diarize_result, txt, {"max_line_width": None, "max_line_count": whisperX_max_line_count, "highlight_words": whisperX_highlight_words, "preserve_segments": True})
                    # with open(ori_srt_path, 'w', encoding='utf-8') as ori_srt:
                    #     writesrt = WriteSRT(ori_srt_path)
                    #     writesrt.write_result(diarize_result, file=ori_srt, options={"max_line_width":None,"max_line_count": whisperX_max_line_count,"highlight_words":whisperX_highlight_words})
            with open(temp_txt_path, "w", encoding="utf-8") as txt:
                writetxt=WriteTXT(temp_txt_path)  #output file directory
                writetxt.write_result(diarize_result, file=txt,options={"max_line_width":None,"max_line_count": whisperX_max_line_count,"highlight_words": whisperX_highlight_words})
            with open(temp_txt_path, 'r', encoding='utf-8') as f:
                result = f.read()
            gc.collect
            torch.cuda.empty_cache()
            del diarize_model
            return(result, )
            
        else:
            if not is_module_imported('WhisperModel'):
                from faster_whisper import WhisperModel
            model = WhisperModel(model_path, device="cuda", compute_type=faster_whisper_dtype)
            
            if target_lanuage == "source":
                segments, info = model.transcribe(audio_save_path, chunk_length=faster_whisper_chunk_length, beam_size=faster_whisper_beam_size, word_timestamps=word_timestamps, vad_filter=vad_filter, vad_parameters=dict(min_silence_duration_ms=faster_whisper_min_silence_duration_ms),)
            else:
                segments, info = model.transcribe(audio_save_path, chunk_length=faster_whisper_chunk_length, beam_size=faster_whisper_beam_size,language=target_lanuage, word_timestamps=word_timestamps, vad_filter=vad_filter, vad_parameters=dict(min_silence_duration_ms=faster_whisper_min_silence_duration_ms),)
                
            results= []
            for s in segments:
                segment_dict = {'start':s.start,'end':s.end,'text':s.text}
                results.append(segment_dict)
            import pysubs2
            subs = pysubs2.load_from_whisper(results)
            subs.save(temp_txt_path)
            if save_subtitles_to_folder == True:
                subs.save(folder+f'\{srt_name}.srt', encoding="utf-8")
                subs.save(folder+f'\{srt_name}.ass', encoding="utf-8")
                if save_other_formats == True:
                    subs.save(folder+f'\{srt_name}.ssa', encoding="utf-8")
                    # subs.save(folder+f'\{srt_name}.sub', encoding="utf-8") # microdvd
                    subs.save(folder+f'\{srt_name}.json', encoding="utf-8")
                    subs.save(folder+f'\{srt_name}.txt', encoding="utf-8") # tmp
                    subs.save(folder+f'\{srt_name}.vtt', encoding="utf-8")
                
            with open(temp_txt_path, "r", encoding="UTF-8") as f:
                result = (f.read(), )
                
            del model
            return (result, )
                

# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "UL_DataProcess_t5_translate_en_ru_zh": UL_DataProcess_t5_translate_en_ru_zh, 
    "UL_DataProcess_nlp_csanmt_translation_zh2en_translator": UL_DataProcess_nlp_csanmt_translation_zh2en_translator, 
    "UL_DataProcess_nllb_200_translator": UL_DataProcess_nllb_200_translator,
    "UL_DataProcess_Summarization": UL_DataProcess_Summarization, 
    "UL_DataProcess_Faster_Whisper": UL_DataProcess_Faster_Whisper,
}



# {
#     "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
#     "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
#     "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
#     "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
#     "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
#     "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
#     "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
#     "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
#     "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
#     "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
#     "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
#     "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
# }