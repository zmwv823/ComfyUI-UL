import os
import folder_paths
from ..UL_common.common import get_device_by_name
from .utils import nlp_csanmt_translation_zh2en, SavedModel_Translator, t5_translate_en_ru_zh

current_directory = os.path.dirname(os.path.abspath(__file__))
temp_txt_path = os.path.join(current_directory, "temp_dir", "temp.txt")

class UL_Data_Process_t5_translate_en_ru_zh:
  
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "translator": (["utrobinmv/t5_translate_en_ru_zh_base_200", "utrobinmv/t5_translate_en_ru_zh_large_1024"],{"default": "utrobinmv/t5_translate_en_ru_zh_base_200"}), 
                "prompt": ("STRING", {"default": "Цель разработки — предоставить пользователям личного синхронного переводчика.", "multiline": True}),
                "Batch_prompt": ("STRING", {"default": "这里是多批次翻译文本输入，使用换行进行分割。\n天上掉馅饼啦，快去看超人！！！\n飞流直下三千尺，疑似银河落九天。\n启用Batch_Newline表示输出的翻译会按换行输入进行二次换行,否则是用空格合并起来的整篇文本。", "multiline": True}),
                "Batch_Newline" :("BOOLEAN", {"default": True}),
                "if_Batch": ("BOOLEAN", {"default": False}),
                "Target_Language": (["en", "zh", "ru", ],{"default": "en"}), 
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
            },
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "ExtraModels/UL Data Process"
    FUNCTION = "UL_Data_Process_t5_translate_en_ru_zh"
    TITLE = "UL Data Process t5_translate_en_ru_zh"

    def UL_Data_Process_t5_translate_en_ru_zh(self, translator, Target_Language, prompt, device, Batch_prompt, Batch_Newline, if_Batch):
        device = get_device_by_name(device)
        # 使用换行(\n)作为分隔符
        Batch_prompt = Batch_prompt.split("\n")  
        if if_Batch == True:
            input_sequence = Batch_prompt
            # 用特定的连接符<SENT_SPLIT>，将多个句子进行串联
            input_sequence = '<SENT_SPLIT>'.join(input_sequence)
        else:
            input_sequence = prompt
        if translator == 'utrobinmv/t5_translate_en_ru_zh_large_1024':
            if not os.access(os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_large_1024", "model.safetensors"), os.F_OK):
                zh2en_path = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
            else:
                zh2en_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_large_1024")
            outputs = t5_translate_en_ru_zh(Target_Language, input_sequence, zh2en_path, device)[0]
        elif translator == 'utrobinmv/t5_translate_en_ru_zh_base_200':
            if not os.access(os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_base_200", "model.safetensors"), os.F_OK):
                zh2en_path = 'utrobinmv/t5_translate_en_ru_zh_base_200'
            else:
                zh2en_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_base_200")
            outputs = t5_translate_en_ru_zh(Target_Language, input_sequence, zh2en_path, device)[0]
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
        # model.to('cpu', torch.float32)
        return (results, )

class UL_Data_Process_nlp_csanmt_translation_zh2en_translator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (['damo/nlp_csanmt_translation_zh2en', 'Saved_Model'] , {"default": "damo/nlp_csanmt_translation_zh2en"}),
                "device": (['cpu', 'cuda', 'gpu'] , {"default": "cuda"}),
                "prompt": ("STRING", {"default": "这里是单批次翻译文本输入。\n声明补充说，沃伦的同事都深感震惊，并且希望他能够投案自首。\n尽量输入单句文本，如果是多句长文本建议人工分句，否则可能出现漏译或未译等情况！！！\n建议换行，效果更佳。", "multiline": True}),
                "Batch_prompt": ("STRING", {"default": "这里是多批次翻译文本输入，使用换行进行分割。\n天上掉馅饼啦，快去看超人！！！\n飞流直下三千尺，疑似银河落九天。\n启用Batch_Newline表示输出的翻译会按换行输入进行二次换行,否则是用空格合并起来的整篇文本。", "multiline": True}),
                "Batch_Newline" :("BOOLEAN", {"default": True}),
                "if_Batch": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("中译英结果",)
    CATEGORY = "ExtraModels/UL Data Process"
    FUNCTION = "UL_Data_Process_nlp_csanmt_translation_zh2en_translator"
    TITLE = "UL 中译英-阿里达摩院damo/nlp_csanmt_translation_zh2en"

    def UL_Data_Process_nlp_csanmt_translation_zh2en_translator(self, prompt, Batch_prompt, if_Batch, device, Batch_Newline, model):
        # 使用换行(\n)作为分隔符
        Batch_prompt = Batch_prompt.split("\n")  
        if if_Batch == True:
            input_sequence = Batch_prompt
            # 用特定的连接符<SENT_SPLIT>，将多个句子进行串联
            input_sequence = '<SENT_SPLIT>'.join(input_sequence)
        else:
            input_sequence = prompt
        if model == 'damo/nlp_csanmt_translation_zh2en':
            outputs = nlp_csanmt_translation_zh2en(device, input_sequence)['translation']
        else:
            outputs = SavedModel_Translator(input_sequence)['translation']
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
        
# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "UL_Data_Process_t5_translate_en_ru_zh": UL_Data_Process_t5_translate_en_ru_zh, 
    "UL_Data_Process_nlp_csanmt_translation_zh2en_translator": UL_Data_Process_nlp_csanmt_translation_zh2en_translator, 
}
