from ..UL_common.common import is_module_imported
import time
import torch
import os
import folder_paths
import time
import numpy as np
import hashlib

comfyui_models_dir = folder_paths.models_dir
current_directory = os.path.dirname(os.path.abspath(__file__))
temp_txt_path = os.path.join(current_directory, "temp_dir", "AnyText_temp.txt")
input_path = folder_paths.get_input_directory()
    
class UL_DataProcess_Create_SavedModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "translator": (["damo/nlp_csanmt_translation_en2zh_base",  "damo/nlp_csanmt_translation_en2zh", "damo/nlp_csanmt_translation_zh2en"], {"default": "damo/nlp_csanmt_translation_zh2en"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Model Path",)
    CATEGORY = "ExtraModels/UL DataProcess"
    FUNCTION = "UL_DataProcess_Create_SavedModel"
    TITLE = "UL DataProcess Create SavedModel"

    def UL_DataProcess_Create_SavedModel(self, translator):
        if translator == 'damo/nlp_csanmt_translation_en2zh_base':
            translator_path = os.path.join(comfyui_models_dir, "prompt_generator\modelscope--damo--nlp_csanmt_translation_en2zh_base")
        elif translator == 'damo/nlp_csanmt_translation_en2zh':
            translator_path = os.path.join(comfyui_models_dir, "prompt_generator\modelscope--damo--nlp_csanmt_translation_en2zh")
        elif translator == 'damo/nlp_csanmt_translation_zh2en':
            translator_path = os.path.join(comfyui_models_dir, "prompt_generator\modelscope--damo--nlp_csanmt_translation_zh2en")
        if not os.access(os.path.join(translator_path, "tf_ckpts", "ckpt-0.data-00000-of-00001"), os.F_OK):
            translator_path = translator
                    
        save_folder = os.path.join(translator_path, "CSANMT")
        Create_SavedModel(translator_path)
        result = ("转换后的模型位置：\n" + save_folder)
        return (result, )

class UL_Load_Data:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["srt", "txt", "pdf", "ass"]]
        return {"required":
                    {"data": (sorted(files),)},
                }

    CATEGORY = "ExtraModels/UL DataProcess"
    RETURN_NAMES = ("data_path", )
    RETURN_TYPES = ("DATA_PATH",)
    FUNCTION = "UL_Load_Data"
    TITLE = "UL Load Data"

    def UL_Load_Data(self, data):
        data_path = folder_paths.get_annotated_filepath(data)
        return (data_path,)

    @classmethod
    def IS_CHANGED(s, data):
        data_path = folder_paths.get_annotated_filepath(data)
        m = hashlib.sha256()
        with open(data_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "UL_DataProcess_Create_SavedModel": UL_DataProcess_Create_SavedModel, 
    "UL_Load_Data": UL_Load_Data, 
}

def SavedModel_Translator(prompt, model):
    sttime = time.time()
    import tensorflow
    from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
    if not is_module_imported('Config'):
        from modelscope.utils.config import Config
    if not is_module_imported('OutputKeys'):
        from modelscope.outputs import OutputKeys
    if not is_module_imported('BPE'):
        from subword_nmt.apply_bpe import BPE
    if tensorflow.__version__ >= '2.0':
        tf = tensorflow.compat.v1
        tf.disable_eager_execution()
    if model == 'Saved_Model_zh2en':
        model_dir = os.path.join(folder_paths.models_dir, "prompt_generator", "modelscope--damo--nlp_csanmt_translation_zh2en")
    elif model == 'Saved_Model_en2zh_base':
        model_dir = os.path.join(folder_paths.models_dir, "prompt_generator", "modelscope--damo--nlp_csanmt_translation_en2zh_base")
    elif model == 'Saved_Model_en2zh':
        model_dir = os.path.join(folder_paths.models_dir, "prompt_generator", "modelscope--damo--nlp_csanmt_translation_en2zh")
    if not os.access(os.path.join(model_dir, "CSANMT", "variables", "variables.data-00000-of-00001"), os.F_OK):
        raise Exception(f'Generate converted model with "UL_Data_Process_Create_SavedModel" node first(先使用“UL_Data_Process_Create_SavedModel”节点生成转换模型).')

    #读取配置文件和中英字典（这些文件需要再模型下载中获得）：
    cfg_dir = os.path.join(model_dir,"configuration.json")
    cfg = Config.from_file(cfg_dir)

    src_vocab_dir = os.path.join(model_dir,"src_vocab.txt")
    # _src_vocab 是一个字典，key是英文src_vocab.txt中的一行，value是index，length = 49998, _trg_rvocab同理
    _src_vocab = dict([
                (w.strip(), i) for i, w in enumerate(open(src_vocab_dir, encoding='UTF-8'))
            ])

    trg_vocab_dir = os.path.join(model_dir,"trg_vocab.txt")
    _trg_rvocab = dict([
                (i, w.strip()) for i, w in enumerate(open(trg_vocab_dir, encoding='UTF-8'))
            ])

    #输入输出配置
    input_wids = tf.placeholder(
                dtype=tf.int64, shape=[None, None], name='input_wids')
    output = {}

    _src_lang = cfg['preprocessor']['src_lang'] #zh
    _tgt_lang = cfg['preprocessor']['tgt_lang'] #en

    # _src_bpe_path = os.path.join(model_dir,"bpe.en")
    _src_bpe_path = os.path.join(model_dir,"bpe.zh")
    if model != 'Saved_Model_zh2en':
        _src_bpe_path = os.path.join(model_dir,"bpe.en")

    _punct_normalizer = MosesPunctNormalizer(lang=_src_lang)
    _tok = MosesTokenizer(lang=_src_lang)
    _detok = MosesDetokenizer(lang=_tgt_lang)
    _bpe = BPE(open(_src_bpe_path, encoding='UTF-8'))

    #文本encode：
    # input = ["这里是多批次翻译文本输入，使用换行进行分割。", "尽量输入单句文本，如果是多句长文本建议人工分句，否则可能出现漏译或未译等情况！！！"]
    input = [prompt]

    input = [_punct_normalizer.normalize(item) for item in input]

    aggressive_dash_splits = True

    if (_src_lang in ['es', 'fr'] and _tgt_lang == 'zh') or (_src_lang == 'zh' and _tgt_lang in ['es', 'fr']):
        aggressive_dash_splits = False
        if model != 'Saved_Model_zh2en':
            if (_src_lang in ['es', 'fr'] and _tgt_lang == 'en') or (_src_lang == 'en' and _tgt_lang in ['es', 'fr']):
                aggressive_dash_splits = False

    input_tok = [
                    _tok.tokenize(
                        item,
                        return_str=True,
                        aggressive_dash_splits=aggressive_dash_splits)
                    for item in input
                ]
    input_bpe = [
                _bpe.process_line(item).strip().split() for item in input_tok
            ]

    MAX_LENGTH = max([len(item) for item in input_bpe])#200

    input_ids = np.array([[
                _src_vocab[w] if w in _src_vocab else
                cfg['model']['src_vocab_size'] - 1 for w in item
            ] + [0] * (MAX_LENGTH - len(item)) for item in input_bpe])

    #模型配置、读取、调用：
    tf_config = tf.ConfigProto(allow_soft_placement=True)

    sess = tf.Session(graph=tf.Graph(), config=tf_config)
    # Restore model from the saved_modle file, that is exported by TensorFlow estimator.
    MetaGraphDef = tf.saved_model.loader.load(sess, ['serve'], os.path.join(model_dir,'CSANMT'))

    # SignatureDef protobuf
    SignatureDef_map = MetaGraphDef.signature_def
    SignatureDef = SignatureDef_map['translation_signature']
    # TensorInfo protobuf
    X_TensorInfo = SignatureDef.inputs['input_wids']
    y_TensorInfo = SignatureDef.outputs['output_seqs']
    X = tf.saved_model.utils.get_tensor_from_tensor_info(
        X_TensorInfo, sess.graph)
    y = tf.saved_model.utils.get_tensor_from_tensor_info(
        y_TensorInfo, sess.graph)
    outputs = sess.run(y, feed_dict={X: input_ids})

    #decode
    x, y, z = outputs.shape

    translation_out = []
    for i in range(x):
        output_seqs = outputs[i]
        wids = list(output_seqs[0]) + [0]
        wids = wids[:wids.index(0)]
        translation = ' '.join([
            _trg_rvocab[wid] if wid in _trg_rvocab else '<unk>'
            for wid in wids
        ]).replace('@@ ', '').replace('@@', '')
        translation_out.append(_detok.detokenize(translation.split()))
    translation_out = '<SENT_SPLIT>'.join(translation_out)
    results = {OutputKeys.TRANSLATION: translation_out}
    # print(results, result)
    endtime = time.time()
    from tensorflow.python.client import device_lib
    print("\033[93m翻译耗时：", endtime - sttime, "\n翻译使用的设备：\n", device_lib.list_local_devices(), "\033[0m")
    del MetaGraphDef
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results
    #以上就是一整套本地调用翻译的全部流程，将它们按顺序放在一整个脚本中就可以顺利翻译了。

#将模型转换为SavedModel格式
def Create_SavedModel(model_path):
    if not is_module_imported('Model'):
        from modelscope.models import Model
    if not is_module_imported('TfModelExporter'):
        from modelscope.exporters import TfModelExporter
    model = Model.from_pretrained(model_path)
    output_files = TfModelExporter.from_model(model).export_saved_model(output_dir=os.path.join(model_path, "CSANMT"))
    print(output_files) # {'model': '/tmp'}
    output_dir = os.path.join(model_path, "CSANMT")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir

def t5_translate_en_ru_zh(Target_Language, prompt, t5_translate_en_ru_zh_base_200_path, device):
    # prefix = 'translate to en: '
    sttime = time.time()
    if not is_module_imported('T5ForConditionalGeneration'):
        from transformers import T5ForConditionalGeneration
    if not is_module_imported('T5Tokenizer'):
        from transformers import T5Tokenizer
    model = T5ForConditionalGeneration.from_pretrained(t5_translate_en_ru_zh_base_200_path,)
    tokenizer = T5Tokenizer.from_pretrained(t5_translate_en_ru_zh_base_200_path)
    if Target_Language == 'zh':
        prefix = 'translate to zh: '
    elif Target_Language == 'en':
        prefix = 'translate to en: '
    else:
        prefix = 'translate to ru: '
    src_text = prefix + prompt
    input_ids = tokenizer(src_text, return_tensors="pt")
    generated_tokens = model.generate(**input_ids).to(device, torch.float32)
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    model.to('cpu')
    endtime = time.time()
    print("\033[93mTime for translating(翻译耗时): ", endtime - sttime, "\033[0m")
    return result

def nlp_csanmt_translation_zh2en(device, prompt, nlp_csanmt_translation_zh2en_path):
    sttime = time.time()
    if not is_module_imported('pipeline'):
        from modelscope.pipelines import pipeline
    if not is_module_imported('Tasks'):
        from modelscope.utils.constant import Tasks
    pipeline_ins = pipeline(task=Tasks.translation, model=nlp_csanmt_translation_zh2en_path, device=device)
    outputs = pipeline_ins(input=prompt)
    endtime = time.time()
    from tensorflow.python.client import device_lib
    print("\033[93m翻译耗时：", endtime - sttime, "\n翻译使用的设备：\n", device_lib.list_local_devices(), "\033[0m")
    del pipeline_ins
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs

def nllb_200_translator(device, dtype, prompt, Source_Language, Target_Language, use_fast, model, max_length):
    sttime = time.time()
    if model == "facebook/nllb-200-distilled-1.3B":
        if not os.access(os.path.join(folder_paths.models_dir, "prompt_generator", "models--facebook--nllb-200-distilled-1.3B", "pytorch_model.bin"), os.F_OK):
            nllb_200_path = "facebook/nllb-200-distilled-1.3B"
        else:
            nllb_200_path = os.path.join(folder_paths.models_dir, r'prompt_generator\models--facebook--nllb-200-distilled-1.3B')
    elif model == "facebook/nllb-200-distilled-600M":
        if not os.access(os.path.join(folder_paths.models_dir, "prompt_generator", "models--facebook--nllb-200-distilled-1.3B", "pytorch_model.bin"), os.F_OK):
            nllb_200_path = "facebook/nllb-200-distilled-600M"
        else:
            nllb_200_path = os.path.join(folder_paths.models_dir, r'prompt_generator\models--facebook--nllb-200-distilled-600M')
    else:
        if not os.access(os.path.join(folder_paths.models_dir, "prompt_generator", "models--facebook--facebook/nllb-200-3.3B", "pytorch_model-00002-of-00003.bin"), os.F_OK):
            nllb_200_path = "facebook/nllb-200-3.3B"
        else:
            nllb_200_path = os.path.join(folder_paths.models_dir, r'prompt_generator\models--facebook--nllb-200-3.3B')
    if not is_module_imported('AutoTokenizer'):
        from transformers import AutoTokenizer
    if not is_module_imported('flores_codes'):
        from .facebook_nllb_200_scripts.flores200_codes import flores_codes
    if not is_module_imported('AutoModelForSeq2SeqLM'):
        from transformers import AutoModelForSeq2SeqLM
    if not is_module_imported('pipeline'):
        from transformers import pipeline
    model = AutoModelForSeq2SeqLM.from_pretrained(nllb_200_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(nllb_200_path)
    source = flores_codes[Source_Language]
    target = flores_codes[Target_Language]
    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source, tgt_lang=target, device_map=device, torch_dtype=dtype, use_fast=use_fast)
    output = translator(prompt, max_length=max_length)
    model.to('cpu')
    outputs= output[0]['translation_text']
    
    endtime = time.time()
    print("\033[93mTime for translating(翻译耗时): ", endtime - sttime, "\033[0m")
    return outputs
    
def convert_time_format(start_time, end_time):
    start_seconds = int(start_time)
    start_minutes = start_seconds // 60
    start_seconds %= 60
    start_milliseconds = int((start_time - int(start_time)) * 1000)

    end_seconds = int(end_time)
    end_minutes = end_seconds // 60
    end_seconds %= 60
    end_milliseconds = int((end_time - int(end_time)) * 1000)

    return f"{start_minutes:02d}:{start_seconds:02d}.{start_milliseconds:03d} --> {end_minutes:02d}:{end_seconds:02d}.{end_milliseconds:03d}"

def write_to_result(result, file_path, keep_speaker):
    mode = "w"
    with open(file_path, mode, encoding="utf-8") as f:
        f.write("WEBVTT")
        f.write("\n\n")
        if keep_speaker == True:
            for segment in result["segments"]:
                f.write(f'{convert_time_format(segment["start"],segment["end"])}')
                f.write("\n")
                f.write( 
                    # (("[[" + segment["speaker"] + "]]") if "speaker" in segment else "") + " "
                    # + segment["text"].strip().replace("\t", " ")
                    ((segment["speaker"] + ": ") if "speaker" in segment else "") + " "
                    + segment["text"].strip().replace("\t", " ")
                )
                f.write("\n\n")
        else:
            for segment in result["segments"]:
                f.write(f'{convert_time_format(segment["start"],segment["end"])}')
                f.write("\n")
                f.write( 
                    # (("[[" + segment["speaker"] + "]]") if "speaker" in segment else "") + " "
                    # + segment["text"].strip().replace("\t", " ")
                    segment["text"].strip().replace("\t", " ")
                )
                f.write("\n\n")