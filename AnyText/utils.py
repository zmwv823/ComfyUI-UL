import os
import folder_paths
import torch
from modelscope.pipelines import pipeline
import node_helpers
from PIL import Image, ImageOps, ImageSequence
import hashlib
import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
comfyui_models_dir = folder_paths.models_dir
comfyui_temp_dir = folder_paths.get_temp_directory()
temp_txt_path = os.path.join(current_directory, "temp_dir", "AnyText_temp.txt")
temp_img_path = os.path.join(current_directory, "temp_dir", "AnyText_mask_pos_img.png")

class AnyText_loader:
    @classmethod
    def INPUT_TYPES(cls):
        font_list = os.listdir(os.path.join(comfyui_models_dir, "fonts"))
        checkpoints_list = folder_paths.get_filename_list("checkpoints")
        clip_list = os.listdir(os.path.join(comfyui_models_dir, "clip"))
        translator_list = os.listdir(os.path.join(comfyui_models_dir, "prompt_generator"))
        font_list.insert(0, "Auto_DownLoad")
        checkpoints_list.insert(0, "Auto_DownLoad")
        clip_list.insert(0, "Auto_DownLoad")
        translator_list.insert(0, "Auto_DownLoad")

        return {
            "required": {
                "font": (font_list, ),
                "ckpt_name": (checkpoints_list, ),
                "clip": (clip_list, ),
                "clip_path_or_repo_id": ("STRING", {"default": "openai/clip-vit-large-patch14"}),
                "translator": (translator_list, ),
                "translator_path_or_repo_id": ("STRING", {"default": "damo/nlp_csanmt_translation_zh2en"}),
                "show_debug": ("BOOLEAN", {"default": False}),
                }
            }

    RETURN_TYPES = ("AnyText_Loader", )
    RETURN_NAMES = ("AnyText_Loader", )
    FUNCTION = "AnyText_loader_fn"
    CATEGORY = "ExtraModels/AnyText"
    TITLE = "AnyText Loader"

    def AnyText_loader_fn(self, font, ckpt_name, clip, clip_path_or_repo_id, translator, translator_path_or_repo_id, show_debug):
        font_path = os.path.join(comfyui_models_dir, "fonts", font)
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        cfg_path = os.path.join(current_directory, 'models_yaml', 'anytext_sd15.yaml')
        if clip_path_or_repo_id == "":
            clip_path = os.path.join(comfyui_models_dir, "clip", clip)
        else:
            if clip != 'Auto_DownLoad':
                clip_path = os.path.join(comfyui_models_dir, "clip", clip)
            else:
                clip_path = clip_path_or_repo_id
        if translator_path_or_repo_id == "":
            translator_path = os.path.join(comfyui_models_dir, "prompt_generator", translator)
        else:
            if translator != 'Auto_DownLoad':
                translator_path = os.path.join(comfyui_models_dir, "prompt_generator", translator)
            else:
                translator_path = translator_path_or_repo_id
        
        #将输入参数合并到一个参数里面传递到.nodes
        loader = (font_path + "|" + str(ckpt_path) + "|" + clip_path + "|" + translator_path + "|" + cfg_path)
        #按|分割
        # loader_s = loader.split("|")
        
        if show_debug == True:
            print("\033[93mloader(合并后的4个输入参数，传递给.nodes):", loader, "\033[0m\n")
            # print("\033[93mfont_path--loader[0]:", loader_s[0], "\033[0m\n")
            # print("\033[93mckpt_path--loader[1]:", loader_s[1], "\033[0m\n")
            # print("\033[93mclip_path--loader[2]:", loader_s[2], "\033[0m\n")
            # print("\033[93mtranslator_path--loader[3]:", loader_s[3], "\033[0m\n")
            print("\033[93mfont_path(字体):", font_path, "\033[0m\n")
            print("\033[93mckpt_path(AnyText模型):", ckpt_path, "\033[0m\n")
            print("\033[93mclip_path(clip模型):", clip_path, "\033[0m\n")
            print("\033[93mtranslator_path(翻译模型):", translator_path, "\033[0m\n")
            print("\033[93myaml_file(yaml配置文件):", cfg_path, "\033[0m\n")
            
        #将未分割参数写入txt，然后读取传递到到.nodes
        with open(temp_txt_path, "w", encoding="UTF-8") as text_file:
            text_file.write(loader)
        with open(temp_txt_path, "r", encoding="UTF-8") as f:
            return (f.read(), )
        # return (all)

class AnyText_Pose_IMG:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {
                        "image": (sorted(files), {"image_upload": True}),
                        "seed": ("INT", {"default": 9999, "min": -1, "max": 99999999}),
                        
                        },
                }

    CATEGORY = "ExtraModels/AnyText"
    RETURN_TYPES = (
        "ref", 
        "pos", 
        "INT", 
        "INT", 
        # "pos", 
        "IMAGE")
    RETURN_NAMES = (
        "ori_img", 
        "comfy_mask_pos_img", 
        "width", 
        "height", 
        # "gr_mask_pose_img", 
        "mask_img")
    FUNCTION = "AnyText_Pose_IMG"
    TITLE = "AnyText Pose IMG"
    
    def AnyText_Pose_IMG(self, image, seed):
        image_path = folder_paths.get_annotated_filepath(image)
        # comfy_mask_pos_img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy_mask_pos_img.png")
        comfy_mask_pos_img_path = os.path.join(temp_img_path)
        img = node_helpers.pillow(Image.open, image_path)
        # width = img.width
        # height = img.height
        width, height = img.size
        if width%64 == 0 and height%64 == 0:
            pass
        else:
            raise Exception(f"Input pos_img resolution must be multiple of 64(输入的pos_img图片分辨率必须为64的倍数).\n")
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            # output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            # output_image = output_images[0]
            output_mask = output_masks[0]
        invert_mask = 1.0 - output_mask
        inverted_mask_image = invert_mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        i = 255. * inverted_mask_image.cpu().numpy()[0]
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        print("\033[93mInput img Resolution<=768x768 Recommended(输入图像分辨率,建议<=768x768):", width, "x", height, "\033[0m")
        img.save(temp_img_path)

        return (
            image_path, 
            comfy_mask_pos_img_path, 
            width, 
            height, 
            # gr_mask_pose_image_path, 
            inverted_mask_image)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

from modelscope.utils.constant import Tasks
class AnyText_translator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (['cpu', 'gpu'] , {"default": "gpu"}),
                "prompt": ("STRING", {"default": "这里是单批次翻译文本输入。声明补充说，沃伦的同事都深感震惊，并且希望他能够投案自首。尽量输入单句文本，如果是多句长文本建议人工分句，否则可能出现漏译或未译等情况！！！", "multiline": True}),
                "Batch_prompt": ("STRING", {"default": "这里是多批次翻译文本输入，使用换行进行分割。\n天上掉馅饼啦，快去看超人！！！\n飞流直下三千尺，疑似银河落九天。\n启用Batch_Newline表示输出的翻译会按换行输入进行二次换行,否则是用空格合并起来的整篇文本。", "multiline": True}),
                "Batch_Newline" :("BOOLEAN", {"default": True}),
                "if_Batch": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("中译英结果",)
    CATEGORY = "ExtraModels/AnyText"
    FUNCTION = "AnyText_translator"
    TITLE = "AnyText中译英-阿里达摩院damo/nlp_csanmt_translation_zh2en"

    def AnyText_translator(self, prompt, Batch_prompt, if_Batch, device, Batch_Newline):
        # 使用换行(\n)作为分隔符
        Batch_prompt = Batch_prompt.split("\n")  
        if if_Batch == True:
            input_sequence = Batch_prompt
            # 用特定的连接符<SENT_SPLIT>，将多个句子进行串联
            input_sequence = '<SENT_SPLIT>'.join(input_sequence)
        else:
            input_sequence = prompt
        if os.access(os.path.join(comfyui_models_dir, "prompt_generator", "nlp_csanmt_translation_zh2en", "tf_ckpts", "ckpt-0.data-00000-of-00001"), os.F_OK):
            zh2en_path = os.path.join(comfyui_models_dir, 'prompt_generator', 'nlp_csanmt_translation_zh2en')
        else:
            zh2en_path = "damo/nlp_csanmt_translation_zh2en"
        pipeline_ins = pipeline(task=Tasks.translation, model=zh2en_path, device=device)
        outputs = pipeline_ins(input=input_sequence)
        if if_Batch == True:
            results = outputs['translation'].split('<SENT_SPLIT>')
            if Batch_Newline == True:
                results = '\n\n'.join(results)
            else:
                results = ' '.join(results)
        else:
            results = outputs['translation']
        with open(temp_txt_path, "w", encoding="UTF-8") as text_file:
            text_file.write(results)
        with open(temp_txt_path, "r", encoding="UTF-8") as f:
            return (f.read(), )

class AnyText_SavedModel_translator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "声明补充说，沃伦的同事都深感震惊，并且希望他能够投案自首。\n需要手动转换模型，请勿使用此节点。\n句子之间最好用换行，否则容易误翻译。", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("中译英结果",)
    CATEGORY = "ExtraModels/AnyText"
    FUNCTION = "AnyText_translator"
    TITLE = "AnyText中译英-SavedModel(不建议)"

    def AnyText_translator(self, prompt):
        result = nlp_csanmt_translation_zh2en(prompt)
        return result

import tensorflow
from modelscope.utils.config import Config
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
from subword_nmt import apply_bpe
from modelscope.outputs import OutputKeys
import time
def nlp_csanmt_translation_zh2en(prompt):
    if tensorflow.__version__ >= '2.0':
        tf = tensorflow.compat.v1
        tf.disable_eager_execution()
    model_dir = os.path.join(folder_paths.models_dir, "prompt_generator", "nlp_csanmt_translation_zh2en")

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

    _punct_normalizer = MosesPunctNormalizer(lang=_src_lang)
    _tok = MosesTokenizer(lang=_src_lang)
    _detok = MosesDetokenizer(lang=_tgt_lang)
    _bpe = apply_bpe.BPE(open(_src_bpe_path, encoding='UTF-8'))

    #文本encode：
    # input = ["这里是多批次翻译文本输入，使用换行进行分割。", "尽量输入单句文本，如果是多句长文本建议人工分句，否则可能出现漏译或未译等情况！！！"]
    input = [prompt]

    input = [_punct_normalizer.normalize(item) for item in input]

    aggressive_dash_splits = True

    if (_src_lang in ['es', 'fr'] and _tgt_lang == 'zh') or (_src_lang == 'zh' and _tgt_lang in ['es', 'fr']):
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
    sttime = time.time()
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
    result = {OutputKeys.TRANSLATION: translation_out}
    results = result['translation']
    endtime = time.time()
    print("\033[93m加载模型之后翻译耗时：\033[0m", endtime - sttime)
    with open(temp_txt_path, "w", encoding="UTF-8") as text_file:
            text_file.write(results)
    with open(temp_txt_path, "r", encoding="UTF-8") as f:
            return (f.read(), )
    #以上就是一整套本地调用翻译的全部流程，将它们按顺序放在一整个脚本中就可以顺利翻译了。
    
    #部署(Flask)
# import tensorflow as tf
# from modelscope.models.base import Model
# from modelscope.utils.constant import ModelFile, Tasks
# from modelscope.utils.config import Config
# from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
# from subword_nmt import apply_bpe
# from modelscope.outputs import OutputKeys
# import numpy as np
# from loguru import logger
# from flask import Flask, request

# if tf.__version__ >= '2.0':
#     tf = tf.compat.v1
#     tf.disable_eager_execution()

# app = Flask(__name__)
# def load_model():
#     cfg_dir = "./model/damo/nlp_csanmt_translation_en2zh/configuration.json"
#     cfg = Config.from_file(cfg_dir)
#     # model = tf.saved_model.load("/path/to/your/exported/model")
#     src_vocab_dir = "/home/sc/vscode/trans/model/damo/nlp_csanmt_translation_en2zh/src_vocab.txt"
#     # _src_vocab 是一个字典，key是英文src_vocab.txt中的一行，value是index，length = 49998, _trg_vocab同理

#     global _src_vocab, _trg_vocab
#     _src_vocab = dict([
#                 (w.strip(), i) for i, w in enumerate(open(src_vocab_dir))
#             ])
#     trg_vocab_dir = "/home/sc/vscode/trans/model/damo/nlp_csanmt_translation_en2zh/trg_vocab.txt"
#     _trg_vocab = dict([
#                 (i, w.strip()) for i, w in enumerate(open(trg_vocab_dir))
#             ])

#     global _src_lang, _tgt_lang
#     _src_lang = cfg['preprocessor']['src_lang'] #en
#     _tgt_lang = cfg['preprocessor']['tgt_lang'] #zh

#     _src_bpe_path = "/home/sc/vscode/trans/model/damo/nlp_csanmt_translation_en2zh/bpe.en"
#     global _punct_normalizer, _tok, _detok, _bpe

#     _punct_normalizer = MosesPunctNormalizer(lang=_src_lang)
#     _tok = MosesTokenizer(lang=_src_lang)
#     _detok = MosesDetokenizer(lang=_tgt_lang)
#     _bpe = apply_bpe.BPE(open(_src_bpe_path))

    
#     global sess
#     tf_config = tf.ConfigProto(allow_soft_placement=True)
#     sess = tf.Session(graph=tf.Graph(), config=tf_config)
#     MetaGraphDef = tf.saved_model.loader.load(sess, ['serve'], '/home/sc/vscode/trans/CSANMT')

#     SignatureDef_map = MetaGraphDef.signature_def
#     SignatureDef = SignatureDef_map['translation_signature']
#     X_TensorInfo = SignatureDef.inputs['input_wids']
#     y_TensorInfo = SignatureDef.outputs['output_seqs']
#     global X, y
#     X = tf.saved_model.utils.get_tensor_from_tensor_info(X_TensorInfo, sess.graph)
#     y = tf.saved_model.utils.get_tensor_from_tensor_info(y_TensorInfo, sess.graph)

    
#     # return sess

# def preprocess_input(data):
#     input = [_punct_normalizer.normalize(item) for item in data]

#     aggressive_dash_splits = True
#     if (_src_lang in ['es', 'fr'] and _tgt_lang == 'en') or (_src_lang == 'en' and _tgt_lang in ['es', 'fr']):
#         aggressive_dash_splits = False
    
#     input_tok = [
#                 _tok.tokenize(
#                     item,
#                     return_str=True,
#                     aggressive_dash_splits=aggressive_dash_splits)
#                 for item in input
#             ]
#     input_bpe = [
#             _bpe.process_line(item).strip().split() for item in input_tok
#         ]
#     MAX_LENGTH = max([len(item) for item in input_bpe])

#     input_ids = np.array([[
#             _src_vocab[w] if w in _src_vocab else
#             cfg['model']['src_vocab_size'] - 1 for w in item
#         ] + [0] * (MAX_LENGTH - len(item)) for item in input_bpe])
    
#     return input_ids

# def postprocess(data):
#     x, y, z = data.shape

#     translation_out = []
#     for i in range(x):
#         output_seqs = data[i]
#         wids = list(output_seqs[0]) + [0]
#         wids = wids[:wids.index(0)]
#         translation = ' '.join([
#             _trg_vocab[wid] if wid in _trg_vocab else '<unk>'
#             for wid in wids
#         ]).replace('@@ ', '').replace('@@', '')
#         translation_out.append(_detok.detokenize(translation.split()))
#     translation_out = ''.join(translation_out)
#     return {OutputKeys.TRANSLATION: translation_out}

# @app.route('/predict', methods=['POST'])
# def predict():
#     logger.info("接受到POST请求，开始翻译...")
#     data = request.get_json()
#     input_ids = preprocess_input(data['text'])
#     outputs = sess.run(y, feed_dict={X: input_ids})
#     return postprocess(outputs)

# if __name__ == '__main__':
#     load_model()
#     app.run(port=6006)

#客户端
# import requests
# import json
# import time

# url = "http://127.0.0.1:6006/predict"
# 准备请求数据
# data = {"text":["How are you?", "What's wrong with you?", "May I help you?", "Today is a good day!"]}

# json_data = json.dumps(data)

# start = time.time()
# # 发送 POST 请求
# response = requests.post(url, data=json_data, headers={"Content-Type": "application/json"})
# end = time.time()
# # 获取并解码返回的数据
# if response.status_code == 200:
#     returned_data = response.content.decode("unicode_escape")
#     print("Returned Data:", returned_data)
#     print("time is ", end - start)
# else:
#     print("Error:", response.status_code, response.text)
    
class Create_SavedModel:
    @classmethod
    def INPUT_TYPES(cls):
        
        translator_list = os.listdir(os.path.join(folder_paths.models_dir, "prompt_generator"))
        translator_list.insert(0, "Auto_DownLoad")
        
        return {
            "required": {
                "translator_local_path": (translator_list, ),
                "translator_path_or_repo_id": ("STRING", {"default": "damo/nlp_csanmt_translation_zh2en"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Model Path",)
    CATEGORY = "ExtraModels/AnyText"
    FUNCTION = "SavedModel"
    TITLE = "AnyText Create SavedModel(Not Recommended-不建议)"

    def SavedModel(self, translator_local_path, translator_path_or_repo_id):
        if translator_path_or_repo_id == "":
            translator_path = os.path.join(comfyui_models_dir, "prompt_generator", translator_local_path)
        else:
            if translator_local_path != 'Auto_DownLoad':
                translator_path = os.path.join(comfyui_models_dir, "prompt_generator", translator_local_path)
            else:
                translator_path = translator_path_or_repo_id
        Save = SavedModel(translator_path)
        result = os.path.join(translator_path, "CSANMT")
        return result

#将模型转换为SavedModel格式
def SavedModel(model_path):
    from modelscope.models import Model
    from modelscope.exporters import TfModelExporter
    model = Model.from_pretrained(model_path)
    output_files = TfModelExporter.from_model(model).export_saved_model(output_dir=os.path.join(model_path, "CSANMT"))
    print(output_files) # {'model': '/tmp'}
    output_dir=os.path.join(model_path, "CSANMT")
    return output_dir

# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "AnyText_loader": AnyText_loader,
    "AnyText_Pose_IMG": AnyText_Pose_IMG,
    "AnyText_translator": AnyText_translator,
    "AnyText_SavedModel_translator": AnyText_SavedModel_translator,
    "Create_SavedModel": Create_SavedModel,
}