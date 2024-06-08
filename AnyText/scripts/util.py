import datetime
import os
import cv2


def save_images(img_list, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    folder_path = os.path.join(folder, date_str)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    time_str = now.strftime("%H_%M_%S")
    for idx, img in enumerate(img_list):
        image_number = idx + 1
        filename = f"{time_str}_{image_number}.jpg"
        save_path = os.path.join(folder_path, filename)
        cv2.imwrite(save_path, img[..., ::-1])


def check_channels(image):
    channels = image.shape[2] if len(image.shape) == 3 else 1
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif channels > 3:
        image = image[:, :, :3]
    return image


def resize_image(img, max_length=768):
    height, width = img.shape[:2]
    max_dimension = max(height, width)

    if max_dimension > max_length:
        scale_factor = max_length / max_dimension
        new_width = int(round(width * scale_factor))
        new_height = int(round(height * scale_factor))
        new_size = (new_width, new_height)
        img = cv2.resize(img, new_size)
    height, width = img.shape[:2]
    img = cv2.resize(img, (width-(width % 64), height-(height % 64)))
    return img

import tensorflow
from modelscope.utils.config import Config
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
from subword_nmt import apply_bpe
from modelscope.outputs import OutputKeys
import time
import folder_paths
import numpy as np
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
    endtime = time.time()
    print("加载模型之后翻译耗时：", endtime - sttime)
    return result
    #以上就是一整套本地调用翻译的全部流程，将它们按顺序放在一整个脚本中就可以顺利翻译了。