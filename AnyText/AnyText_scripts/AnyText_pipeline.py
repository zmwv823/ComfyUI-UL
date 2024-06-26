import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import random
import re
import numpy as np
import cv2
import einops
import time
from PIL import ImageFont
from .cldm.model import create_model, load_state_dict
from .cldm.ddim_hacked import DDIMSampler
from .AnyText_t3_dataset import draw_glyph, draw_glyph2
from .AnyText_pipeline_util import check_channels, resize_image
from pytorch_lightning import seed_everything
# from modelscope.hub.snapshot_download import snapshot_download
from .AnyText_bert_tokenizer import BasicTokenizer
import folder_paths
from huggingface_hub import hf_hub_download
from ...DataProcess.utils import SavedModel_Translator, t5_translate_en_ru_zh, nlp_csanmt_translation_zh2en
from comfy.utils import ProgressBar

checker = BasicTokenizer()
BBOX_MAX_NUM = 8
PLACE_HOLDER = '*'
max_chars = 20

comfyui_models_dir = folder_paths.models_dir
class AnyText_Pipeline():
    def __init__(self, ckpt_path, clip_path, translator, cfg_path, use_translator, device, use_fp16, all_to_device):
        self.device = device
        self.use_fp16 = use_fp16
        self.use_translator = use_translator
        self.translator_path = translator
        self.cfg_path = cfg_path
        self.all_to_device = all_to_device
        if ckpt_path != 'None':
            self.ckpt_path = ckpt_path
        else:
            #第一个方法为从魔搭modelscope(https://modelscope.cn/models/iic/cv_anytext_text_generation_editing/)下载v1.1.1版本下FP32版本的anytext_v1.1.ckpt到指定文件夹ComfyUI\models\checkpoints\15。第二个方法从笑脸huggingface(https://huggingface.co/Sanster/AnyText)下载FP16版本的pytorch_model.fp16.safetensors到指定文件夹ComfyUI\models\checkpoints\15，然后重命名为anytext_v1.1.safetensors
            # ckpt_path = model_file_download(model_id='damo/cv_anytext_text_generation_editing',file_path='anytext_v1.1.ckpt', cache_dir=os.path.join(folder_paths.models_dir, "checkpoints", "15"), revision='v1.1.1')
            if os.access(os.path.join(comfyui_models_dir, "checkpoints", "15", "anytext_v1.1.safetensors"), os.F_OK):
                self.ckpt_path = os.path.join(comfyui_models_dir, "checkpoints", "15", "anytext_v1.1.safetensors")
            else:
                hf_hub_download(repo_id="Sanster/AnyText", filename="pytorch_model.fp16.safetensors",local_dir=os.path.join(comfyui_models_dir, "checkpoints", "15"))
                old_file = os.path.join(comfyui_models_dir, "checkpoints", "15", "pytorch_model.fp16.safetensors")
                new_file = os.path.join(comfyui_models_dir, "checkpoints", "15", "anytext_v1.1.safetensors")
                os.rename(old_file, new_file)
                self.ckpt_path = new_file
                
        if "Auto_DownLoad" not in clip_path:
            self.clip_path = clip_path
        else:
            self.clip_path = "openai/clip-vit-large-patch14"
        
        self.model = create_model(self.cfg_path, cond_stage_path=self.clip_path, use_fp16=self.use_fp16)
        if self.use_fp16:
            # self.model = self.model.eval().half().to(self.device)
            self.model = self.model.half().to(self.device)
        if self.all_to_device ==True:
            self.model.load_state_dict(load_state_dict(self.ckpt_path, location=device), strict=False)
        else:
            self.model.load_state_dict(load_state_dict(self.ckpt_path, location='cpu'), strict=False)
        self.ddim_sampler = DDIMSampler(self.model, device=self.device)
        
        if self.use_translator == True:
            if translator == 'damo/nlp_csanmt_translation_zh2en':
                self.trans_pipe = 'damo/nlp_csanmt_translation_zh2en'
            elif translator == "utrobinmv/t5_translate_en_ru_zh_base_200":
                self.trans_pipe = 'utrobinmv/t5_translate_en_ru_zh_base_200'
            elif translator == "utrobinmv/t5_translate_en_ru_zh_large_1024":
                self.trans_pipe = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
            elif translator == "utrobinmv/t5_translate_en_ru_zh_small_1024":
                self.trans_pipe = 'utrobinmv/t5_translate_en_ru_zh_small_1024'
            else:
                self.trans_pipe = 'SavedModel_Translation'
        else:
            self.trans_pipe = None
    
    def __call__(self, input_tensor, font_path, **forward_params):
        # self.use_fp16 = use_fp16
        if "Auto_DownLoad" not in font_path:
            font_path = font_path
        else:
            if os.access(os.path.join(comfyui_models_dir, "fonts", "SourceHanSansSC-Medium.otf"), os.F_OK):
                font_path = os.path.join(comfyui_models_dir, "fonts", "SourceHanSansSC-Medium.otf")
            else:
                hf_hub_download(repo_id="Sanster/AnyText", filename="SourceHanSansSC-Medium.otf",local_dir=os.path.join(comfyui_models_dir, "fonts"))
                font_path = os.path.join(comfyui_models_dir, "fonts", "SourceHanSansSC-Medium.otf")
        self.font = ImageFont.truetype(font_path, size=60, encoding='utf-8')
        tic = time.time()
        str_warning = ''
        # get inputs
        seed = input_tensor.get('seed', -1)
        if seed == -1:
            seed = random.randint(0, 99999999)
        seed_everything(seed)
        prompt = input_tensor.get('prompt')
        draw_pos = input_tensor.get('draw_pos')
        ori_image = input_tensor.get('ori_image')

        mode = forward_params.get('mode')
        device = forward_params.get('device')
        use_fp16 = forward_params.get('use_fp16')
        Random_Gen = forward_params.get('Random_Gen')
        sort_priority = forward_params.get('sort_priority', '↕')
        show_debug = forward_params.get('show_debug', False)
        revise_pos = forward_params.get('revise_pos', False)
        img_count = forward_params.get('image_count', 1)
        ddim_steps = forward_params.get('ddim_steps', 20)
        w = forward_params.get('image_width', 512)
        h = forward_params.get('image_height', 512)
        strength = forward_params.get('strength', 1.0)
        cfg_scale = forward_params.get('cfg_scale', 9.0)
        eta = forward_params.get('eta', 0.0)
        a_prompt = forward_params.get('a_prompt', 'best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks')
        n_prompt = forward_params.get('n_prompt', 'low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture')
        
        prompt, texts = self.modify_prompt(prompt, device)
        if prompt is None and texts is None:
            return None, -1, "You have input Chinese prompt but the translator is not loaded!", ""
        n_lines = len(texts)
        if mode in ['text-generation', 'gen']:
            if Random_Gen == True:
                edit_image = np.ones((h, w, 3)) * 127.5  # empty mask image
                edit_image = resize_image(edit_image, max_length=768)
                h, w = edit_image.shape[:2]
            else:
                edit_image = cv2.imread(draw_pos)[..., ::-1]
                edit_image = resize_image(edit_image, max_length=768)
                h, w = edit_image.shape[:2]
                edit_image = np.ones((h, w, 3)) * 127.5  # empty mask image
        elif mode in ['text-editing', 'edit']:
            if draw_pos is None or ori_image is None:
                return None, -1, "Reference image and position image are needed for text editing!", ""
            if isinstance(ori_image, str):
                ori_image = cv2.imread(ori_image)[..., ::-1]
                assert ori_image is not None, f"Can't read ori_image image from{ori_image}!"
            elif isinstance(ori_image, torch.Tensor):
                ori_image = ori_image.cpu().numpy()
            else:
                assert isinstance(ori_image, np.ndarray), f'Unknown format of ori_image: {type(ori_image)}'
            edit_image = ori_image.clip(1, 255)  # for mask reason
            edit_image = check_channels(edit_image)
            edit_image = resize_image(edit_image, max_length=768)  # make w h multiple of 64, resize if w or h > max_length
            h, w = edit_image.shape[:2]  # change h, w by input ref_img
        # preprocess pos_imgs(if numpy, make sure it's white pos in black bg)
        if draw_pos is None:
            pos_imgs = np.zeros((w, h, 1))
        if isinstance(draw_pos, str):
            draw_pos = cv2.imread(draw_pos)[..., ::-1]
            draw_pos = resize_image(draw_pos, max_length=768)
            draw_pos = cv2.resize(draw_pos, (w, h))
            assert draw_pos is not None, f"Can't read draw_pos image from{draw_pos}!"
            pos_imgs = 255-draw_pos
        elif isinstance(draw_pos, torch.Tensor):
            pos_imgs = draw_pos.cpu().numpy()
        else:
            assert isinstance(draw_pos, np.ndarray), f'Unknown format of draw_pos: {type(draw_pos)}'
        pos_imgs = pos_imgs[..., 0:1]
        pos_imgs = cv2.convertScaleAbs(pos_imgs)
        _, pos_imgs = cv2.threshold(pos_imgs, 254, 255, cv2.THRESH_BINARY)
        # seprate pos_imgs
        pos_imgs = self.separate_pos_imgs(pos_imgs, sort_priority)
        if len(pos_imgs) == 0:
            pos_imgs = [np.zeros((h, w, 1))]
        if len(pos_imgs) < n_lines:
            if n_lines == 1 and texts[0] == ' ':
                pass  # text-to-image without text
            else:
                return None, -1, f'Found {len(pos_imgs)} positions that < needed {n_lines} from prompt, check and try again!', ''
        elif len(pos_imgs) > n_lines:
            str_warning = f'Warning: found {len(pos_imgs)} positions that > needed {n_lines} from prompt.'
        # get pre_pos, poly_list, hint that needed for anytext
        pre_pos = []
        poly_list = []
        for input_pos in pos_imgs:
            if input_pos.mean() != 0:
                input_pos = input_pos[..., np.newaxis] if len(input_pos.shape) == 2 else input_pos
                poly, pos_img = self.find_polygon(input_pos)
                pre_pos += [pos_img/255.]
                poly_list += [poly]
            else:
                pre_pos += [np.zeros((h, w, 1))]
                poly_list += [None]
        np_hint = np.sum(pre_pos, axis=0).clip(0, 1)
        # prepare info dict
        info = {}
        info['glyphs'] = []
        info['gly_line'] = []
        info['positions'] = []
        info['n_lines'] = [len(texts)]*img_count
        gly_pos_imgs = []
        for i in range(len(texts)):
            text = texts[i]
            if len(text) > max_chars:
                str_warning = f'"{text}" length > max_chars: {max_chars}, will be cut off...'
                text = text[:max_chars]
            gly_scale = 2
            if pre_pos[i].mean() != 0:
                gly_line = draw_glyph(self.font, text)
                glyphs = draw_glyph2(self.font, text, poly_list[i], scale=gly_scale, width=w, height=h, add_space=False)
                gly_pos_img = cv2.drawContours(glyphs*255, [poly_list[i]*gly_scale], 0, (255, 255, 255), 1)
                if revise_pos:
                    resize_gly = cv2.resize(glyphs, (pre_pos[i].shape[1], pre_pos[i].shape[0]))
                    new_pos = cv2.morphologyEx((resize_gly*255).astype(np.uint8), cv2.MORPH_CLOSE, kernel=np.ones((resize_gly.shape[0]//10, resize_gly.shape[1]//10), dtype=np.uint8), iterations=1)
                    new_pos = new_pos[..., np.newaxis] if len(new_pos.shape) == 2 else new_pos
                    contours, _ = cv2.findContours(new_pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if len(contours) != 1:
                        str_warning = f'Fail to revise position {i} to bounding rect, remain position unchanged...'
                    else:
                        rect = cv2.minAreaRect(contours[0])
                        poly = np.int0(cv2.boxPoints(rect))
                        pre_pos[i] = cv2.drawContours(new_pos, [poly], -1, 255, -1) / 255.
                        gly_pos_img = cv2.drawContours(glyphs*255, [poly*gly_scale], 0, (255, 255, 255), 1)
                gly_pos_imgs += [gly_pos_img]  # for show
            else:
                glyphs = np.zeros((h*gly_scale, w*gly_scale, 1))
                gly_line = np.zeros((80, 512, 1))
                gly_pos_imgs += [np.zeros((h*gly_scale, w*gly_scale, 1))]  # for show
            pos = pre_pos[i]
            info['glyphs'] += [self.arr2tensor(glyphs, img_count, use_fp16)]
            info['gly_line'] += [self.arr2tensor(gly_line, img_count, use_fp16)]
            info['positions'] += [self.arr2tensor(pos, img_count, use_fp16)]
        # get masked_x
        masked_img = ((edit_image.astype(np.float32) / 127.5) - 1.0)*(1-np_hint)
        masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.from_numpy(masked_img.copy()).float().to(self.device)
        if self.use_fp16:
            masked_img = masked_img.half()
        encoder_posterior = self.model.encode_first_stage(masked_img[None, ...])
        masked_x = self.model.get_first_stage_encoding(encoder_posterior).detach()
        if self.use_fp16:
            masked_x = masked_x.half()
        info['masked_x'] = torch.cat([masked_x for _ in range(img_count)], dim=0)

        hint = self.arr2tensor(np_hint, img_count, use_fp16)
        cond = self.model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[prompt + ' , ' + a_prompt] * img_count], text_info=info))
        un_cond = self.model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[n_prompt] * img_count], text_info=info))
        shape = (4, h // 8, w // 8)
        self.model.control_scales = ([strength] * 13)
        
        callback = self.callback_util(ddim_steps)
        samples, intermediates = self.ddim_sampler.sample(ddim_steps, img_count,
                                                          shape, cond, verbose=False, eta=eta,
                                                          unconditional_guidance_scale=cfg_scale,
                                                          unconditional_conditioning=un_cond,
                                                        #   callback_steps=1,
                                                          callback=callback,
                                                          )
        if self.use_fp16:
            samples = samples.half()
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(img_count)]
        # if mode == 'edit' and False:  # replace backgound in text editing but not ideal yet
        #     results = [r*np_hint+edit_image*(1-np_hint) for r in results]
        #     results = [r.clip(0, 255).astype(np.uint8) for r in results]
        if len(gly_pos_imgs) > 0 and show_debug:
            glyph_bs = np.stack(gly_pos_imgs, axis=2)
            glyph_img = np.sum(glyph_bs, axis=2) * 255
            glyph_img = glyph_img.clip(0, 255).astype(np.uint8)
            results += [np.repeat(glyph_img, 3, axis=2)]
        input_prompt = prompt
        for t in texts:
            input_prompt = input_prompt.replace('*', f'"{t}"', 1)
        print(f'Prompt: {input_prompt}')
        # debug_info
        if not show_debug:
            debug_info = ''
        else:
            debug_info = f'\033[93mPrompt(提示词): {input_prompt}\n\033[0m \
                           \033[93mSize(尺寸): {w}x{h}\n\033[0m \
                           \033[93mImage Count(生成数量): {img_count}\n\033[0m \
                           \033[93mSeed(种子): {seed}\n\033[0m \
                           \033[93mUse FP16(使用FP16): {self.use_fp16}\n\033[0m \
                           \033[93mCost Time(生成耗时): {(time.time()-tic):.2f}s\033[0m'
        rst_code = 1 if str_warning else 0
        return x_samples, results, rst_code, str_warning, debug_info

    #尝试写进度条，ddim_sampler没有callback_steps参数，无法实现。
    def callback_util(self, ddim_steps, *_):
        pbar = ProgressBar(int(ddim_steps))
        pbar.update(1)
    
    def modify_prompt(self, prompt, device):
        prompt = prompt.replace('“', '"')
        prompt = prompt.replace('”', '"')
        p = '"(.*?)"'
        strs = re.findall(p, prompt)
        if len(strs) == 0:
            strs = [' ']
        else:
            for s in strs:
                prompt = prompt.replace(f'"{s}"', f' {PLACE_HOLDER} ', 1)
        if self.is_chinese(prompt):
            if self.trans_pipe is None:
                return None, None
            old_prompt = prompt
            
            if self.trans_pipe == 'utrobinmv/t5_translate_en_ru_zh_small_1024':
                zh2en_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_small_1024")
                if not os.access(os.path.join(zh2en_path, "model.safetensors"), os.F_OK):
                    zh2en_path = 'utrobinmv/t5_translate_en_ru_zh_small_1024'
                prompt = t5_translate_en_ru_zh('en', prompt + ' .', zh2en_path, device)[0]
                
            elif self.trans_pipe == 'utrobinmv/t5_translate_en_ru_zh_base_200':
                zh2en_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_base_200")
                if not os.access(os.path.join(zh2en_path, "model.safetensors"), os.F_OK):
                    zh2en_path = 'utrobinmv/t5_translate_en_ru_zh_base_200'
                prompt = t5_translate_en_ru_zh('en', prompt + ' .', zh2en_path, device)[0]
                
            elif self.trans_pipe == 'utrobinmv/t5_translate_en_ru_zh_large_1024':
                zh2en_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_large_1024")
                if not os.access(os.path.join(zh2en_path, "model.safetensors"), os.F_OK):
                    zh2en_path = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
                prompt = t5_translate_en_ru_zh('en', prompt + ' .', zh2en_path, device)[0]
                
            elif self.trans_pipe == 'SavedModel_Translation':
                input = (prompt + ' .')
                #获取translation值(tuple)，然后再处理
                prompt = SavedModel_Translator(input, 'Saved_Model_zh2en')['translation']
                
            else:
                zh2en_path = os.path.join(comfyui_models_dir, 'prompt_generator', 'modelscope--damo--nlp_csanmt_translation_zh2en')
                if not os.access(os.path.join(zh2en_path, "tf_ckpts", "ckpt-0.data-00000-of-00001"), os.F_OK):
                    zh2en_path = "damo/nlp_csanmt_translation_zh2en"
                prompt = nlp_csanmt_translation_zh2en(device, prompt + ' .', zh2en_path)['translation']
            print(f'Translate: {old_prompt} --> {prompt}')
        return prompt, strs

    def is_chinese(self, text):
        text = checker._clean_text(text)
        for char in text:
            cp = ord(char)
            if checker._is_chinese_char(cp):
                return True
        return False

    def separate_pos_imgs(self, img, sort_priority, gap=102):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        components = []
        for label in range(1, num_labels):
            component = np.zeros_like(img)
            component[labels == label] = 255
            components.append((component, centroids[label]))
        if sort_priority == '↕':
            fir, sec = 1, 0  # top-down first
        elif sort_priority == '↔':
            fir, sec = 0, 1  # left-right first
        components.sort(key=lambda c: (c[1][fir]//gap, c[1][sec]//gap))
        sorted_components = [c[0] for c in components]
        return sorted_components

    def find_polygon(self, image, min_rect=False):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour = max(contours, key=cv2.contourArea)  # get contour with max area
        if min_rect:
            # get minimum enclosing rectangle
            rect = cv2.minAreaRect(max_contour)
            poly = np.int0(cv2.boxPoints(rect))
        else:
            # get approximate polygon
            epsilon = 0.01 * cv2.arcLength(max_contour, True)
            poly = cv2.approxPolyDP(max_contour, epsilon, True)
            n, _, xy = poly.shape
            poly = poly.reshape(n, xy)
        cv2.drawContours(image, [poly], -1, 255, -1)
        return poly, image

    def arr2tensor(self, arr, bs, use_fp16):
        self.use_fp16 = use_fp16
        arr = np.transpose(arr, (2, 0, 1))
        _arr = torch.from_numpy(arr.copy()).float().to(self.device)
        if self.use_fp16:
            _arr = _arr.half()
        _arr = torch.stack([_arr for _ in range(bs)], dim=0)
        return _arr