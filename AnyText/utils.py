import os
import folder_paths
import torch
import node_helpers
from PIL import Image, ImageOps, ImageSequence
import hashlib
import numpy as np
import re

current_directory = os.path.dirname(os.path.abspath(__file__))
comfyui_models_dir = folder_paths.models_dir
temp_txt_path = os.path.join(current_directory, "temp_dir", "AnyText_temp.txt")
temp_img_path = os.path.join(current_directory, "temp_dir", "AnyText_manual_mask_pos_img.png")

class UL_AnyText_loader:
    @classmethod
    def INPUT_TYPES(cls):
        font_list = os.listdir(os.path.join(comfyui_models_dir, "fonts"))
        checkpoints_list = folder_paths.get_filename_list("checkpoints")
        clip_list = os.listdir(os.path.join(comfyui_models_dir, "clip"))
        font_list.insert(0, "Auto_DownLoad")
        checkpoints_list.insert(0, "Auto_DownLoad")
        clip_list.insert(0, "Auto_DownLoad")

        return {
            "required": {
                "font": (font_list, ),
                "ckpt_name": (checkpoints_list, ),
                "clip": (clip_list, ),
                "clip_path_or_repo_id": ("STRING", {"default": "openai/clip-vit-large-patch14"}),
                "translator": (["utrobinmv/t5_translate_en_ru_zh_base_200", "utrobinmv/t5_translate_en_ru_zh_large_1024", "damo/nlp_csanmt_translation_zh2en", "SavedModel"],{"default": "t5_translate_en_ru_zh_base_200"}), 
                "show_debug": ("BOOLEAN", {"default": False}),
                }
            }

    RETURN_TYPES = ("AnyText_Loader", )
    RETURN_NAMES = ("AnyText_Loader", )
    FUNCTION = "AnyText_loader_fn"
    CATEGORY = "ExtraModels/UL AnyText"
    TITLE = "UL AnyText Loader"

    def AnyText_loader_fn(self, font, ckpt_name, clip, clip_path_or_repo_id, translator, show_debug):
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
        
        #将输入参数合并到一一起传递到.nodes，此时为字符串，使用特殊符号|拼接，方便后面nodes分割。
        loader = (font_path + "|" + str(ckpt_path) + "|" + clip_path + "|" + translator + "|" + cfg_path)
        print(loader)
        #按|分割
        # loader_s = loader.split("|")
        
        if show_debug == True:
            print(f'\033[93mloader(合并后的4个输入参数，传递给nodes): {loader} \033[0m\n \
                    \033[93mfont_path(字体): {font_path} \033[0m\n \
                    \033[93mckpt_path(AnyText模型): {ckpt_path} \033[0m\n \
                    \033[93mclip_path(clip模型): {clip_path} \033[0m\n \
                    \033[93mtranslator_path(翻译模型): {translator} \033[0m\n \
                    \033[93myaml_file(yaml配置文件): {cfg_path} \033[0m\n')
            # print("\033[93mfont_path--loader[0]: loader_s[0], "\033[0m\n")
            # print("\033[93mckpt_path--loader[1]: loader_s[1], "\033[0m\n")
            # print("\033[93mclip_path--loader[2]: loader_s[2], "\033[0m\n")
            # print("\033[93mtranslator_path--loader[3]: loader_s[3], "\033[0m\n")
            
        #将未分割参数写入txt，然后读取传递到到.nodes。要输出STRING一定要括号加逗号return (STRING, )(否则只能输出第一个字)，这样就可以直接输出文本，不用写入文件再读文件再输出文本。将输出结果STRING值写入到插件下中间文件ComfyUI-UL\AnyText\temp_dir\AnyText_temp.txt内，然后再打开txt文件再输出STRING。
        return (loader, )
        # with open(temp_txt_path, "w", encoding="UTF-8") as text_file:
        #     text_file.write(loader)
        # with open(temp_txt_path, "r", encoding="UTF-8") as f:
        #     return (f.read(), )

class UL_AnyText_Pose_IMG:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {
                        "image": (sorted(files), {"image_upload": True}),
                        },
                }

    CATEGORY = "ExtraModels/UL AnyText"
    RETURN_TYPES = (
        # "ori", 
        # "pos", 
        "AnyText_images",
        "IMAGE")
    RETURN_NAMES = (
        # "ori_img", 
        # "pos_img", 
        "AnyText_images",
        "mask_img")
    FUNCTION = "AnyText_Pose_IMG"
    TITLE = "UL AnyText Pose IMG"
    
    def AnyText_Pose_IMG(self, image):
        ori_image_path = folder_paths.get_annotated_filepath(image)
        pos_img_path = os.path.join(temp_img_path)
        AnyText_images = ori_image_path + '|' + pos_img_path
        img = node_helpers.pillow(Image.open, ori_image_path)
        # width = img.width
        # height = img.height
        width, height = img.size
        # if width%64 == 0 and height%64 == 0:
        #     pass
        # else:
        #     raise Exception(f"Input pos_img resolution must be multiple of 64(输入的pos_img图片分辨率必须为64的倍数).\n")
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
        print("\033[93mInput img Resolution<=768x768(输入图像分辨率):", width, "x", height, "\033[0m")
        img.save(temp_img_path)

        return (
            # ori_image_path, 
            # pos_img_path, 
            AnyText_images,
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

# def replace_between(s, start, end, replacement):
#     # 正则表达式，用以匹配从start到end之间的所有字符
#     pattern = r"%s(.*?)%s" % (re.escape(start), re.escape(end))
#     # 使用re.DOTALL标志来匹配包括换行在内的所有字符
#     return re.sub(pattern, replacement, s, flags=re.DOTALL)

def prompt_replace(prompt):
    #将中文符号“”中的所有内容替换为空内容，防止输入中文被检测到，从而加载翻译模型。
    # prompt = replace_between(prompt, "“", "”", "*")
    prompt = prompt.replace('“', '"')
    prompt = prompt.replace('”', '"')
    p = '"(.*?)"'
    strs = re.findall(p, prompt)
    if len(strs) == 0:
        strs = [' ']
    else:
        for s in strs:
            prompt = prompt.replace(f'"{s}"', f'*', 1)
    return prompt
    

# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "UL_AnyText_loader": UL_AnyText_loader,
    "UL_AnyText_Pose_IMG": UL_AnyText_Pose_IMG,
}