from .UL_common.common import is_folder_exist
import os
import folder_paths
import site

#加载插件前先检查是否在os.listdir里存在自定义目录，没有则自动创建，防止加载节点失败，官方目录可无视。
fonts_path = os.path.join(folder_paths.models_dir, 'fonts')
translator_path = os.path.join(folder_paths.models_dir, 'prompt_generator')
audio_checkpoints = os.path.join(folder_paths.models_dir, 'audio_checkpoints')
audio_output_path = os.path.join(folder_paths.get_output_directory(), 'audio')
comfy_temp_dir = folder_paths.get_temp_directory()
if not is_folder_exist(fonts_path):
    os.makedirs(fonts_path)
if not is_folder_exist(translator_path):
    os.makedirs(translator_path)
if not is_folder_exist(audio_checkpoints):
    os.makedirs(audio_checkpoints)
if not is_folder_exist(audio_output_path):
    os.makedirs(audio_output_path)
if not is_folder_exist(comfy_temp_dir):
    os.makedirs(comfy_temp_dir)

# 将插件内的tts、deepspeed包site-packages添加到到python包site-packages环境
custom_node_dir = os.path.dirname(os.path.abspath(__file__))
now_dir = os.path.join(custom_node_dir, 'audio\site_packages')
# audio_packages_dir = os.path.join(custom_node_dir, 'audio\site_packages')
site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if(site_packages_roots==[]):site_packages_roots=["%s/runtime/Lib/site-packages" % now_dir]
# if(site_packages_roots==[]):site_packages_roots=["%s/runtime/Lib/site-packages" % now_dir % audio_packages_dir]
#os.environ["OPENBLAS_NUM_THREADS"] = "4"
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/ComfyUI-UL.pth" % (site_packages_root), "w") as f:
                f.write(
                    "%s\n"
                    % (now_dir)
                )
                # f.write(
                #     "%s\n%s\n"
                #     % (now_dir, audio_packages_dir)
                # )
            break
        except PermissionError:
            raise PermissionError

if os.path.isfile("%s/ComfyUI-UL.pth" % (site_packages_root)):
    print("\033[93m!!!ComfyUI-UL/Audio/site_packages [packages: TTS] path was added to " + "%s/ComfyUI-UL.pth" % (site_packages_root) 
    + "\nWe can add custom packages into the folder if we don't want to pip install.\nIf meet `No module` error, try restart comfyui.\033[0m")

# only import if running as a custom node
try:
	pass
except ImportError:
	pass
else:
	NODE_CLASS_MAPPINGS = {}
 
 	# AnyText
	from .AnyText.nodes import NODE_CLASS_MAPPINGS as UL_AnyText_Nodes
	NODE_CLASS_MAPPINGS.update(UL_AnyText_Nodes)
 
	# AnyText_utils
	from .AnyText.utils import NODE_CLASS_MAPPINGS as UL_AnyText_loader_Nodes
	NODE_CLASS_MAPPINGS.update(UL_AnyText_loader_Nodes)
 
	# MiaoBi
	from .MiaoBi.nodes import NODE_CLASS_MAPPINGS as UL_MiaoBi_Nodes
	NODE_CLASS_MAPPINGS.update(UL_MiaoBi_Nodes)
 
	# MiaoBi_utils
	from .MiaoBi.utils import NODE_CLASS_MAPPINGS as UL_MiaoBi_Loader_Nodes
	NODE_CLASS_MAPPINGS.update(UL_MiaoBi_Loader_Nodes)
 
 	# Audio
	from .Audio.nodes import NODE_CLASS_MAPPINGS as UL_Audio_Stable_Audio_Open_Nodes
	NODE_CLASS_MAPPINGS.update(UL_Audio_Stable_Audio_Open_Nodes)
 
 	# Audio_utils
	from .Audio.utils import NODE_CLASS_MAPPINGS as UL_Audio_Preview_AutoPlay_Nodes
	NODE_CLASS_MAPPINGS.update(UL_Audio_Preview_AutoPlay_Nodes)
 
  	# UL_common
	from .UL_common.common import NODE_CLASS_MAPPINGS as UL_Text_Input_Nodes
	NODE_CLASS_MAPPINGS.update(UL_Text_Input_Nodes)
 
   	# Data_Procee
	from .DataProcess.nodes import NODE_CLASS_MAPPINGS as UL_DataProcess_t5_translate_en_ru_zh_Nodes
	NODE_CLASS_MAPPINGS.update(UL_DataProcess_t5_translate_en_ru_zh_Nodes)
 
    # Data_Procee_utils
	from .DataProcess.utils import NODE_CLASS_MAPPINGS as UL_DataProcess_Create_SavedModel_Nodes
	NODE_CLASS_MAPPINGS.update(UL_DataProcess_Create_SavedModel_Nodes)
 
    # Video
	from .Video.nodes import NODE_CLASS_MAPPINGS as UL_DataProcess_t5_translate_en_ru_zh_Nodes
	NODE_CLASS_MAPPINGS.update(UL_DataProcess_t5_translate_en_ru_zh_Nodes)
 
     # Image
	# from .Image.nodes import NODE_CLASS_MAPPINGS as UL_Image_cv_resnet_carddetection_Nodes
	# NODE_CLASS_MAPPINGS.update(UL_Image_cv_resnet_carddetection_Nodes)
 
	NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}
	__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# web ui的节点功能
WEB_DIRECTORY = "./UL_web"