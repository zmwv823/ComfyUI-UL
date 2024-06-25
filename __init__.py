from .UL_common.common import is_folder_exist
import os
import folder_paths

#加载插件前先检查是否在os.listdir里存在自定义目录，没有则自动创建，防止加载节点失败，官方目录可无视。
fonts_path = os.path.join(folder_paths.models_dir, 'fonts')
translator_path = os.path.join(folder_paths.models_dir, 'prompt_generator')
audio_checkpoints = os.path.join(folder_paths.models_dir, 'audio_checkpoints')
if not is_folder_exist(fonts_path):
    os.makedirs(fonts_path)
if not is_folder_exist(translator_path):
    os.makedirs(translator_path)
if not is_folder_exist(audio_checkpoints):
    os.makedirs(audio_checkpoints)

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
	from .Audio.utils import NODE_CLASS_MAPPINGS as UL_Advance_AutoPlay_Nodes
	NODE_CLASS_MAPPINGS.update(UL_Advance_AutoPlay_Nodes)
 
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
 
	NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}
	__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# web ui的节点功能
WEB_DIRECTORY = "./UL_web"