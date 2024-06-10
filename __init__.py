# only import if running as a custom node
try:
	pass
except ImportError:
	pass
else:
	NODE_CLASS_MAPPINGS = {}
 
 	# AnyText
	from .AnyText.nodes import NODE_CLASS_MAPPINGS as AnyText_Nodes
	NODE_CLASS_MAPPINGS.update(AnyText_Nodes)
 
	# AnyText_utils
	from .AnyText.utils import NODE_CLASS_MAPPINGS as AnyText_loader_Nodes
	NODE_CLASS_MAPPINGS.update(AnyText_loader_Nodes)
 
	# MiaoBi
	from .MiaoBi.nodes import NODE_CLASS_MAPPINGS as MiaoBi_Nodes
	NODE_CLASS_MAPPINGS.update(MiaoBi_Nodes)
 
	# MiaoBi_utils
	from .MiaoBi.utils import NODE_CLASS_MAPPINGS as MiaoBi_Loader_Nodes
	NODE_CLASS_MAPPINGS.update(MiaoBi_Loader_Nodes)
 
 	# Audio
	from .Audio.nodes import NODE_CLASS_MAPPINGS as UL_StableAudio_Nodes
	NODE_CLASS_MAPPINGS.update(UL_StableAudio_Nodes)
 
 	# Audio_utils
	from .Audio.utils import NODE_CLASS_MAPPINGS as AudioPlayNode_Nodes
	NODE_CLASS_MAPPINGS.update(AudioPlayNode_Nodes)
 
	NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}
	__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# web ui的节点功能
WEB_DIRECTORY = "./web"