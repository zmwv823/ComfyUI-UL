class UL_AudioPlay_AutoPlay:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio": ("AUDIO",),
              }, 
                }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "UL_AudioPlay_AutoPlay"
    CATEGORY = "ExtraModels/UL"
    TITLE = "UL AudioPlay_AutoPlay"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = ()

    OUTPUT_NODE = True
  
    def UL_AudioPlay_AutoPlay(self,audio):
        return {"ui": {"audio":[audio]}}
    
class UL_AudioPlay_noAutoPlay:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio": ("AUDIO",),
              }, 
                }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "UL_AudioPlay_noAutoPlay"
    CATEGORY = "ExtraModels/UL"
    TITLE = "UL AudioPlay noAutoPlay"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = ()

    OUTPUT_NODE = True
  
    def UL_AudioPlay_noAutoPlay(self,audio):
        return {"ui": {"audio":[audio]}}
    
NODE_CLASS_MAPPINGS = {
    "UL_AudioPlay_AutoPlay": UL_AudioPlay_AutoPlay,
    "UL_AudioPlay_noAutoPlay": UL_AudioPlay_noAutoPlay,
}