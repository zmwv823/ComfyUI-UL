class UL_AudioPlay:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio": ("AUDIO",),
              }, 
                }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "UL_AudioPlay"
    CATEGORY = "ExtraModels/UL"
    TITLE = "UL AudioPlay"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = ()

    OUTPUT_NODE = True
  
    def UL_AudioPlay(self,audio):
        return {"ui": {"audio":[audio]}}
    
NODE_CLASS_MAPPINGS = {
    "UL_AudioPlay": UL_AudioPlay
}