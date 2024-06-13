import folder_paths

class UL_MiaoBi_Loader:
    @classmethod
    def INPUT_TYPES(s):
        file_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "Apply_Hidiffusion": ("BOOLEAN", {"default": False},),
                "clip_layer":("INT", {"default": -1, "min": -9999, "max": 9999}),
                "LoRA_name1": (file_list, ),
                "LoRA_switch1": ("BOOLEAN", {"default": False},),
                "lora_weight1": ("FLOAT", {
                    "default": 1,
                    "min": -100,
                    "max": 100, 
                    "step": 0.01
                }),
                "LoRA_name2": (file_list, ),
                "LoRA_switch2": ("BOOLEAN", {"default": False},),
                "lora_weight2": ("FLOAT", {
                    "default": 1,
                    "min": -100,
                    "max": 100, 
                    "step": 0.01
                }),
                "LoRA_name3": (file_list, ),
                "LoRA_switch3": ("BOOLEAN", {"default": False},),
                "lora_weight3": ("FLOAT", {
                    "default": 1,
                    "min": -100,
                    "max": 100, 
                    "step": 0.01
                }),
                "multi_loras_weight": ("FLOAT", {
                    "default": 1.00,
                    "min": -999999,
                    "max": 9999999,
                    "step": 0.01
                }),
                }
        }
         
    RETURN_TYPES = ("MiaoBi_Loader", )
    RETURN_NAMES = ("MiaoBi_Loader", )
    FUNCTION = "UL_MiaoBi_Loader"
    CATEGORY = "ExtraModels/MiaoBi"
    TITLE = "UL MiaoBi Loader"
    
    def UL_MiaoBi_Loader(self,
                    Apply_Hidiffusion, #0
                    clip_layer, #1
                    LoRA_name1, #2
                    LoRA_name2, #3
                    LoRA_name3, #4
                    lora_weight1, #5
                    lora_weight2, #6
                    lora_weight3, #7
                    LoRA_switch1, #8
                    LoRA_switch2, #9
                    LoRA_switch3, #10
                    multi_loras_weight, #11
                    ):
        self.MiaoBi_paths = str(Apply_Hidiffusion) + '|' + str(clip_layer) + '|' + LoRA_name1 + '|' + LoRA_name2 + '|' + LoRA_name3 + '|' + str(lora_weight1) + '|' + str(lora_weight2) + '|' + str(lora_weight3) + '|' + str(LoRA_switch1) + '|' + str(LoRA_switch2) + '|' + str(LoRA_switch3) + '|' + str(multi_loras_weight)
        # print(self.MiaoBi_paths)
        return (self.MiaoBi_paths, )
    
NODE_CLASS_MAPPINGS = {
    "UL_MiaoBi_Loader": UL_MiaoBi_Loader,
}