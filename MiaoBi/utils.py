import os
import folder_paths

class MiaoBi_Loader:
    @classmethod
    def INPUT_TYPES(s):
        file_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                #os.listdir列出clip文件夹下所有文件及文件夹，输出选择的项目名字非路径，类型为字符串string，使用utf-8开启非英文支持。
                "clip":(os.listdir(os.path.join(folder_paths.models_dir, "clip")), 'utf-8', ),
                #从configs获取所有文件,默认输出字符串值v1-inference_fp16.yaml，定义默认加载哪个yaml文件。
                "config_name": (folder_paths.get_filename_list("configs"),  {"default": "v1-inference_fp16.yaml"}),
                "LoRA_name1": (file_list, ),
                #获取列表值，如果140行启用将添加None到第一位，可以用于开启判定
                "LoRA_switch1": ("BOOLEAN", {"default": False},),
                "lora_weight1": ("FLOAT", {
                    "default": 1,
                    "min": -100,
                    "max": 100, 
                    "step": 0.01
                }),
                "LoRA_name2": (file_list, ),#可以添加None判定
                "LoRA_switch2": ("BOOLEAN", {"default": False},),
                "lora_weight2": ("FLOAT", {
                    "default": 1,
                    "min": -100,
                    "max": 100, 
                    "step": 0.01
                }),
                "LoRA_name3": (file_list, ),#可以添加None判定
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
                # "diffusion_name": (os.listdir(os.path.join(folder_paths.models_dir, "diffusion")), 'utf-8', ),
                # os.listdir列出diffusers文件夹下的所有文件（包括文件夹），使用utf-8开启非英文支持，输出选择的文件夹或者文件的名字，类型字符串string值。
                "diffusion_name": (["ShineChen1024/MiaoBi","SG161222/Realistic_Vision_V6.0_B1_noVAE","Lykon/dreamshaper-8","emilianJR/epiCRealism","Yntec/epiCPhotoGasm","runwayml/stable-diffusion-v1-5","Niggendar/counterfeitv30_fix_fp16","stablediffusionapi/rev-animated","sam749/CyberRealistic-v4-2","Lykon/dreamshaper-8-lcm"], {"default": "ShineChen1024/MiaoBi"}),
                }
        }
         
    RETURN_TYPES = ("MiaoBi_Loader","STRING")
    RETURN_NAMES = ("MiaoBi_Loader","MiaoBi_Loader_text")
    FUNCTION = "MiaoBi_Loader"
    CATEGORY = "ExtraModels/MiaoBi"
    TITLE = "MiaoBi Loader"
    
    def MiaoBi_Loader(self,
                    ckpt_name, #0
                    clip, #1
                    config_name, #2
                    LoRA_name1, #3
                    LoRA_name2, #4
                    LoRA_name3, #5
                    lora_weight1, #6
                    lora_weight2, #7
                    lora_weight3, #8
                    LoRA_switch1, #9
                    LoRA_switch2, #10
                    LoRA_switch3, #11
                    multi_loras_weight, #12
                    diffusion_name, #13
                    ):
        self.MiaoBi_paths = ckpt_name + '|' + clip + '|' + config_name + '|' + LoRA_name1 + '|' + LoRA_name2 + '|' + LoRA_name3 + '|' + str(lora_weight1) + '|' + str(lora_weight2) + '|' + str(lora_weight3) + '|' + str(LoRA_switch1) + '|' + str(LoRA_switch2) + '|' + str(LoRA_switch3) + '|' + str(multi_loras_weight) + '|' + diffusion_name
        print(self.MiaoBi_paths)
        return (self.MiaoBi_paths, self.MiaoBi_paths,)
    
NODE_CLASS_MAPPINGS = {
    "MiaoBi_Loader": MiaoBi_Loader,
}