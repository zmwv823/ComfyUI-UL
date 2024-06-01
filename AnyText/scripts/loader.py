import os
import folder_paths

# class AnyText_loader:
#     @classmethod
#     def INPUT_TYPES(s):
#         font_path =os.path.join(folder_paths.models_dir,"fonts")
#         clip_paths = []
#         for search_path in folder_paths.get_folder_paths("clip"):
#             if os.path.exists(search_path):
#                 for root, subdir, files in os.walk(search_path, followlinks=True):
#                     if "config.json" in files:
#                         clip_paths.append(os.path.relpath(root, start=search_path))
#         {
#         "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
#         "clip_path": (clip_paths,),
#         "font": (os.listdir(font_path), ),
#         "text": ("STRING", {"default": "eee"}),
#         }
    
#     RETURN_TYPES = ("STRING",)
#     CATEGORY = "ExtraModels/AnyText"
#     FUNCTION = "paths"
#     TITLE = "AnyText-loader"
    
#     def paths(self, text, cfg, font, ckpt_name, clip_path):
#         for search_path in folder_paths.get_folder_paths("clip"):
#             if os.path.exists(search_path):
#                 path = os.path.join(search_path, clip_path)
#                 if os.path.exists(path):
#                     clip_path = path
#                     break
#         current_directory = os.path.dirname(os.path.abspath(__file__))
#         cfg = os.path.join(current_directory, "models_yaml/anytext_sd15.yaml")
#         clip_path = clip_path
#         font = os.path.join(folder_paths.models_dir,font)
#         ckpt_name = folder_paths.get_full_path("checkpoints", ckpt_name)
#         return (text, cfg, font, ckpt_name, clip_path,)
    
class AnyText_loader:
    @classmethod
    def INPUT_TYPES(cls):
        font_path =os.path.join(folder_paths.models_dir,"fonts")
        clip_paths = []
        for search_path in folder_paths.get_folder_paths("clip"):
            if os.path.exists(search_path):
                for root, subdir, files in os.walk(search_path, followlinks=True):
                    if "config.json" in files:
                        clip_paths.append(os.path.relpath(root, start=search_path))

        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "clip_path": (clip_paths,),
                "font": (os.listdir(font_path), ),
                "text": ("STRING", {"default": "eee"}),
                }
            }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("for_debug", "clip_path", "font", "ckpt_name")
    FUNCTION = "AnyText_loader_fn"
    CATEGORY = "ExtraModels/MiaoBi"
    TITLE = "AnyText-loader"

    def AnyText_loader_fn(self, text, font, ckpt_name, clip_path):
        for search_path in folder_paths.get_folder_paths("clip"):
            if os.path.exists(search_path):
                path = os.path.join(search_path, clip_path)
                if os.path.exists(path):
                    clip_path = path
                    break
        text = text
        clip_path = clip_path
        font = os.path.join(folder_paths.models_dir, "fonts", font)
        ckpt_name = folder_paths.get_full_path("checkpoints", ckpt_name)
        return (text, clip_path, font, ckpt_name)

# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "AnyText_loader": AnyText_loader,
}