from ..UL_common.common import is_module_imported
import os
import folder_paths

# https://www.modelscope.cn/models/iic/cv_dut-raft_video-stabilization_base
# 适用范围
# 该模型适用于多种格式的视频输入，给定抖动视频，生成稳像后的稳定视频；
# 需要注意的是，如果输入视频包含镜头切换，运动轨迹估计将出现错误，导致错误的稳像结果，因此建议输入单一镜头的抖动视频；
# 建议输入横屏视频，由于训练数据中未包含竖屏视频，本算法模型对于竖屏输入（宽<高）的视频稳像表现可能不佳；
# 使用16G显存的显卡测试时，建议的最大输入为 30fps帧率下30s时长的1920x1080分辨率视频。
# 模型局限性以及可能的偏差
# 由于训练数据中未包含竖屏视频，本算法模型对于竖屏输入（宽<高）的视频稳像表现可能不佳；
# 对于快速场景切换的视频输入，本算法模型可能表现不佳。
# 视频稳像后可能会在视频边缘处出现内容缺失（黑边），因此，本模型对输出视频进行了裁剪，再缩放至原视频尺寸。
class UL_Video_Stabilization:
    @classmethod
    def INPUT_TYPES(s):
        model_list = ["damo/cv_dut-raft_video-stabilization_base"]
        return {
            "required": 
                { 
                 "video_path": ("AUDIO_PATH", ), 
                "model": (model_list,{
                    "default": "damo/cv_dut-raft_video-stabilization_base"
                }),
                 "save_to_folder": ("STRING", {"default": r"C:\Users\pc\Desktop"})
                 }
                }
    
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("UL_Video_Stabilization", )
    FUNCTION = "UL_Video_Stabilization"
    CATEGORY = "ExtraModels/UL Video"
    TITLE = "UL Video Video_Stabilization-视频防抖"

    def UL_Video_Stabilization(self, video_path, save_to_folder, model):
        if not is_module_imported('OutputKeys'):
            from modelscope.outputs import OutputKeys
        if not is_module_imported('pipeline'):
            from modelscope.pipelines import pipeline
        if not is_module_imported('Tasks'):
            from modelscope.utils.constant import Tasks
        import shutil
            
        dirname, filename = os.path.split(video_path)    
        # out_put_path = os.path.join(os.path.expanduser("~"), r"Desktop", f'Stabilized_{filename}')
        out_put_path = f'{save_to_folder}\Stabilized_{filename}'
        if model == 'damo/cv_dut-raft_video-stabilization_base':
            cv_dut_raft_video_stabilization_base_path = os.path.join(folder_paths.models_dir, 'audio_checkpoints\ExtraModels\modelscope--damo--cv_dut-raft_video-stabilization_base')
            if not os.access(os.path.join(cv_dut_raft_video_stabilization_base_path, r'ckpt\raft-things.pth'), os.F_OK):
                cv_dut_raft_video_stabilization_base_path = 'damo/cv_dut-raft_video-stabilization_base'
            video_stabilization = pipeline(Tasks.video_stabilization, 
                                model=cv_dut_raft_video_stabilization_base_path)
        out_video_path = video_stabilization(video_path)[OutputKeys.OUTPUT_VIDEO]
        print('Pipeline: the output video path is {}'.format(out_video_path))
        shutil.move(out_video_path, out_put_path)
        return (out_put_path, )
    
NODE_CLASS_MAPPINGS = {
    "UL_Video_Stabilization": UL_Video_Stabilization, 
}