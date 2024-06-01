---
license: apache-2.0
language:
- en
library_name: diffusers
pipeline_tag: text-to-image
cite: arxiv.org/abs/2403.02084
---
# ResAdapter Model Card

<div align="center">

[**Project Page**](https://res-adapter.github.io/) **|** [**Paper**](https://arxiv.org/abs/2403.02084) **|** [**Code**](https://github.com/bytedance/res-adapter) **|** [ **Gradio demo**](https://huggingface.co/spaces/jiaxiangc/res-adapter) **|** [**ComfyUI Extension**](https://github.com/jiaxiangc/ComfyUI-ResAdapter)


</div>

## Introduction

We propose ResAdapter, a plug-and-play resolution adapter for enabling any diffusion model generate resolution-free images: no additional training, no additional inference and no style transfer.


<div  align="center">
<img src='https://github.com/bytedance/res-adapter/blob/main/assets/misc/dreamlike1.png?raw=true'>
</div>


## Usage

We provide a standalone [example code](https://github.com/bytedance/res-adapter/blob/main/quicktour.py) to help you quickly use resadapter with diffusion models.

<img src="https://github.com/bytedance/res-adapter/blob/main/assets/misc/dreamshaper_resadapter.png?raw=true" width="100%">
<img src="https://github.com/bytedance/res-adapter/blob/main/assets/misc/dreamshaper_baseline.png?raw=true" width="100%">

Comparison examples (640x384) between resadapter and [dreamshaper-xl-1.0](https://huggingface.co/Lykon/dreamshaper-xl-1-0). Top: with resadapter. Bottom: without resadapter.


```python
# pip install diffusers, transformers, accelerate, safetensors, huggingface_hub
import torch
from torchvision.utils import save_image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

generator = torch.manual_seed(0)
prompt = "portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors"
width, height = 640, 384

# Load baseline pipe
model_name = "lykon-models/dreamshaper-xl-1-0"
pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")

# Inference baseline pipe
image = pipe(prompt, width=width, height=height, num_inference_steps=25, num_images_per_prompt=4, output_type="pt").images
save_image(image, f"image_baseline.png", normalize=True, padding=0)

# Load resadapter for baseline
resadapter_model_name = "resadapter_v1_sdxl"
pipe.load_lora_weights(
    hf_hub_download(repo_id="jiaxiangc/res-adapter", subfolder=resadapter_model_name, filename="pytorch_lora_weights.safetensors"), 
    adapter_name="res_adapter",
    ) # load lora weights
pipe.set_adapters(["res_adapter"], adapter_weights=[1.0])
pipe.unet.load_state_dict(
    load_file(hf_hub_download(repo_id="jiaxiangc/res-adapter", subfolder=resadapter_model_name, filename="diffusion_pytorch_model.safetensors")),
    strict=False,
    ) # load norm weights

# Inference resadapter pipe
image = pipe(prompt, width=width, height=height, num_inference_steps=25, num_images_per_prompt=4, output_type="pt").images
save_image(image, f"image_resadapter.png", normalize=True, padding=0)
```

For more details, please follow the instructions in our [GitHub repository](https://github.com/bytedance/res-adapter). 

## Models
|Models  | Parameters | Resolution Range | Ratio Range | Links |
| --- | --- |--- | --- | --- |
|resadapter_v2_sd1.5| 0.9M | 128 <= x <= 1024 | 0.28 <= r <= 3.5 | [Download](https://huggingface.co/jiaxiangc/res-adapter/tree/main/resadapter_v2_sd1.5)|
|resadapter_v2_sdxl| 0.5M | 256 <= x <= 1536 | 0.28 <= r <= 3.5 | [Download](https://huggingface.co/jiaxiangc/res-adapter/tree/main/resadapter_v2_sdxl)|
|resadapter_v1_sd1.5| 0.9M | 128 <= x <= 1024 | 0.5 <= r <= 2 | [Download](https://huggingface.co/jiaxiangc/res-adapter/tree/main/resadapter_v1_sd1.5)|
|resadapter_v1_sd1.5_extrapolation| 0.9M | 512 <= x <= 1024 | 0.5 <= r <= 2  | [Download](https://huggingface.co/jiaxiangc/res-adapter/tree/main/resadapter_v1_sd1.5_extrapolation)|
|resadapter_v1_sd1.5_interpolation| 0.9M | 128 <= x <= 512 | 0.5 <= r <= 2  | [Download](https://huggingface.co/jiaxiangc/res-adapter/tree/main/resadapter_v1_sd1.5_interpolation)|
|resadapter_v1_sdxl| 0.5M | 256 <= x <= 1536 | 0.5 <= r <= 2  | [Download](https://huggingface.co/jiaxiangc/res-adapter/tree/main/resadapter_v1_sdxl) |
|resadapter_v1_sdxl_extrapolation| 0.5M | 1024 <= x <= 1536 | 0.5 <= r <= 2  | [Download](https://huggingface.co/jiaxiangc/res-adapter/tree/main/resadapter_v1_sdxl_extrapolation) |
|resadapter_v1_sdxl_interpolation| 0.5M | 256 <= x <= 1024 | 0.5 <= r <= 2  | [Download](https://huggingface.co/jiaxiangc/res-adapter/tree/main/resadapter_v1_sdxl_interpolation) |

## ComfyUI Extension

### Text-to-Image

- workflow: [resadapter_text_to_image_workflow](https://github.com/jiaxiangc/ComfyUI-ResAdapter/blob/main/examples/resadapter_text_to_image_workflow.json). 
- models: [dreamlike-diffusion-1.0](https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0)

![](https://github.com/jiaxiangc/ComfyUI-ResAdapter/blob/main/misc/resadapter_text-to-image.png?raw=true)

### ControlNet

- workflow: [resadapter_controlnet_workflow](https://github.com/jiaxiangc/ComfyUI-ResAdapter/blob/main/examples/resadapter_controlnet_workflow.json). 
- models: [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)

![](https://github.com/jiaxiangc/ComfyUI-ResAdapter/blob/main/misc/resadapter_controlnet.png?raw=true)

### IPAdapter

- workflow: [resadapter_ipadapter_workflow](https://github.com/jiaxiangc/ComfyUI-ResAdapter/blob/main/examples/resadapter_ipadapter_workflow.json). 
- models: [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [IP-Adapter](https://huggingface.co/h94/IP-Adapter)

![](https://github.com/jiaxiangc/ComfyUI-ResAdapter/blob/main/misc/resadapter_ipadapter.png?raw=true)

### Accelerate LoRA

- workflow: [resadapter_accelerate_lora_workflow](https://github.com/jiaxiangc/ComfyUI-ResAdapter/blob/main/examples/resadapter_accelerate_lora_workflow.json). 
- models: [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [lcm-lora-sdv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)

![](https://github.com/jiaxiangc/ComfyUI-ResAdapter/blob/main/misc/resadapter_lcm_lora_workflow.png?raw=true)

## Usage Tips

1. If you are not satisfied with interpolation images, try to increase the alpha of resadapter to 1.0.
2. If you are not satisfied with extrapolate images, try to choose the alpha of resadapter in 0.3 ~ 0.7.
3. If you find the images with style conflicts, try to decrease the alpha of resadapter.
4. If you find resadapter is not compatible with other accelerate lora, try to decrease the alpha of resadapter to 0.5 ~ 0.7.

## Citation
If you find ResAdapter useful for your research and applications, please cite us using this BibTeX:
```
@article{cheng2024resadapter,
  title={ResAdapter: Domain Consistent Resolution Adapter for Diffusion Models},
  author={Cheng, Jiaxiang and Xie, Pan and Xia, Xin and Li, Jiashi and Wu, Jie and Ren, Yuxi and Li, Huixia and Xiao, Xuefeng and Zheng, Min and Fu, Lean},
  booktitle={arXiv preprint arxiv:2403.02084},
  year={2024}
}
```
For any question, please feel free to contact us via chengjiaxiang@bytedance.com or xiepan.01@bytedance.com.