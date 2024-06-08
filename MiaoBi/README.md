## sd1.5中文文生图-妙笔-非官方Diffusers Pipeline & Hidiffusion(Higher Resolution) & ResAdapter(Higher Resolution)实现,Unofficial implementation of MiaoBi.  |  [Index-目录](../README.md)

[@ShineChen1024](https://github.com/ShineChen1024)

- https://huggingface.co/ShineChen1024/MiaoBi
- https://github.com/ShineChen1024/MiaoBi

# 官方介绍：

## MiaoBi-beta

- 这是妙笔的测试版本。妙笔，一个中文文生图模型，与经典的stable-diffusion 1.5版本拥有一致的结构，兼容现有的lora，controlnet，T2I-Adapter等主流插件及其权重。
- This is the beta version of MiaoBi, a chinese text-to-image model, following the classical structure of sd-v1.5, compatible with existing mainstream plugins such as Lora, Controlnet, T2I Adapter, etc.

## Instructions:

### MiaoBi SingleFile:

- Download all files of [text_encoder](https://huggingface.co/ShineChen1024/MiaoBi/tree/main/miaobi_beta0.9/text_encoder) and put them in a folder. Then rename this folder to "MiaoBi_clip" or any name U like. Finally move this folder into **ComfyUI\models\clip**.

![](./assets/MiaoBi_clip.jpg)
### MiaoBi Diffsuers SD15:

- Models will automatically download into **ComfyUI\models\diffusers** when node execute.

## MiaoBi-beta-v0.9 Chinese Example

一只精致的陶瓷猫咪雕像，全身绘有精美的传统花纹，眼睛仿佛会发光。

![](./assets/fig1.jpg)

- 动漫风格的风景画，有山脉、湖泊，也有繁华的小镇子，色彩鲜艳，光影效果明显。
- 极具真实感的复杂农村的漂亮年轻女人肖像。
- 红烧狮子头
- 车水马龙的上海街道，春节，舞龙舞狮。
- 枯藤老树昏鸦，小桥流水人家。水墨画。
- 飞流直下三千尺，疑似银河落九天。

## MiaoBi workflow:

![](./assets/MiaoBi-wf.png)

# Limitations

妙笔的训练数据包含Laion-5B中的中文子集（经过清洗过滤），Midjourney相关的开源数据（将英文提示词翻译成中文），以及我们收集的一批数十万的caption数据。由于整个数据集大量缺少成语与古诗词数据，所以对成语与古诗词的理解可能存在偏差，对中国的名胜地标建筑数据的缺少以及大量的英译中数据，可能会导致出现一些对象的混乱。如果有以上较高数据质量的伙伴，希望能完善此项目，请与我们联系，我们将根据提供的数据训练全新的版本。妙笔Beta0.9在8张4090显卡上完成训练，我们正在拓展我们的机器资源来训练SDXL来获得更优的结果，敬请期待。

Due to limitations in computing power and the size of Chinese datasets, the performance of Miaobi may be inferior to commercial models at this stage. We are expanding our computing resources and collecting larger scale data, looking forward to the future performance of Miaobi.

# 感谢以下插件的作者、comfyui以及所有相关项目的贡献者:

[ltdrdata](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) | [florestefano1975](https://github.com/florestefano1975/ComfyUI-HiDiffusion) | [blepping](https://github.com/blepping/ComfyUI-ApplyResAdapterUnet)
| [Jannchie](https://github.com/Jannchie/ComfyUI-J)
