# My Personal custom-nodes | For Study Purpose Only | Not Recommeded  |  Will Not Accept Any Issue

## Warning: this custom_node takes 0.4s+ for importing when startup ComfyUI.

## 1、[AnyText](./AnyText/README.md) 
- Original Github Repo: [tyxsspa/AnyText](https://github.com/tyxsspa/AnyText)
- Original Modelscope Repo: [iic/cv_anytext_text_generation_editing](https://modelscope.cn/models/iic/cv_anytext_text_generation_editing/summary)
- Use [ComfyUI-AnyText](https://github.com/zmwv823/ComfyUI-AnyText) instead. 
- ![](./AnyText/assets/AnyText-wf.png)

## 2、[MiaoBi](./MiaoBi/README.md)
- Original Github Repo: [ShineChen1024/MiaoBi](https://github.com/ShineChen1024/MiaoBi)
- Original Huggingface Repo: [ShineChen1024/MiaoBi](https://huggingface.co/ShineChen1024/MiaoBi)
- Use [ComfyUI_ExtraModels](https://github.com/city96/ComfyUI_ExtraModels) instead.
- ![](./MiaoBi/assets/MiaoBi-wf.png)

## 3、[Audio](./Audio/README.md)
### Warning: [mono2stereo--function]() works on windows only.
### stable-audio-open-1.0
- Original Github Repo: [stable-audio-open-1.0](https://github.com/Stability-AI/stable-audio-tools)
- Original Huggingface Repo: [stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
### ChatTTS
- Original Github Repo: [ChatTTS](https://github.com/2noise/ChatTTS)
- Original Huggingface Repo: [ChatTTS](https://huggingface.co/2Noise/ChatTTS)
- 如果尾字吞字不读，可以试试结尾加上 [lbreak]
- If the input text is all in English, it is recommended to check disable_normalize
- 'oral' means add filler words, 'laugh' means add laughter, and 'break' means add a pause. (0-10)
### facebook--musicgen-small
- **Need ffmpeg env set.**
- Original Huggingface Repo: [facebook--musicgen-small](https://huggingface.co/facebook/musicgen-small)

## Fork from Github Repo: [shadowcz007/comfyui-sound-lab](https://github.com/shadowcz007/comfyui-sound-lab)
- ![](./Audio/assets/Audio-wf.png)