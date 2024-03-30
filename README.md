# BraVE
Brain Vision Explainability
使用基于深度学习的方法，对人类的视觉皮层进行功能或者结构的建模


## Module 1: image2caption via CLIP and BLIP
### Setup Environment
```shell
conda create --name CLIP python=3.11
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/openai/CLIP.git
pip install pip install fairscale -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

Run this module: `cd BraVE && python -m image2caption`