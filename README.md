# BraVE
Brain Vision Explainability
使用基于深度学习的方法，对人类的视觉皮层进行功能或者结构的建模


## Module 1: image2caption via CLIP and BLIP
### Setup Environment
```shell
conda create --name BandCLIP python=3.11
source activate BandCLIP

git clone https://github.com/RosalindFok/BraVE.git
cd ./BraVE

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install regex -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install fairscale -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install timm -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple/
cd image2caption/CLIP && python setup.py install && cd ../../
```

### Run 
Run this module in your platform: `python -m image2caption`
Run this module in BsccCloud: 
``` shell
module load anaconda/2021.11 
module load cuda/11.8
chmod 777 run.sh
dsub -s run.sh # submit 
djob           # check id
djob -T ID     # cancel
```