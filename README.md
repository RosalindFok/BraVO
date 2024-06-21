# BraVO: Brain Visual Observer
Utilizing deep learning-based methods for functional or structural modeling of the human visual cortex.

Decoding(解码): Engineering issues, neural activity $\rightarrow$ visual stimulation 

Encoding(编码): Scientific issues, visual stimulation $\rightarrow$ neural activity


## Module 1: image2caption via CLIP and BLIP
### Setup Environment
```shell
module load anaconda/2021.11 cuda/11.8
conda create --name BandCLIP python=3.11
source activate BandCLIP

git clone https://github.com/RosalindFok/BraVE.git
cd ./BraVE

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple/

```

### Run 
Run this module in your platform: `python -m image2caption.img2cap` </br>
Run this module in BsccCloud: 
``` shell
chmod 777 run.sh
dsub -s run.sh # submit 
djob           # check id
djob -T ID     # cancel
```
