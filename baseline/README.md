# Baseline

## Platform
Beijing Super Cloud Computing Center - N32EA14P: NVIDIA A100-PCIE-40GB*8

``` shell
module load anaconda/2021.11 cuda/11.8
conda create --name YOUR_NAME python=3.11
source activate YOUR_NAME
conda env remove -n YOUR_NAME

dsub -s run.sh # submit the job
djob # check id of the job
djob -T job_id # cancel the job via its id
```

## High-resolution image reconstruction with latent diffusion models from human brain activity
[code](https://github.com/yu-takagi/StableDiffusionReconstruction)
###### Prepare the environment:
``` shell
conda create --name hrirLDM python=3.11
source activate hrirLDM

# https://download.pytorch.org/whl/cu118/torch-2.3.0%2Bcu118-cp311-cp311-linux_x86_64.whl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install nibabel -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install pycocotools -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install IPython -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install himalaya -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install timm -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install fairscale -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install omegaconf -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install pytorch_lightning -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install ldm -i https://pypi.tuna.tsinghua.edu.cn/simple/ # you may need to change the type of print in its source code to avoid the SyntaxError
conda install dlib 
卡在这里了
```

###### Prepare the pre-trained models:
download `sd-v1-4.ckpt` form `https://hf-mirror.com/CompVis/stable-diffusion-v-1-4-original/tree/main` and then place it under `StableDiffusionReconstruction/codes/diffusion_sd1/stable-diffusion/models/ldm/stable-diffusion-v1`, or cd into this folder and run `sh download_ckpt.sh` and rename it. <br>
download `512-depth-ema.ckpt` from `https://hf-mirror.com/stabilityai/stable-diffusion-2-depth/tree/main` and the place it under `StableDiffusionReconstruction/codes/diffusion_sd2/stablediffusion/models/`, or cd into this folder and run `sh download_ckpt.sh` and rename it.

###### Prepare the data:
copy `nsddata`, `nsddata_betas`, and `nsddata_stimuli` into `StableDiffusionReconstruction/nsd` <br>
``` shell
cd codes/utils/
python make_subjmri.py --subject subj01
```
###### Reconstruction based on the method mentioned in the paper:
``` shell
cd codes/utils/
chmod 777 run.sh
dsub -s run.sh

python make_subjstim.py --featname init_latent --use_stim each --subject subj01
python make_subjstim.py --featname init_latent --use_stim ave --subject subj01
python make_subjstim.py --featname c --use_stim each --subject subj01
python make_subjstim.py --featname c --use_stim ave --subject subj01
python ridge.py --target c --roi ventral --subject subj01
python ridge.py --target init_latent --roi early --subject subj01

cd codes/diffusion_sd1/
python diffusion_decoding.py --imgidx 0 10 --gpu 1 --subject subj01 --method cvpr
```

###### Delete the conda environment:
``` shell
conda env remove -n hrirLDM
```


## MindDiffuser
[code](https://github.com/ReedOnePeck/MindDiffuser)
###### Prepare the environment:
``` shell
# conda create --name MindDiffuser python=3.11
# source activate MindDiffuser

# https://download.pytorch.org/whl/cu118/torch-2.3.0%2Bcu118-cp311-cp311-linux_x86_64.whl
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

cd ./BraVO/baseline/MindDiffuser/
conda env create -f environment_1.yaml  
conda activate MindDiffuser          
pip install -r pip_install.txt
```

conda env remove -n MindDiffuser

## brain-diffuser
[code](https://github.com/ozcelikfu/brain-diffuser)
###### Prepare the environment:
``` shell
conda env create -f environment.yml
source activate brain-diffuser
```
