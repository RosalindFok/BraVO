# BraVO: Brain Visual Observer
Utilizing deep learning-based methods for functional or structural modeling of the human visual cortex.

Decoding(解码): Engineering issues, neural activity $\rightarrow$ visual stimulation 

Encoding(编码): Scientific issues, visual stimulation $\rightarrow$ neural activity

## Platform
Beijing Super Cloud Computing Center - N32EA14P: `NVIDIA A100-PCIE-40GB*8`
``` shell
dsub -s run.sh # submit the job
djob # check id of the job
djob -T job_id # cancel the job via its id
```

## Prepare the Enviorment:
``` shell
module load anaconda/2021.11 cuda/11.8
conda create --name BraVO python=3.11
source activate BraVO

# Note: The lastest version was installed for each package, the version was shown after '#' in each line.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Successfully installed MarkupSafe-2.1.5 filelock-3.13.1 fsspec-2024.2.0 jinja2-3.1.3 mpmath-1.3.0 networkx-3.2.1 nvidia-cublas-cu11-11.11.3.6 nvidia-cuda-cupti-cu11-11.8.87 nvidia-cuda-nvrtc-cu11-11.8.89 nvidia-cuda-runtime-cu11-11.8.89 nvidia-cudnn-cu11-8.7.0.84 nvidia-cufft-cu11-10.9.0.58 nvidia-curand-cu11-10.3.0.86 nvidia-cusolver-cu11-11.4.1.48 nvidia-cusparse-cu11-11.7.5.86 nvidia-nccl-cu11-2.20.5 nvidia-nvtx-cu11-11.8.86 pillow-10.2.0 sympy-1.12 torch-2.3.1+cu118 torchaudio-2.3.1+cu118 torchvision-0.18.1+cu118 triton-2.3.1 typing-extensions-4.9.0
pip install nibabel -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed nibabel-5.2.1 numpy-2.0.0 packaging-24.1
pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed tqdm-4.66.4
pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed h5py-3.11.0
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed pandas-2.2.2 python-dateutil-2.9.0.post0 pytz-2024.1 six-1.16.0 tzdata-2024.1
pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed scipy-1.14.0
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed opencv-python-4.10.0.84
```

## Prepare the Data:
[The Natural Scenes Dataset(NSD)](https://naturalscenesdataset.org/)
> Allen, E.J., St-Yves, G., Wu, Y., Breedlove, J.L., Prince, J.S., Dowdle, L.T., Nau, M., Caron, B., Pestilli, F., Charest, I., Hutchinson, J.B., Naselaris, T.*, Kay, K.* A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nature Neuroscience (2021).
**Note:**
- Full data: subj01, subj02, subj05, subj07
- Download annotation of COCO from [link](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), unzip it and then place the whole folder `annotations` under `dataset/NSD/nsddata_stimuli/stimuli/nsd`
总结下CVPR2023、BrainDiVE和那个什么2用了哪些数据，怎么用的
- [NSD Data Manual](https://cvnlab.slite.page/p/CT9Fwl4_hc/NSD-Data-Manual)

**File path | Description**
``` shell
..
├── dataset
│   └── NSD
│       ├── nsddata
│       ├── nsdata_betas
│       ├── nsddata_diffusion
│       ├── nsddata_other
│       ├── nsddata_rawdata
│       ├── nsddata_stimuli
│       └── nsddata_timeseries
├── BraVO
│   ├── dirs ...
│   └── files ...
└── YOUR_OWN_FILE
```

## Delete the Enviorment:
``` shell
conda env remove -n BraVO
```



## Module 1: image2caption via CLIP and BLIP
### Setup Environment
```shell
module load anaconda/2021.11 cuda/11.8
conda create --name BandCLIP python=3.11
source activate BandCLIP

git clone https://github.com/RosalindFok/BraVE.git
cd ./BraVE

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install nibabel -i https://pypi.tuna.tsinghua.edu.cn/simple/
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
