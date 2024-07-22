# BraVO: Brain Visual Observer
Utilizing deep learning-based methods for functional or structural modeling of the human visual cortex.

Decoding(解码): Engineering issues, neural activity $\rightarrow$ visual stimulation 

Encoding(编码): Scientific issues, visual stimulation $\rightarrow$ neural activity

## Platform
Beijing Super Cloud Computing Center - N32EA14P: `NVIDIA A100-PCIE-40GB*8`
``` shell
chmod 777 run.sh
sbatch --gpus=num_gpus -p gpu run.sh # submit the job
parajobs # check id of the job
scancel job_id # cancel the job via its id
```

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
└── large_files_for_BraVO 
```

## Prepare the Enviorment:
``` shell
module load anaconda/2021.11 cuda/12.1
conda create --name BraVO python=3.11
source activate BraVO

# Download torch torchvision torchaudio
# Note: The lastest version was installed for each package, the version was shown after '#' in each line.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # Successfully installed MarkupSafe-2.1.5 filelock-3.13.1 fsspec-2024.2.0 jinja2-3.1.3 mpmath-1.3.0 networkx-3.2.1 numpy-1.26.3 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.20.5 nvidia-nvjitlink-cu12-12.1.105 nvidia-nvtx-cu12-12.1.105 pillow-10.2.0 sympy-1.12 torch-2.3.1+cu121 torchaudio-2.3.1+cu121 torchvision-0.18.1+cu121 triton-2.3.1 typing-extensions-4.9.0
pip install nibabel -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed nibabel-5.2.1 packaging-24.1
pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed tqdm-4.66.4
pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed h5py-3.11.0
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed pandas-2.2.2 python-dateutil-2.9.0.post0 pytz-2024.1 six-1.16.0 tzdata-2024.1
pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed scipy-1.14.0
pip install omegaconf -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed PyYAML-6.0.1 antlr4-python3-runtime-4.9.3 omegaconf-2.3.0
pip install diffusers -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed certifi-2024.7.4 charset-normalizer-3.3.2 diffusers-0.29.2 huggingface-hub-0.23.4 idna-3.7 importlib-metadata-8.0.0 regex-2024.5.15 requests-2.32.3 safetensors-0.4.3 urllib3-2.2.2 zipp-3.19.2
pip install timm -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed timm-1.0.7
pip install iopath -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed iopath-0.1.10 portalocker-2.10.1
pip install decord -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed decord-0.6.0
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed opencv-python-4.10.0.84
pip install webdataset -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed braceexpand-0.1.7 webdataset-0.2.86
# pip install transformers==4.33.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/ # 
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed tokenizers-0.19.1 transformers-4.42.4
pip install fairscale -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed fairscale-0.4.13
pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed einops-0.8.0
pip install spacy -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed annotated-types-0.7.0 blis-0.7.11 catalogue-2.0.10 click-8.1.7 cloudpathlib-0.18.1 confection-0.1.5 cymem-2.0.8 langcodes-3.4.0 language-data-1.2.0 marisa-trie-1.2.0 markdown-it-py-3.0.0 mdurl-0.1.2 murmurhash-1.0.10 preshed-3.0.9 pydantic-2.8.2 pydantic-core-2.20.1 pygments-2.18.0 rich-13.7.1 shellingham-1.5.4 smart-open-7.0.4 spacy-3.7.5 spacy-legacy-3.0.12 spacy-loggers-1.0.5 srsly-2.4.8 thinc-8.2.5 typer-0.12.3 wasabi-1.1.3 weasel-0.4.1 wrapt-1.16.0
pip install pycocoevalcap -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed contourpy-1.2.1 cycler-0.12.1 fonttools-4.53.1 kiwisolver-1.4.5 matplotlib-3.9.1 pycocoevalcap-1.2 pycocotools-2.0.8 pyparsing-3.1.2
pip install moviepy -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed decorator-4.4.2 imageio-2.34.2 imageio-ffmpeg-0.5.1 moviepy-1.0.3 proglog-0.1.10
pip install peft -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed accelerate-0.32.1 peft-0.11.1 psutil-6.0.0
pip install easydict -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed easydict-1.13
pip install nltk -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed joblib-1.4.2 nltk-3.8.1
```

## Prepare the Data:
1. bert-base-uncased
Download files from `https://hf-mirror.com/google-bert/bert-base-uncased/tree/main` and put them under `../large_files_for_BraVO/bert-base-uncased` <br>

2. stable-diffusion-v1-5
Download files from `https://hf-mirror.com/runwayml/stable-diffusion-v1-5/tree/main` and put them under `../large_files_for_BraVO/stable-diffusion-v1-5` <br>

3. blip-diffusion
Download tar.gz from `https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP-Diffusion/blip-diffusion.tar.gz`, untar it and put the whole folder under `../large_files_for_BraVO/blip-diffusion`  <br>

4. [The Natural Scenes Dataset(NSD)](https://naturalscenesdataset.org/)
> Allen, E.J., St-Yves, G., Wu, Y., Breedlove, J.L., Prince, J.S., Dowdle, L.T., Nau, M., Caron, B., Pestilli, F., Charest, I., Hutchinson, J.B., Naselaris, T.*, Kay, K.* A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nature Neuroscience (2021).
**Note:**
- Full data: subj01, subj02, subj05, subj07
- Download annotation of COCO from [link](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), unzip it and then place the whole folder `annotations` under `dataset/NSD/nsddata_stimuli/stimuli/nsd`
- [NSD Data Manual](https://cvnlab.slite.page/p/CT9Fwl4_hc/NSD-Data-Manual)

1. **Step 1:**
run `sbatch --gpus=2 -p gpu step1_run.sh`  for subj 01, 02, 05, 07

**Step 2:**
run `sbatch --gpus=1 -p gpu step2_run.sh`

## Delete the Enviorment:
``` shell
conda deactivate
conda env remove -n BraVO
```