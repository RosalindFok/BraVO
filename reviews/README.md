# A Survey List of Visual Neural Encoding and Decoding via AI

- [A Survey List of Visual Neural Encoding and Decoding via AI](#a-survey-list-of-visual-neural-encoding-and-decoding-via-ai)
  - [1. Principles of Neural Science](#1-principles-of-neural-science)
    - [1.1. Learning Neurology](#11-learning-neurology)
  - [2. Diffusiom Model's Family](#2-diffusiom-models-family)
    - [2.1. Reviews](#21-reviews)
    - [2.2. Articles](#22-articles)
      - [\[2023, Nature Communications\] High-dimensional topographic organization of visual features in the primate temporal lobe](#2023-nature-communications-high-dimensional-topographic-organization-of-visual-features-in-the-primate-temporal-lobe)
  - [3. Neural Encoding and Decoding](#3-neural-encoding-and-decoding)
    - [3.1. Reviews](#31-reviews)
    - [3.2. Articles](#32-articles)
      - [\[2018, IEEE TNNLS\] Reconstructing Perceived Images From Human Brain Activities With Bayesian Deep Multiview Learning](#2018-ieee-tnnls-reconstructing-perceived-images-from-human-brain-activities-with-bayesian-deep-multiview-learning)
      - [\[2022, Cereb Cortex\]Reconstructing rapid natural vision with fMRI-conditional video generative adversarial network](#2022-cereb-cortexreconstructing-rapid-natural-vision-with-fmri-conditional-video-generative-adversarial-network)
      - [\[2023, CVPR\]Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding](#2023-cvprseeing-beyond-the-brain-conditional-diffusion-model-with-sparse-masked-modeling-for-vision-decoding)
  - [4. Datasets and Tools](#4-datasets-and-tools)
    - [4.1. BIDS(Brain Imaging Data Structure)](#41-bidsbrain-imaging-data-structure)
    - [4.2. CBICA Image Processing Portal](#42-cbica-image-processing-portal)
    - [4.3. GLIRT (Groupwise and Longitudinal Image Registration Toolbox)](#43-glirt-groupwise-and-longitudinal-image-registration-toolbox)
    - [4.4. FreeSurfer Download and Install](#44-freesurfer-download-and-install)
  - [5. Tips](#5-tips)

## 1. Principles of Neural Science
### 1.1. Learning Neurology
[website](https://learningneurology.com/)

## 2. Diffusiom Model's Family
### 2.1. Reviews

### 2.2. Articles
#### [2023, Nature Communications] High-dimensional topographic organization of visual features in the primate temporal lobe
**Cite as:**
> Yao, M., Wen, B., Yang, M. et al. High-dimensional topographic organization of visual features in the primate temporal lobe. Nat Commun 14, 5931 (2023). https://doi.org/10.1038/s41467-023-41584-0

<details>
  <summary> ris </summary>
TY  - JOUR
AU  - Yao, Mengna
AU  - Wen, Bincheng
AU  - Yang, Mingpo
AU  - Guo, Jiebin
AU  - Jiang, Haozhou
AU  - Feng, Chao
AU  - Cao, Yilei
AU  - He, Huiguang
AU  - Chang, Le
PY  - 2023
DA  - 2023/09/22
TI  - High-dimensional topographic organization of visual features in the primate temporal lobe
JO  - Nature Communications
SP  - 5931
VL  - 14
IS  - 1
SN  - 2041-1723
UR  - https://doi.org/10.1038/s41467-023-41584-0
DO  - 10.1038/s41467-023-41584-0
ID  - Yao2023
ER  - 
</details>

Inferotemporal Cortex (ITC) 
Demonstrating the existence of a pair of orthogonal gradients that differ in spatial scale and revealing significant differences in the functional organization of high-level visual areas between monkey and human brains.
We set out to tackle these challenges by constructing a high-dimensional object space and compute the selectivity of each brain location within this object space. Inspired by recent advances at the intersection of artificial intelligence and neuroscience18,35,36,37, a 25D object space was constructed using responses of units in a deep neural network to a large database of natural images, and functional MRI experiments were conducted in both monkeys and humans to map out the feature preference of the visual temporal lobe. The resulting preference maps helped us determine the functions of previously uncharted territories and reveal differences in the functional organization of high-level visual areas between monkey and human brains.

## 3. Neural Encoding and Decoding
### 3.1. Reviews

### 3.2. Articles
#### [2018, IEEE TNNLS] Reconstructing Perceived Images From Human Brain Activities With Bayesian Deep Multiview Learning
**Cite as:**
> C. Du, C. Du, L. Huang and H. He, "Reconstructing Perceived Images From Human Brain Activities With Bayesian Deep Multiview Learning," in IEEE Transactions on Neural Networks and Learning Systems, vol. 30, no. 8, pp. 2310-2323, Aug. 2019, doi: 10.1109/TNNLS.2018.2882456.

<details>
  <summary> bibtex </summary> 
@ARTICLE{8574054,
  author={Du, Changde and Du, Changying and Huang, Lijie and He, Huiguang},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Reconstructing Perceived Images From Human Brain Activities With Bayesian Deep Multiview Learning}, 
  year={2019},
  volume={30},
  number={8},
  pages={2310-2323},
  doi={10.1109/TNNLS.2018.2882456}
}
</details>

we propose a novel deep generative multiview model(Variational Auto-Encoders, VAE) for the accurate visual image reconstruction from the human brain activities measured by functional magnetic resonance imaging (fMRI). On the one hand, we adopt a deep neural network architecture for visual image generation, which mimics the stages of human visual processing. On the other hand, we design a sparse Bayesian linear model for fMRI activity generation, which can effectively capture voxel correlations, suppress data noise, and avoid overfitting. Furthermore, we devise an efficient mean-field variational inference method to train the proposed model. The proposed method can accurately reconstruct visual images via Bayesian inference. In particular, we exploit a posterior regularization technique in the Bayesian inference to regularize the model posterior. The quantitative and qualitative evaluations conducted on multiple fMRI data sets demonstrate the proposed method can reconstruct visual images more accurately than the state of the art.

#### [2022, Cereb Cortex]Reconstructing rapid natural vision with fMRI-conditional video generative adversarial network
> Wang C, Yan H, Huang W, Li J, Wang Y, Fan YS, Sheng W, Liu T, Li R, Chen H. Reconstructing rapid natural vision with fMRI-conditional video generative adversarial network. Cereb Cortex. 2022 Oct 8;32(20):4502-4511. doi: 10.1093/cercor/bhab498. PMID: 35078227.

<details>
  <summary> nbib </summary> 
PMID- 35078227
OWN - NLM
STAT- MEDLINE
DCOM- 20221018
LR  - 20221027
IS  - 1460-2199 (Electronic)
IS  - 1047-3211 (Linking)
VI  - 32
IP  - 20
DP  - 2022 Oct 8
TI  - Reconstructing rapid natural vision with fMRI-conditional video generative 
      adversarial network.
PG  - 4502-4511
LID - 10.1093/cercor/bhab498 [doi]
CI  - © The Author(s) 2022. Published by Oxford University Press. All rights reserved. 
      For permissions, please e-mail: journals.permissions@oup.com.
FAU - Wang, Chong
AU  - Wang C
AD  - The Clinical Hospital of Chengdu Brain Science Institute, School of Life Science 
      and Technology, University of Electronic Science and Technology of China, Chengdu 
      610054, China.
AD  - MOE Key Lab for Neuroinformation; High-Field Magnetic Resonance Brain Imaging Key 
      Laboratory of Sichuan Province, University of Electronic Science and Technology 
      of China, Chengdu 610054, China.
FAU - Yan, Hongmei
AU  - Yan H
AD  - The Clinical Hospital of Chengdu Brain Science Institute, School of Life Science 
      and Technology, University of Electronic Science and Technology of China, Chengdu 
      610054, China.
AD  - MOE Key Lab for Neuroinformation; High-Field Magnetic Resonance Brain Imaging Key 
      Laboratory of Sichuan Province, University of Electronic Science and Technology 
      of China, Chengdu 610054, China.
FAU - Huang, Wei
AU  - Huang W
AD  - The Clinical Hospital of Chengdu Brain Science Institute, School of Life Science 
      and Technology, University of Electronic Science and Technology of China, Chengdu 
      610054, China.
FAU - Li, Jiyi
AU  - Li J
AD  - The Clinical Hospital of Chengdu Brain Science Institute, School of Life Science 
      and Technology, University of Electronic Science and Technology of China, Chengdu 
      610054, China.
FAU - Wang, Yuting
AU  - Wang Y
AD  - The Clinical Hospital of Chengdu Brain Science Institute, School of Life Science 
      and Technology, University of Electronic Science and Technology of China, Chengdu 
      610054, China.
FAU - Fan, Yun-Shuang
AU  - Fan YS
AD  - The Clinical Hospital of Chengdu Brain Science Institute, School of Life Science 
      and Technology, University of Electronic Science and Technology of China, Chengdu 
      610054, China.
FAU - Sheng, Wei
AU  - Sheng W
AD  - The Clinical Hospital of Chengdu Brain Science Institute, School of Life Science 
      and Technology, University of Electronic Science and Technology of China, Chengdu 
      610054, China.
FAU - Liu, Tao
AU  - Liu T
AD  - The Clinical Hospital of Chengdu Brain Science Institute, School of Life Science 
      and Technology, University of Electronic Science and Technology of China, Chengdu 
      610054, China.
FAU - Li, Rong
AU  - Li R
AD  - The Clinical Hospital of Chengdu Brain Science Institute, School of Life Science 
      and Technology, University of Electronic Science and Technology of China, Chengdu 
      610054, China.
AD  - MOE Key Lab for Neuroinformation; High-Field Magnetic Resonance Brain Imaging Key 
      Laboratory of Sichuan Province, University of Electronic Science and Technology 
      of China, Chengdu 610054, China.
FAU - Chen, Huafu
AU  - Chen H
AD  - The Clinical Hospital of Chengdu Brain Science Institute, School of Life Science 
      and Technology, University of Electronic Science and Technology of China, Chengdu 
      610054, China.
AD  - MOE Key Lab for Neuroinformation; High-Field Magnetic Resonance Brain Imaging Key 
      Laboratory of Sichuan Province, University of Electronic Science and Technology 
      of China, Chengdu 610054, China.
AD  - The Center of Psychosomatic Medicine, Sichuan Provincial Center for Mental 
      Health, Sichuan Provincial People's Hospital, University of Electronic Science 
      and Technology of China, Chengdu 611731, China.
LA  - eng
PT  - Journal Article
PT  - Research Support, Non-U.S. Gov't
PL  - United States
TA  - Cereb Cortex
JT  - Cerebral cortex (New York, N.Y. : 1991)
JID - 9110718
SB  - IM
MH  - Image Processing, Computer-Assisted/methods
MH  - *Magnetic Resonance Imaging/methods
MH  - *Visual Cortex/diagnostic imaging/physiology
OTO - NOTNLM
OT  - conditional generative adversarial networks
OT  - fMRI
OT  - visual reconstruction
EDAT- 2022/01/26 06:00
MHDA- 2022/10/19 06:00
CRDT- 2022/01/25 20:18
PHST- 2021/07/15 00:00 [received]
PHST- 2021/10/24 00:00 [revised]
PHST- 2021/12/03 00:00 [accepted]
PHST- 2022/01/26 06:00 [pubmed]
PHST- 2022/10/19 06:00 [medline]
PHST- 2022/01/25 20:18 [entrez]
AID - 6515038 [pii]
AID - 10.1093/cercor/bhab498 [doi]
PST - ppublish
SO  - Cereb Cortex. 2022 Oct 8;32(20):4502-4511. doi: 10.1093/cercor/bhab498.
</details>

Here, we developed a novel fMRI-conditional video generative adversarial network (f-CVGAN) to reconstruct rapid video stimuli from evoked fMRI responses. In this model, we employed a generator to produce spatiotemporal reconstructions and employed two separate discriminators (spatial and temporal discriminators) for the assessment. We trained and tested the f-CVGAN on two publicly available video-fMRI datasets, and the model produced pixel-level reconstructions of 8 perceived video frames from each fMRI volume.

#### [2023, CVPR]Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding
**Cite as:**
> Chen Z, Qing J, Xiang T, et al. Seeing beyond the brain: Conditional diffusion model with sparse masked modeling for vision decoding[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 22710-22720.

<details>
  <summary> bibtex </summary> 
@inproceedings{chen2023seeing,
  title={Seeing beyond the brain: Conditional diffusion model with sparse masked modeling for vision decoding},
  author={Chen, Zijiao and Qing, Jiaxin and Xiang, Tiange and Yue, Wan Lin and Zhou, Juan Helen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={22710--22720},
  year={2023}
}
</details>

[Project Link](https://mind-vis.github.io/)

 In this work, we present MinD-Vis: Sparse Masked Brain Modeling with Double-Conditioned Latent Diffusion Model for Human Vision Decoding. Firstly, we learn an effective self-supervised representation of fMRI data using mask modeling in a large latent space inspired by the sparse coding of information in the primary visual cortex. Then by augmenting a latent diffusion model with double-conditioning, we show that MinD-Vis can reconstruct highly plausible images with semantically matching details from brain recordings using very few paired annotations.

## 4. Datasets and Tools
### 4.1. BIDS(Brain Imaging Data Structure)
[BIDS](https://bids.neuroimaging.io/)
[Filenames Link](https://bids-standard.github.io/bids-starter-kit/folders_and_files/files.html)

### 4.2. CBICA Image Processing Portal
[CBICA](https://ipp.cbica.upenn.edu/)
[Cancer Imaging Phenomics Toolkit (CaPTk), currently focusing on brain, breast, and lung cancer.](https://www.med.upenn.edu/cbica/captk/)

### 4.3. GLIRT (Groupwise and Longitudinal Image Registration Toolbox) 
[groupwise registration and longitudinal registration](https://www.nitrc.org/projects/glirt)

### 4.4. FreeSurfer Download and Install
[Latest Version Release](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) 

## 5. Tips
- 尽可能全面的引用别人的论文，努力碰到审稿人的论文
