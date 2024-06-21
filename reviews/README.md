# A Survey List of Visual Neural Encoding and Decoding via AI
## Content
- [A Survey List of Visual Neural Encoding and Decoding via AI](#a-survey-list-of-visual-neural-encoding-and-decoding-via-ai)
  - [Content](#content)
  - [1. Principles of Neural Science](#1-principles-of-neural-science)
    - [1.1. Website](#11-website)
      - [Learning Neurology](#learning-neurology)
    - [1.2. Articles](#12-articles)
      - [\[2001, Science\]Distributed and Overlapping Representations of Faces and Objects in Ventral Temporal Cortex](#2001-sciencedistributed-and-overlapping-representations-of-faces-and-objects-in-ventral-temporal-cortex)
      - [\[2008, Front Syst Neurosci\]Representational Similarity Analysis – Connecting the Branches of Systems Neurosci](#2008-front-syst-neuroscirepresentational-similarity-analysis--connecting-the-branches-of-systems-neurosci)
      - [\[2013, jov\]When crowding of crowding leads to uncrowding](#2013-jovwhen-crowding-of-crowding-leads-to-uncrowding)
      - [\[2013, J Neurophysiol\]Compressive spatial summation in human visual cortex](#2013-j-neurophysiolcompressive-spatial-summation-in-human-visual-cortex)
  - [2. Diffusiom Model's Family](#2-diffusiom-models-family)
    - [2.1. Reviews](#21-reviews)
      - [\[2023, ACM Computing Surveys\]Diffusion Models: A Comprehensive Survey of Methods and Applications](#2023-acm-computing-surveysdiffusion-models-a-comprehensive-survey-of-methods-and-applications)
    - [2.2. Articles](#22-articles)
      - [\[2023, Nature Communications\] High-dimensional topographic organization of visual features in the primate temporal lobe](#2023-nature-communications-high-dimensional-topographic-organization-of-visual-features-in-the-primate-temporal-lobe)
  - [3. Neural Encoding and Decoding](#3-neural-encoding-and-decoding)
    - [3.1. Reviews](#31-reviews)
      - [\[2022, Machine Intelligence Research\]Neural Decoding of Visual Information Across Different Neural Recording Modalities and Approaches](#2022-machine-intelligence-researchneural-decoding-of-visual-information-across-different-neural-recording-modalities-and-approaches)
    - [3.2. Articles](#32-articles)
      - [\[2018, IEEE TNNLS\] Reconstructing Perceived Images From Human Brain Activities With Bayesian Deep Multiview Learning](#2018-ieee-tnnls-reconstructing-perceived-images-from-human-brain-activities-with-bayesian-deep-multiview-learning)
      - [\[2022, Cereb Cortex\]Reconstructing rapid natural vision with fMRI-conditional video generative adversarial network](#2022-cereb-cortexreconstructing-rapid-natural-vision-with-fmri-conditional-video-generative-adversarial-network)
      - [\[2023, CVPR\] High-resolution image reconstruction with latent diffusion models from human brain activity](#2023-cvpr-high-resolution-image-reconstruction-with-latent-diffusion-models-from-human-brain-activity)
      - [\[2023, CVPR\]Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding](#2023-cvprseeing-beyond-the-brain-conditional-diffusion-model-with-sparse-masked-modeling-for-vision-decoding)
      - [\[2023, Scientific Reports\] Natural scene reconstruction from fMRI signals using generative latent diffusion](#2023-scientific-reports-natural-scene-reconstruction-from-fmri-signals-using-generative-latent-diffusion)
      - [\[2023, NeurIPS(spotlight)\]Reconstructing the Mind’s Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors](#2023-neuripsspotlightreconstructing-the-minds-eye-fmri-to-image-with-contrastive-learning-and-diffusion-priors)
      - [\[2023, NeurIPS(oral)\]Brain Diffusion for Visual Exploration: Cortical Discovery using Large Scale Generative Models](#2023-neuripsoralbrain-diffusion-for-visual-exploration-cortical-discovery-using-large-scale-generative-models)
      - [\[2023, ACMMM\]MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion](#2023-acmmmminddiffuser-controlled-image-reconstruction-from-human-brain-activity-with-semantic-and-structural-diffusion)
  - [4. Datasets and Tools](#4-datasets-and-tools)
    - [4.1. BIDS(Brain Imaging Data Structure)](#41-bidsbrain-imaging-data-structure)
    - [4.2. CBICA Image Processing Portal](#42-cbica-image-processing-portal)
    - [4.3. GLIRT (Groupwise and Longitudinal Image Registration Toolbox)](#43-glirt-groupwise-and-longitudinal-image-registration-toolbox)
    - [4.4. FreeSurfer Download and Install](#44-freesurfer-download-and-install)
  - [5. Tips](#5-tips)

## 1. Principles of Neural Science
### 1.1. Website
#### Learning Neurology
[website](https://learningneurology.com/)
### 1.2. Articles
#### [2001, Science]Distributed and Overlapping Representations of Faces and Objects in Ventral Temporal Cortex
**Cite as:**
> James V. Haxby et al. ,Distributed and Overlapping Representations of Faces and Objects in Ventral Temporal Cortex.Science293,2425-2430(2001).DOI:10.1126/science.1063736

#### [2008, Front Syst Neurosci]Representational Similarity Analysis – Connecting the Branches of Systems Neurosci
**Cite as:**
> Kriegeskorte N, Mur M, Bandettini P. Representational similarity analysis - connecting the branches of systems neuroscience. Front Syst Neurosci. 2008 Nov 24;2:4. doi: 10.3389/neuro.06.004.2008. PMID: 19104670; PMCID: PMC2605405. 

在特征空间上找到哪个脑区对图像的哪个特征反映。

#### [2013, jov]When crowding of crowding leads to uncrowding
**Cite as:**
> Mauro Manassi, Bilge Sayim, Michael H. Herzog; When crowding of crowding leads to uncrowding. Journal of Vision 2013;13(13):10. https://doi.org/10.1167/13.13.10.

从V1$\rightarrow$V2$\rightarrow$V4$\rightarrow$PIT$\rightarrow$IT 信息的逐层处理过程中，对应的神经元的感受野越来越大

每层之间感受野增大的系数大体为2.5

高级别的神经元将信息集成在具有较小感受野的多个低级神经元上，编码更复杂的特征

#### [2013, J Neurophysiol]Compressive spatial summation in human visual cortex
**Cite as:**
> Kay KN, Winawer J, Mezer A, Wandell BA. Compressive spatial summation in human visual cortex. J Neurophysiol. 2013 Jul;110(2):481-94. doi: 10.1152/jn.00105.2013. Epub 2013 Apr 24. PMID: 23615546; PMCID: PMC3727075.

V1区是编码边缘和线条等基本特征

V2区神经元对错觉轮廓有反应，是色调敏感区

V3区是信息过渡区

V4是颜色感知的主要区域，参与曲率计算、运动方向选择和背景分离

IT区是物体表达和识别区

## 2. Diffusiom Model's Family
### 2.1. Reviews
#### [2023, ACM Computing Surveys]Diffusion Models: A Comprehensive Survey of Methods and Applications
**Cite as:**
> Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. 2023. Diffusion Models: A Comprehensive Survey of Methods and Applications. ACM Comput. Surv. 56, 4, Article 105 (April 2024), 39 pages. https://doi.org/10.1145/3626235

### 2.2. Articles
#### [2023, Nature Communications] High-dimensional topographic organization of visual features in the primate temporal lobe
**Cite as:**
> Yao, M., Wen, B., Yang, M. et al. High-dimensional topographic organization of visual features in the primate temporal lobe. Nat Commun 14, 5931 (2023). https://doi.org/10.1038/s41467-023-41584-0

Inferotemporal Cortex (ITC) 
Demonstrating the existence of a pair of orthogonal gradients that differ in spatial scale and revealing significant differences in the functional organization of high-level visual areas between monkey and human brains.
We set out to tackle these challenges by constructing a high-dimensional object space and compute the selectivity of each brain location within this object space. Inspired by recent advances at the intersection of artificial intelligence and neuroscience18,35,36,37, a 25D object space was constructed using responses of units in a deep neural network to a large database of natural images, and functional MRI experiments were conducted in both monkeys and humans to map out the feature preference of the visual temporal lobe. The resulting preference maps helped us determine the functions of previously uncharted territories and reveal differences in the functional organization of high-level visual areas between monkey and human brains.

## 3. Neural Encoding and Decoding
### 3.1. Reviews
#### [2022, Machine Intelligence Research]Neural Decoding of Visual Information Across Different Neural Recording Modalities and Approaches
**Cite as:**
> Zhang, YJ., Yu, ZF., Liu, J.K. et al. Neural Decoding of Visual Information Across Different Neural Recording Modalities and Approaches. Mach. Intell. Res. 19, 350–365 (2022). https://doi.org/10.1007/s11633-022-1335-2

### 3.2. Articles
#### [2018, IEEE TNNLS] Reconstructing Perceived Images From Human Brain Activities With Bayesian Deep Multiview Learning
**Cite as:**
> C. Du, C. Du, L. Huang and H. He, "Reconstructing Perceived Images From Human Brain Activities With Bayesian Deep Multiview Learning," in IEEE Transactions on Neural Networks and Learning Systems, vol. 30, no. 8, pp. 2310-2323, Aug. 2019, doi: 10.1109/TNNLS.2018.2882456.

we propose a novel deep generative multiview model(Variational Auto-Encoders, VAE) for the accurate visual image reconstruction from the human brain activities measured by functional magnetic resonance imaging (fMRI). On the one hand, we adopt a deep neural network architecture for visual image generation, which mimics the stages of human visual processing. On the other hand, we design a sparse Bayesian linear model for fMRI activity generation, which can effectively capture voxel correlations, suppress data noise, and avoid overfitting. Furthermore, we devise an efficient mean-field variational inference method to train the proposed model. The proposed method can accurately reconstruct visual images via Bayesian inference. In particular, we exploit a posterior regularization technique in the Bayesian inference to regularize the model posterior. The quantitative and qualitative evaluations conducted on multiple fMRI data sets demonstrate the proposed method can reconstruct visual images more accurately than the state of the art.

#### [2022, Cereb Cortex]Reconstructing rapid natural vision with fMRI-conditional video generative adversarial network
> Wang C, Yan H, Huang W, Li J, Wang Y, Fan YS, Sheng W, Liu T, Li R, Chen H. Reconstructing rapid natural vision with fMRI-conditional video generative adversarial network. Cereb Cortex. 2022 Oct 8;32(20):4502-4511. doi: 10.1093/cercor/bhab498. PMID: 35078227.

Here, we developed a novel fMRI-conditional video generative adversarial network (f-CVGAN) to reconstruct rapid video stimuli from evoked fMRI responses. In this model, we employed a generator to produce spatiotemporal reconstructions and employed two separate discriminators (spatial and temporal discriminators) for the assessment. We trained and tested the f-CVGAN on two publicly available video-fMRI datasets, and the model produced pixel-level reconstructions of 8 perceived video frames from each fMRI volume.

#### [2023, CVPR] High-resolution image reconstruction with latent diffusion models from human brain activity
**Cite as:**
> Y. Takagi and S. Nishimoto, "High-resolution image reconstruction with latent diffusion models from human brain activity," 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, BC, Canada, 2023, pp. 14453-14463, doi: 10.1109/CVPR52729.2023.01389.

#### [2023, CVPR]Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding
**Cite as:**
> Chen Z, Qing J, Xiang T, et al. Seeing beyond the brain: Conditional diffusion model with sparse masked modeling for vision decoding[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 22710-22720.

[Project Link](https://mind-vis.github.io/)

 In this work, we present MinD-Vis: Sparse Masked Brain Modeling with Double-Conditioned Latent Diffusion Model for Human Vision Decoding. Firstly, we learn an effective self-supervised representation of fMRI data using mask modeling in a large latent space inspired by the sparse coding of information in the primary visual cortex. Then by augmenting a latent diffusion model with double-conditioning, we show that MinD-Vis can reconstruct highly plausible images with semantically matching details from brain recordings using very few paired annotations.

#### [2023, Scientific Reports] Natural scene reconstruction from fMRI signals using generative latent diffusion
**Cite as:**
> Ozcelik, F., VanRullen, R. Natural scene reconstruction from fMRI signals using generative latent diffusion. Sci Rep 13, 15666 (2023). https://doi.org/10.1038/s41598-023-42891-8

#### [2023, NeurIPS(spotlight)]Reconstructing the Mind’s Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors
**Cite as:**
> Scotti P, Banerjee A, Goode J, et al. Reconstructing the mind's eye: fMRI-to-image with contrastive learning and diffusion priors[J]. Advances in Neural Information Processing Systems, 2024, 36.

#### [2023, NeurIPS(oral)]Brain Diffusion for Visual Exploration: Cortical Discovery using Large Scale Generative Models
**Cite as:**
> Luo A, Henderson M, Wehbe L, et al. Brain diffusion for visual exploration: Cortical discovery using large scale generative models[J]. Advances in Neural Information Processing Systems, 2024, 36.

#### [2023, ACMMM]MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion
**Cite as:**
> Lu Y, Du C, Zhou Q, et al. MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion[C]//Proceedings of the 31st ACM International Conference on Multimedia. 2023: 5899-5908.



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
