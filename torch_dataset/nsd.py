# -*- coding: utf-8 -*-

""" 
    NSD: Natural Scenes Dataset
    Cite Info: Allen, E.J., St-Yves, G., Wu, Y. et al. A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nat Neurosci 25, 116–126 (2022). https://doi.org/10.1038/s41593-021-00962-x
"""

import re
import os
import h5py
from torch.utils.data import Dataset
from torch_dataset import nsd_path, hdf5_dir

# nsd 下文件夹有 nsddata, nsddata_betas, nsddata_stimuli, nsddata_timeseries, nsddata_other, nsddata_diffusion, nsddata_rawdata
nsddata_path = os.path.join(nsd_path, 'nsddata_path')
nsddata_betas_path = os.path.join(nsd_path, 'nsddata_betas')
nsddata_stimuli_path = os.path.join(nsd_path, 'nsddata_stimuli')
nsddata_timeseries_path = os.path.join(nsd_path, 'nsddata_timeseries')
nsddata_other_path = os.path.join(nsd_path, 'nsddata_other')
nsddata_diffusion_path = os.path.join(nsd_path, 'nsddata_diffusion')
nsddata_rawdata_path = os.path.join(nsd_path, 'nsddata_rawdata')

'''cvpr2023使用了 nsddata, nsddata_betas, nsddata_stimuli'''
# nsddata 下文件夹有 bdata, experiments, freesurfer, information, inspections, ppdata, stimuli, templates
nsd_bdata_path = os.path.join(nsddata_path, 'bdata')
nsd_experiments_path = os.path.join(nsddata_path, 'experiments')
nsd_freesurfer_path = os.path.join(nsddata_path, 'freesurfer')
nsd_information_path = os.path.join(nsddata_path, 'information')
nsd_inspections_path = os.path.join(nsddata_path, 'inspections')
nsd_ppdata_path = os.path.join(nsddata_path, 'ppdata')
nsd_stimuli_path = os.path.join(nsddata_path, 'stimuli')
nsd_templates_path = os.path.join(nsddata_path, 'templates')

# nsddata_stimuli 下文件夹有 stimuli, stimuli 下文件夹有 nsd, nsdimagery, nsdsynthetic
nsddata_stimuli_path = os.path.join(nsddata_stimuli_path, 'stimuli')
nsddata_stimuli_nsd_path = os.path.join(nsddata_stimuli_path, 'nsd')
nsddata_stimuli_nsdimagery_path = os.path.join(nsddata_stimuli_path, 'nsdimagery')
nsddata_stimuli_nsdsynthetic_path = os.path.join(nsddata_stimuli_path, 'nsdsynthetic')


# nsddata_betas 下文件夹有 ppdata, ppdata 下文件夹有 subj01,subj02,subj03,subj04,subj05,subj06,subj07,subj08
nsddata_betas_path = os.path.join(nsddata_betas_path, 'ppdata')
nsddata_betas_subs_dir_path_list = [os.path.join(nsddata_betas_path, x) for x in os.listdir(nsddata_betas_path)]
nsddata_betas_subs_dir_path_dict = {int(re.search(f'subj0(\d)', path).group(1)) : path for path in nsddata_betas_subs_dir_path_list}

class nsd(Dataset):
    def __init__(self, subject_id : int) -> None:
        super().__init__()
        # 当前选择的受试者id
        self.subject_id = subject_id

    def __getitem__(self, index) -> None:
        pass
    def __len__(self) -> int:
        return 0