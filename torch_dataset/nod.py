# -*- coding: utf-8 -*-


""" 
    NOD: Natural Object Dataset
    Cite Info: Gong, Z., Zhou, M., Dai, Y. et al. A large-scale fMRI dataset for the visual processing of naturalistic scenes. Sci Data 10, 559 (2023). https://doi.org/10.1038/s41597-023-02471-x
"""

import os
import h5py
from torch.utils.data import Dataset
from torch_dataset import nod_path, hdf5_dir

# HDF5文件
nod_hdf5_path = os.path.join(hdf5_dir, '_nod.hdf5')

# nod下的文件夹或文件
derivatives_path = [os.path.join(nod_path, x) for x in os.listdir(nod_path) if 'derivatives' in x and os.path.isdir(os.path.join(nod_path, x))][0]
stimuli_path = [os.path.join(nod_path, x) for x in os.listdir(nod_path) if 'stimuli' in x and os.path.isdir(os.path.join(nod_path, x))][0]
subj_path_list = [os.path.join(nod_path, x) for x in os.listdir(nod_path) if 'sub-' in x and os.path.isdir(os.path.join(nod_path, x))]
participants_tsv_path = [os.path.join(nod_path, x) for x in os.listdir(nod_path) if 'participants.tsv' in x and os.path.isfile(os.path.join(nod_path, x))][0]

# derivatives下的文件夹
ciftify_path = [os.path.join(derivatives_path, x) for x in os.listdir(derivatives_path) if 'ciftify' in x][0]
fmriprep_path = [os.path.join(derivatives_path, x) for x in os.listdir(derivatives_path) if 'fmriprep' in x][0]

# stimuli下的文件夹
coco_path = [os.path.join(stimuli_path, x) for x in os.listdir(stimuli_path) if 'coco' in x][0]
imagenet_path = [os.path.join(stimuli_path, x) for x in os.listdir(stimuli_path) if 'imagenet' in x][0]

class nod(Dataset):
    '''
    单个受试者 每个受试者训练一个模型
    '''
    def __init__(self, subject_id : int) -> None:
        super().__init__()
        # 被选中的受试者编号
        self.subject_id = subject_id
        # 该受试者的HDF5文件路径
        self.hdf5_path = nod_hdf5_path.replace('_', ''.join([str(self.subject_id), '_']))
        # 该受试者的anat

        # 该受试者的coco

        # 该受试者的imagenet



    def __getitem__(self, index) -> None:
        pass
    def __len__(self) -> int:
        return 0
    
