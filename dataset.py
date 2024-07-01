import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from utils import BraVO_saved_dir_path
from utils import join_paths, read_nii_file

class NSD_Dataset(Dataset):
    def __init__(self, root_dir : str = BraVO_saved_dir_path, subj_id : int = None, mode : str = None) -> None:
        super().__init__()
        assert 1<= subj_id <= 8, print(f'Invalid subj_id={subj_id}. Please choose from 1 to 8.')
        assert mode in ['train', 'test'], print(f'Invalid mode={mode}. Please choose from "train" or "test".')
        self.dir_path = os.path.join(root_dir, f'subj{str(subj_id).zfill(2)}_pairs', mode)
        self.trials_path_list = [join_paths(self.dir_path, x) for x in os.listdir(self.dir_path)]
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        trial_path = self.trials_path_list[index]
        fmri_path = join_paths(trial_path, 'fmri.nii.gz')
        image_path = join_paths(trial_path, 'image.png')
        info_path = join_paths(trial_path, 'info.json')
        assert os.path.exists(info_path)
        fmri_header, fmri_data = read_nii_file(fmri_path)
        image_data = cv2.imread(image_path)
        fmri_data = fmri_data.astype(np.int16)
        return fmri_data, image_data, info_path
    
    def __len__(self) -> int:
        return  len(self.trials_path_list)
