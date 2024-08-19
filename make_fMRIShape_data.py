import os
import numpy as np

from utils import join_paths
from utils import fMRI_Shape_dir_path, fmrishape_saved_dir_path

class fMRI_Shape_DATA():
    def __init__(self, root_dir_path : str = fMRI_Shape_dir_path, subj_id : int | str = None) -> None:
        self.annotations_dir_path = join_paths(root_dir_path, 'annotations')
        self.camera_pose_dir_path = join_paths(root_dir_path, 'camera_pose')
        self.stimuli_dir_path = join_paths(root_dir_path, 'stimuli')
        self.subj_dir_path = join_paths(root_dir_path, f'sub-{str(subj_id).zfill(4)}')
        for dir_path in [self.annotations_dir_path, self.camera_pose_dir_path, self.stimuli_dir_path, self.subj_dir_path]:
            assert os.path.exists(dir_path), f'{dir_path} does not exist'
