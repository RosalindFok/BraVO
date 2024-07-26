import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils import BraVO_saved_dir_path
from utils import join_paths, read_nii_file

__all__ = ['NSD_Dataset',
           'make_paths_dict', 'make_rois_dict']


def make_paths_dict(subj_id : int) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, list[str]]]]:
    """  
    Constructs and returns dictionaries containing paths to training and testing trial data and ROIs for a given subject.  

    Args:  
        subj_id (int): The ID of the subject for which the paths are to be generated.  

    Returns:  
        tuple: A tuple containing three dictionaries:  
            - train_trial_path_dict (dict[str, dict[str, str]]): Paths for training trials.  
            - test_trial_path_dict (dict[str, dict[str, str]]): Paths for testing trials.  
            - rois_path_dict (dict[str, dict[str, list[str]]]): Paths for ROIs (Regions of Interest), organized by type and name.  
    """ 
    sujb_path = join_paths(BraVO_saved_dir_path, f'subj{str(subj_id).zfill(2)}_pairs')
    rois_path = join_paths(sujb_path, 'ROIs')
    assert os.path.exists(rois_path), print(f'dir_path={rois_path} does not exist.')

    # Train or Test set
    def __make_train_or_test_path_dict__(mode : str) -> dict[str, dict[str, str]]:
        """  
        Constructs a dictionary of paths for either training or testing trials.  

        Args:  
            mode (str): Mode specifying whether to construct paths for 'train' or 'test'.  

        Returns:  
            dict[str, dict[str, str]]: A dictionary where keys are trial indices and  
                                       values are dictionaries containing paths to  
                                       'fmri', 'image', 'embedding', and 'strings' files.  
        """   
        dir_path = join_paths(sujb_path, mode)
        assert os.path.exists(dir_path), print(f'dir_path={dir_path} does not exist.')
        trial_paths_list = [join_paths(dir_path, x) for x in os.listdir(dir_path)]
        trial_path_dict = {}
        for index, trail_path in enumerate(trial_paths_list):
            trial_path_dict[index] = {
                'fmri' : join_paths(trail_path, 'fmri.nii.gz'),
                'image' : join_paths(trail_path, 'image.png'),
                'embedding' : join_paths(trail_path, 'embedding.npy'),
                'strings' : join_paths(trail_path,'strings.json')
            }
        return trial_path_dict
        
    train_trial_path_dict = __make_train_or_test_path_dict__(mode = 'train')
    test_trial_path_dict  = __make_train_or_test_path_dict__(mode = 'test')

    # ROIs
    rois_path_dict = {} # {key=surface or volume, value=dict{key=roi_name, value=list[roi_path]}}
    for roi_type in os.listdir(rois_path):
        roi_type_path = join_paths(rois_path, roi_type)
        rois_path_dict[roi_type] = {}
        for roi_name in os.listdir(roi_type_path):
            roi_name_path = join_paths(roi_type_path, roi_name)
            rois_path_dict[roi_type][roi_name] = [join_paths(roi_name_path, x) for x in os.listdir(roi_name_path)]

    return train_trial_path_dict, test_trial_path_dict, rois_path_dict


class NSD_Dataset(Dataset):
    """
    load proprocessed data
    """
    def __init__(self, trial_path_dict : dict[str, dict[str, str]], mask_path_list : list[str], threshold : int) -> None:
        super().__init__()
        self.trial_path_dict = trial_path_dict
        self.lh_mask_header, self.lh_mask_data = read_nii_file([x for x in mask_path_list if 'lh.' in x and '.nii.gz' in x][0])
        self.rh_mask_header, self.rh_mask_data = read_nii_file([x for x in mask_path_list if 'rh.' in x and '.nii.gz' in x][0])
        self.mask_header, self.mask_data = read_nii_file([x for x in mask_path_list if '.nii.gz' in x and not 'lh.' in x and not 'rh.' in x][0])
        assert self.lh_mask_data.shape == self.rh_mask_data.shape == self.mask_data.shape, print(f'lh_mask_data.shape={self.lh_mask_data.shape} != rh_mask_data.shape={self.rh_mask_data.shape} != mask_data.shape={self.mask_data.shape}')
        self.threshold = threshold

    def __masking__(self, fmri_data : np.ndarray) -> np.ndarray:
        mask_data = self.mask_data.astype(np.int16)
        # -1 = non-cortical voxels, 0 = Unknown
        assert self.threshold >= 0, print(f'threshold={self.threshold} should be non-negative.')
        assert self.threshold <= max(np.max(mask_data), np.max(fmri_data)), print(f'threshold={self.threshold} should be less than or equal to the maximum value of mask_data={np.max(mask_data)} and fmri_data={np.max(fmri_data)}.')
        mask_bool = mask_data > self.threshold
        masked_data = fmri_data[mask_bool]
        return masked_data

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fmri_path = self.trial_path_dict[index]['fmri']
        image_path = self.trial_path_dict[index]['image']
        embedding_path = self.trial_path_dict[index]['embedding']

        # data
        fmri_header, fmri_data = read_nii_file(fmri_path)
        assert fmri_data.shape == self.mask_data.shape, print(f'fmri_data.shape={fmri_data.shape} != mask_data.shape={self.mask_data.shape}')
        masked_data = self.__masking__(fmri_data=fmri_data)
        image_data = np.array(Image.open(image_path))
        embedding  = np.load(embedding_path)

        return index, masked_data, image_data, embedding
    
    def __len__(self) -> int:
        return  len(self.trial_path_dict)
