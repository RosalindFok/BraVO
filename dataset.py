import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils import BraVO_saved_dir_path
from utils import join_paths, read_nii_file

__all__ = ['NSD_Dataset',
           'make_paths_dict']


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
    for derived_type in os.listdir(rois_path):
        derived_type_path = join_paths(rois_path, derived_type)
        rois_path_dict[derived_type] = {}
        for roi_name in os.listdir(derived_type_path):
            roi_name_path = join_paths(derived_type_path, roi_name)
            rois_path_dict[derived_type][roi_name] = [join_paths(roi_name_path, x) for x in os.listdir(roi_name_path)]

    return train_trial_path_dict, test_trial_path_dict, rois_path_dict

def masking_fmri_to_array(fmri_data : np.ndarray, mask_data : np.ndarray, threshold : int) -> np.ndarray | None:
    """  
    Applies a mask to the fmri_data array based on the mask_data and a given threshold.  
    
    Args:  
        fmri_data (np.ndarray): The fMRI data array to be masked.  
        mask_data (np.ndarray): The mask data array used to mask the fMRI data.  
        threshold (int): The threshold value used to create the mask. The threshold determines   
                         which voxels in the mask_data will be applied to the fMRI data.  
    
    Returns:  
        np.ndarray: The fMRI data array after masking.  
    
    Raises:  
        AssertionError: If the threshold is negative or greater than the maximum value in   
                        mask_data and fmri_data.  
    """
    mask_data = mask_data.astype(np.int16)
    # -1 = non-cortical voxels, 0 = Unknown
    assert threshold >= 0, print(f'threshold={threshold} should be non-negative.')
    assert threshold <= np.max(mask_data), print(f'threshold={threshold} should be less than or equal to the maximum value of mask_data={np.max(mask_data)}.')
    mask_bool = mask_data > threshold if threshold == 0 else mask_data == threshold
    masked_data = fmri_data[mask_bool] if np.any(mask_bool) else None
    return masked_data

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

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # path
        fmri_path = self.trial_path_dict[index]['fmri']
        image_path = self.trial_path_dict[index]['image']
        embedding_path = self.trial_path_dict[index]['embedding']
        
        # data
        fmri_header, fmri_data = read_nii_file(fmri_path)
        # Preprocessed the fmri_data: make all negative values 0, the 5% most maximum into the threshold of 5%, then normalize to [0, 1]
        fmri_data[fmri_data < 0] = 0
        sorted_fmri_data = np.sort(fmri_data.flatten())
        t = sorted_fmri_data[int(len(sorted_fmri_data) * 0.95)]
        fmri_data[fmri_data > t] = t
        fmri_data = (fmri_data - np.min(fmri_data)) / (np.max(fmri_data) - np.min(fmri_data))
        assert fmri_data.shape == self.mask_data.shape, print(f'fmri_data.shape={fmri_data.shape} != mask_data.shape={self.mask_data.shape}')
        masked_data = masking_fmri_to_array(fmri_data=fmri_data, mask_data=self.mask_data, threshold=self.threshold)
        image_data = np.array(Image.open(image_path))
        embedding  = np.load(embedding_path)

        # Shape: fmri([145, 186, 148]), masked([K]), image([425, 425, 3]), embedding([2, 77, 768])
        # embedding = [uncond_embeddings, text_embeddings]
        return index, fmri_data, masked_data, image_data, embedding
    
    def __len__(self) -> int:
        return  len(self.trial_path_dict)

