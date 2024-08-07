import os
import copy
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils import join_paths, read_nii_file, read_json_file
from utils import NSD_saved_dir_path, fmrishape_saved_dir_path

__all__ = ['NSD_Dataset',
           'make_paths_dict', 'fetch_roi_files_and_labels']

########################
###### NSD Dataset #####
######################## 

def make_paths_dict(subj_id : int, dataset_name : str
                    ) -> tuple[dict[str, dict[str, str]], 
                               dict[str, dict[str, str]], 
                               dict[str, dict[str,   list[str]]],
                               np.ndarray]:
    """  
    """  
    dataset_name = dataset_name.lower()
    subjid_string = f'subj{str(subj_id).zfill(2)}_pairs'
    if dataset_name == 'nsd':
        sujb_path = join_paths(NSD_saved_dir_path, subjid_string)
    elif dataset_name == 'fmri_shape':
        sujb_path = join_paths(fmrishape_saved_dir_path, subjid_string)
    else:
        raise ValueError(f'dataset_name={dataset_name} is not supported.')
    assert os.path.exists(sujb_path), f'dir_path={sujb_path} does not exist.'
    rois_path = join_paths(sujb_path, 'ROIs')
    assert os.path.exists(rois_path), f'dir_path={rois_path} does not exist.'
    uncond_embedding_npy_path = join_paths(sujb_path, 'uncond_embedding.npy')
    assert os.path.exists(uncond_embedding_npy_path), f'file path={uncond_embedding_npy_path} does not exist.'

    # Train or Test set
    def __make_train_or_test_path_dict__(mode : str) -> dict[str, dict[str, str]]:
        """  
        """  
        dir_path = join_paths(sujb_path, mode)
        assert os.path.exists(dir_path), f'dir_path={dir_path} does not exist.'
        trial_paths_list = [join_paths(dir_path, x) for x in os.listdir(dir_path)]
        trial_path_dict = {}
        find_and_join_paths = lambda trail_path, string: [os.path.join(trail_path, filename) for filename in os.listdir(trail_path) if string+'.' in filename]
        for index, trail_path in enumerate(trial_paths_list):
            trial_path_dict[index] = {}
            for key in ['fmri', 'image','multimodal_embedding','strings']:
                path_list = find_and_join_paths(trail_path=trail_path, string=key)
                assert len(path_list) > 0, f'No {key} files found in {trail_path}.'
                assert len(path_list) == 1, f'Multiple {key} files found in {trail_path}.'
                trial_path_dict[index][key] = path_list[0]
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

    # uncond_embedding
    uncond_embedding = np.load(uncond_embedding_npy_path)
    assert uncond_embedding.shape == (77, 768), f'uncond_embedding.shape={uncond_embedding.shape} != (77, 768)'
    return train_trial_path_dict, test_trial_path_dict, rois_path_dict, uncond_embedding

def fetch_roi_files_and_labels(derived_type : str, roi_name : str, thresholds : list[int] | list[None],
                               rois_path_dict : dict[str, dict[str, list[str]]]
                        ) -> tuple[list[str], str]:
    """  
    Fetches the file paths and corresponding label string for a specific region of interest (ROI) based on the  
    derived type and thresholds.  

    Args:  
        derived_type (str): The type of derived ROI data (e.g., 'structural', 'functional').  
        roi_name (str): The name of the ROI.  
        thresholds (list[int] | list[None]): A list of integer thresholds or None. Thresholds define the   
                                              specific labels to be included.  
        rois_path_dict (dict[str, dict[str, list[str]]]): A nested dictionary where the keys are derived types,   
                                                          and the values are dictionaries with ROI names as keys   
                                                          and lists of file paths as values.  
    
    Returns:  
        tuple[list[str], str]: A tuple containing a list of file paths related to the ROI and a concatenated   
                               label string corresponding to the thresholds.  

    Raises:  
        ValueError: If the derived_type or roi_name is not found in the rois_path_dict.  
        AssertionError: If the number of files related to the ROI is not exactly 4.  
    """   
    if not derived_type in rois_path_dict.keys():
        raise ValueError(f'derived_type should be one of {rois_path_dict.keys()}, but got {derived_type}')
    if not roi_name in rois_path_dict[derived_type].keys():
        raise ValueError(f'roi_name should be one of {rois_path_dict[derived_type].keys()}, but got {roi_name}')
    # 4 = roi_name.nii.gz, lh.name.nii.gz, rh.name.nii.gz, label_tags.json
    rois_path_dict_copy = copy.deepcopy(rois_path_dict)
    files_path_list = rois_path_dict_copy[derived_type][roi_name]
    assert len(files_path_list) == 4, print(f'{files_path_list}')
    json_path = [f for f in files_path_list if f.endswith('.json')][0]
    files_path_list.remove(json_path)

    label_tags = read_json_file(json_path)
    label_tags = {int(key) : value for key, value in label_tags.items()}
    labels = [label_tags[threshold] for threshold in thresholds] if not len(thresholds) == 0 else [value for key, value in label_tags.items() if key > 0]
    labels_string = '_'.join(labels)
    return files_path_list, labels_string

def masking_fmri_to_array(fmri_data : np.ndarray, mask_data : np.ndarray, thresholds : list[int]) -> np.ndarray:
    """  
    This function applies a mask to fMRI data and extracts the relevant data points based on the specified thresholds.  
    It ensures that the extracted data is normalized between 0 and 1.  
    
    Args:  
        fmri_data (np.ndarray): The fMRI data as a NumPy array.  
        mask_data (np.ndarray): The mask data as a NumPy array.  
        thresholds (list[int]): A list of integer threshold values to apply to the mask data.  
        
    Returns:  
        np.ndarray: A normalized NumPy array of the fMRI data corresponding to the mask.  
        
    Raises:  
        AssertionError: If no voxels in the specified thresholds are found in mask_data.  
    """ 
    mask_data = mask_data.astype(np.int16)
    mask_bool = np.isin(mask_data, thresholds)
    masked_data = fmri_data[mask_bool] if np.any(mask_bool) else None
    assert masked_data is not None, f'No voxels in thresholds={thresholds} found in mask_data.'
    # Normalize the masked_data to [0, 1]
    masked_data = (masked_data - np.min(masked_data)) / (np.max(masked_data) - np.min(masked_data))
    return masked_data

class NSD_Dataset(Dataset):
    """
    load proprocessed data
    """
    def __init__(self, trial_path_dict : dict[str, dict[str, str]], 
                 mask_path_list : list[str], 
                 thresholds : list[int] | list[None]
                 ) -> None:
        super().__init__()
        self.trial_path_dict = trial_path_dict
        self.lh_mask_header, self.lh_mask_data = read_nii_file([x for x in mask_path_list if 'lh.' in x and '.nii.gz' in x][0])
        self.rh_mask_header, self.rh_mask_data = read_nii_file([x for x in mask_path_list if 'rh.' in x and '.nii.gz' in x][0])
        self.mask_header, self.mask_data = read_nii_file([x for x in mask_path_list if '.nii.gz' in x and not 'lh.' in x and not 'rh.' in x][0])
        assert self.lh_mask_data.shape == self.rh_mask_data.shape == self.mask_data.shape, f'lh_mask_data.shape={self.lh_mask_data.shape} != rh_mask_data.shape={self.rh_mask_data.shape} != mask_data.shape={self.mask_data.shape}'
        # thresholds: -1 = non-cortical voxels, 0 = Unknown
        self.thresholds = thresholds if not len(thresholds) == 0 else list(range(1, int(np.max(self.mask_data))+1))

    def __getitem__(self, index) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # path
        fmri_path = self.trial_path_dict[index]['fmri']
        image_path = self.trial_path_dict[index]['image']
        multimodal_embedding_path = self.trial_path_dict[index]['multimodal_embedding']
        
        # data
        fmri_header, fmri_data = read_nii_file(fmri_path) # Shape of fmri_data: [145, 186, 148]
        masked_data = masking_fmri_to_array(fmri_data=fmri_data, mask_data=self.mask_data, thresholds=self.thresholds)
        image_data = np.array(Image.open(image_path))
        multimodal_embedding  = np.load(multimodal_embedding_path)

        # Shape: masked_data([K]), image_data([425, 425, 3]), multimodal_embedding([77, 768])
        return index, masked_data, image_data, multimodal_embedding
    
    def __len__(self) -> int:
        return  len(self.trial_path_dict)

########################
###### fMRI_Shape  #####
######################## 
class fMRI_Shape_Dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index) -> None:
        super().__init__()
        return None

    def __len__(self) -> int:
        return 0