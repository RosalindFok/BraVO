import os
import gc
import h5py
import copy
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import namedtuple 
from torch.utils.data import Dataset

from models import device, load_blip_models
from utils import join_paths, read_nii_file, check_and_make_dirs, read_json_file, get_file_size

__all__ = ['NSD_Dataset',
           'make_paths_dict', 'fetch_roi_files_and_labels']

########################
###### NSD Dataset #####
######################## 

def make_paths_dict(subj_path : str) -> tuple[dict[str, dict[str, str]], 
                                              dict[str, dict[str, str]], 
                                              dict[str, dict[str, list[str]]],
                                              np.ndarray, np.ndarray]:
    """  
    """  
    rois_path = join_paths(subj_path, 'ROIs')
    assert os.path.exists(rois_path), f'dir_path={rois_path} does not exist.'
    uncond_embedding_npy_path = join_paths(subj_path, 'uncond_embedding.npy')
    assert os.path.exists(uncond_embedding_npy_path), f'file path={uncond_embedding_npy_path} does not exist.'
    causal_attention_mask_path = join_paths(subj_path, 'causal_attention_mask.npy')
    assert os.path.exists(causal_attention_mask_path), f'file path={causal_attention_mask_path} does not exist.'

    def __make_path_dict__(mode : str) -> dict[str, dict[str, str]]:
        """  
        """  
        dir_path = join_paths(subj_path, mode)
        assert os.path.exists(dir_path), f'dir_path={dir_path} does not exist.'
        trial_paths_list = [join_paths(dir_path, x) for x in os.listdir(dir_path)]
        trial_path_dict = {}
        find_and_join_paths = lambda trail_path, string: [os.path.join(trail_path, filename) for filename in os.listdir(trail_path) if string+'.' in filename]
        for index, trail_path in tqdm(enumerate(trial_paths_list), desc=f'Making {mode} paths dict', leave=True, total=len(trial_paths_list)):
            trial_path_dict[index] = {}
            for key in ['fmri', 'image', 'canny', 'hidden_states', 'strings']:
                path_list = find_and_join_paths(trail_path=trail_path, string=key)
                assert len(path_list) > 0, f'No {key} files found in {trail_path}.'
                assert len(path_list) == 1, f'Multiple {key} files found in {trail_path}.'
                trial_path_dict[index][key] = path_list[0]
        return trial_path_dict
        
    train_trial_path_dict = __make_path_dict__(mode = 'train')
    test_trial_path_dict  = __make_path_dict__(mode = 'test')

    # ROIs
    rois_path_dict = {} # {key=surface or volume, value=dict{key=roi_name, value=list[roi_path]}}
    for derived_type in os.listdir(rois_path):
        derived_type_path = join_paths(rois_path, derived_type)
        rois_path_dict[derived_type] = {}
        for roi_name in os.listdir(derived_type_path):
            roi_name_path = join_paths(derived_type_path, roi_name)
            rois_path_dict[derived_type][roi_name] = [join_paths(roi_name_path, x) for x in os.listdir(roi_name_path)]

    # uncond_embedding
    uncond_embedding = np.load(uncond_embedding_npy_path, allow_pickle=True)
    assert uncond_embedding.shape == (1, 77, 768), f'uncond_embedding.shape={uncond_embedding.shape} != (1, 77, 768)'
    # causal_attention_mask
    causal_attention_mask = np.load(causal_attention_mask_path, allow_pickle=True)
    assert causal_attention_mask.shape == (1, 1, 77, 77), f'causal_attention_mask.shape={causal_attention_mask.shape} != (1, 1, 77, 77)'
    return train_trial_path_dict, test_trial_path_dict, rois_path_dict, uncond_embedding, causal_attention_mask

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

def make_hdf5(trial_path_dict : dict[str, dict[str, str]], 
              mask_path_list : list[str], 
              thresholds : list[int] | list[None],
              hdf5_path : str, temp_dir_path : str, device : torch.device = device
            ) -> None:
    """  
    """ 
    def __find_factors__(n : int) -> tuple[int, int]:
        """
        """
        a = int(n**0.5)  
        b = n // a  
        min_diff = float('inf')  
        best_a, best_b = a, b  
        for i in range(a, 0, -1):  
            if n % i == 0: 
                j = n // i  
                diff = abs(i - j)  
                if diff < min_diff:  
                    min_diff = diff  
                    best_a, best_b = i, j  
        return best_a, best_b
    
    def __masking_fmri_to_array__(fmri_data : np.ndarray, 
                              mask_data : np.ndarray, thresholds : list[int]) -> np.ndarray:
        """  
        """ 
        mask_data = mask_data.astype(np.int16)
        mask_bool = np.isin(mask_data, thresholds)
        masked_data = fmri_data[mask_bool] if np.any(mask_bool) else None
        assert masked_data is not None, f'No voxels in thresholds={thresholds} found in mask_data.'
        return masked_data
    
    if os.path.exists(hdf5_path):
        try:
            with h5py.File(hdf5_path, 'r') as hdf5_file:  
                if  len(list(hdf5_file.keys())) == len(trial_path_dict):
                    print(f'{hdf5_path} already exists, the size of it is {get_file_size(hdf5_path)}')
                    return None
        except Exception as e:
            os.remove(hdf5_path)
            print(f'{hdf5_path} is removed.')

    blip_diffusion_model, bd_vis_processors, bd_txt_processors = load_blip_models(mode = 'diffusion')
    _, mask_data = read_nii_file([x for x in mask_path_list if '.nii.gz' in x and all(sub not in x for sub in ['lh.', 'rh.'])][0])
    thresholds = thresholds if not len(thresholds) == 0 else list(range(1, int(np.max(mask_data))+1))

    npz_files_path_dict = {} # {index : path}
    reshape_a, reshape_b = -99, -99
    for index, path_dict in tqdm(trial_path_dict.items(), desc=f'Masking fMRI and Reading imgs', leave=True):
        npz_path = join_paths(temp_dir_path, f'{index}.npz')
        npz_files_path_dict[index] = npz_path
        # check
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            del data
            continue
        # path
        image_path = path_dict['image']
        canny_path = path_dict['canny'] 
        hidden_states_path = path_dict['hidden_states']
        fmri_path = path_dict['fmri']
        # data
        _, fmri_data = read_nii_file(fmri_path) # Shape of fmri_data: [145, 186, 148]
        masked_data = __masking_fmri_to_array__(fmri_data=fmri_data, mask_data=mask_data, thresholds=thresholds)
        masked_data = (masked_data - np.min(masked_data)) / (np.max(masked_data) - np.min(masked_data)) # to 0~1
        if max(reshape_a, reshape_b) <= 0:
            reshape_a, reshape_b = __find_factors__(masked_data.shape[0])
            if min(reshape_a, reshape_b) == 1:
                masked_data = np.append(masked_data, 0)
                reshape_a, reshape_b = __find_factors__(masked_data.shape[0])
        masked_data = (masked_data*(np.iinfo(np.uint8).max)).astype(np.uint8) # to 0~255
        masked_data = masked_data.reshape(1, reshape_a, reshape_b).repeat(3, axis=0)
        masked_data = np.transpose(masked_data, (1, 2, 0))
        masked_data = Image.fromarray(masked_data)
        masked_data = bd_vis_processors['eval'](masked_data).cpu().numpy()
        np.savez(npz_path,  masked_data=masked_data,
                            image=np.array(Image.open(image_path)),
                            canny=np.array(Image.open(canny_path)),
                            hidden_states=np.squeeze(np.load(hidden_states_path, allow_pickle=True))
            )
        del masked_data, fmri_data

    prompt =  [bd_txt_processors['eval']('')]   

    with h5py.File(hdf5_path, 'w') as hdf5_file:
        for index, npz_path in tqdm(npz_files_path_dict.items(), desc=f'{hdf5_path.split(os.sep)[-1]}', leave=True):
            hdf5_group = hdf5_file.create_group(name=str(index))
            content = np.load(npz_path, allow_pickle=True)
            cond_images = torch.from_numpy(content['masked_data']).unsqueeze(0).to(device)
            sample = {
                'cond_images' : cond_images,
                'prompt' : prompt,
                'cond_subject' : 'fMRI',
                'tgt_subject'  : 'natural scenes image'
            }     
            masked_embedding, _ = blip_diffusion_model.generate_embedding(samples=sample) 
            masked_embedding = masked_embedding[-1].squeeze().cpu().numpy()
            for name, data in zip(['masked_embedding', 'image', 'canny', 'hidden_states'], 
                                  [masked_embedding, content['image'], content['canny'],  content['hidden_states']]):
                hdf5_dataset = hdf5_group.create_dataset(name=name, shape=data.shape, dtype=data.dtype)
                hdf5_dataset[:] = data
    print(f'{hdf5_path} is created, the size of it is {get_file_size(hdf5_path)}')

    # Release memory and garbage collect
    del blip_diffusion_model, bd_vis_processors, bd_txt_processors
    torch.cuda.empty_cache()
    gc.collect() 

DataPoint = namedtuple('DataPoint', ['index', 'masked_embedding_image', 'masked_embedding_caption',   
                                     'image', 'canny', 'hidden_states_image', 'hidden_states_caption'])  
class NSD_HDF5_Dataset(Dataset):
    """
    load proprocessed data from hdf5 file
    """
    def __init__(self, hdf5_path : str) -> None:
        super().__init__()
        assert os.path.exists(hdf5_path), f'{hdf5_path} does not exist.'
        self.hdf5_file = h5py.File(hdf5_path, 'r') 

    def __split_and_concat__(self, array : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        array_1, array_2, arrayr_3 = np.split(array, [2, 18]) # BLIP decides, 77 = 2+16+59
        image_embedding = array_2
        text_embedding = np.concatenate((array_1, arrayr_3), axis=0)
        return image_embedding, text_embedding

    def __getitem__(self, index) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.hdf5_file[str(index)]
        masked_embedding = data['masked_embedding'][:]
        image = data['image'][:]
        canny = data['canny'][:]
        hidden_states = data['hidden_states'][:]
        masked_embedding_image, masked_embedding_caption = self.__split_and_concat__(masked_embedding)
        hidden_states_image, hidden_states_caption = self.__split_and_concat__(hidden_states)
        masked_embedding_image = torch.tensor(masked_embedding_image, dtype=torch.float32) # [16, 768]
        masked_embedding_caption = torch.tensor(masked_embedding_caption, dtype=torch.float32)   # [61, 768]
        image = torch.tensor(image, dtype=torch.float32)                                   # [425, 425, 3]
        canny = torch.tensor(canny, dtype=torch.float32)                                   # [448, 448, 3] 
        hidden_states_image = torch.tensor(hidden_states_image, dtype=torch.float32)       # [16, 768]
        hidden_states_caption = torch.tensor(hidden_states_caption, dtype=torch.float32)   # [61, 768]
        return DataPoint(index, masked_embedding_image, masked_embedding_caption, 
                         image, canny, hidden_states_image, hidden_states_caption)

    def __len__(self) -> int:
        return len(self.hdf5_file.keys())
    
    def __del__(self):  
        self.hdf5_file.close()  
        
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