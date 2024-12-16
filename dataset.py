import os
import copy
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import namedtuple 
from torch.utils.data import Dataset

from utils import run_files_path, nsd_subject_saved_dir_path, join_paths, read_nii_file, read_json_file, get_items_in_list_via_substrs

os.environ['TOKENIZERS_PARALLELISM'] = 'false' 

########################
###### NSD Dataset #####
######################## 

def fetch_nsd_rois_and_labels(functional_space : str, rois_setup : tuple) -> tuple[np.array, list[int], str]:
    """  
    Fetches the fMRI mask, thresholds, and labels for a specific ROI (Region of Interest) setup in the NSD (Natural Scenes Dataset).  

    This function retrieves the fMRI mask and associated metadata (thresholds and labels) for a given functional space  
    and ROI setup. It validates the input parameters, locates the required files, and processes the data to return  
    a structured representation of the ROI information.  

    Args:  
        functional_space (str): The resolution of the fMRI data. Must be one of:  
            ['1', '1.8', 'func1mm', 'func1pt8mm'].  
        rois_setup (tuple): A named tuple containing:  
            - derived_type (str): The type of ROI data (e.g., 'surface' or 'volume').  
            - roi_name (str): The name of the ROI to fetch.  
            - thresholds (list[int]): A list of integer threshold values for filtering the ROI data.  

    Returns:  
        tuple: A named tuple containing:  
            - fmri_mask (np.array): A 3D numpy array representing the fMRI mask for the specified ROI.  
            - thresholds (list[int]): A list of integer threshold values used for filtering the ROI data.  
            - labels_string (str): A string representation of the labels associated with the thresholds.  

    Raises:  
        AssertionError: If any of the input parameters are invalid or required files are missing.  
        ValueError: If the derived type or ROI name is not found in the available data.  
    """ 
    derived_type = rois_setup.derived_type
    roi_name = rois_setup.roi_name
    thresholds = rois_setup.thresholds

    assert os.path.exists(run_files_path), f'run_files_path={run_files_path} does not exist, please run step1_run.sh first.'
    rois_path = read_json_file(run_files_path)['ROIs'][functional_space]
    assert os.path.exists(rois_path), f'dir_path={rois_path} does not exist.'
    rois_path_dict = {} # {key=surface or volume, value=dict{key=roi_name, value=list[roi_path]}}
    for d in os.listdir(rois_path):
        derived_type_path = join_paths(rois_path, d)
        rois_path_dict[d] = {}
        for r in os.listdir(derived_type_path):
            roi_name_path = join_paths(derived_type_path, r)
            rois_path_dict[d][r] = [join_paths(roi_name_path, x) for x in os.listdir(roi_name_path)]

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
    _, fmri_mask = read_nii_file([x for x in files_path_list if '.nii.gz' in x and all(sub not in x for sub in ['lh.', 'rh.'])][0])

    label_tags = read_json_file(json_path)
    label_tags = {int(key) : value for key, value in label_tags.items()}
    labels = [label_tags[threshold] for threshold in thresholds] if not len(thresholds) == 0 else [value for key, value in label_tags.items() if key > 0]
    labels_string = '_'.join(labels)

    thresholds = thresholds if not len(thresholds) == 0 else list(range(1, int(np.max(fmri_mask))+1))

    rois_datapoint = namedtuple('rois_datapoint', ['fmri_mask', 'thresholds', 'labels_string'])
    return rois_datapoint(fmri_mask, thresholds, labels_string)

def mask_fMRI(functional_space : str, rois_datapoint : tuple[np.array, list[int], str]) -> None:
    """  
    Masks fMRI data using a specified mask and thresholds, and saves the masked data for each trial.  

    This function processes fMRI data by applying a mask and extracting data corresponding to specific  
    regions of interest (ROIs) defined by threshold values. The masked data is saved as a `.npy` file  
    for each trial in the dataset.  

    Args:  
        functional_space (str): A string representing the functional space (e.g., 'MNI', 'native') in  
            which the fMRI data is represented.  
        rois_datapoint (tuple[np.array, list[int], str]): A tuple containing:  
            - fmri_mask (np.array): A 3D numpy array representing the mask for the fMRI data.  
            - thresholds (list[int]): A list of integer values specifying the mask labels to include.  
            - labels_string (str): A string used to label the output files.  

    Returns:  
        None: The function performs file I/O operations and does not return any value.  

    Raises:  
        AssertionError: If required files or directories are missing, or if unexpected conditions  
            (e.g., multiple fMRI paths) are encountered.  
    """ 
    
    def __masking_fmri_to_array__(fmri_data : np.array, fmri_mask : np.array, thresholds : list[int]) -> np.array:
        """  
        Applies a mask to fMRI data based on specified threshold values and returns the masked data as a 1D array.  

        This function takes in fMRI data, a mask, and a list of threshold values. It identifies the voxels in the mask  
        that match the threshold values and extracts the corresponding data from the fMRI dataset. If no voxels match  
        the thresholds, the function raises an assertion error.  

        Args:  
            fmri_data (np.array): A 3D or 4D numpy array representing the fMRI data. Each voxel contains a value  
                (e.g., intensity or activation level) for a specific spatial location in the brain.  
            fmri_mask (np.array): A 3D numpy array representing the mask for the fMRI data. Each voxel in the mask  
                contains an integer label or value that can be used to filter the fMRI data.  
            thresholds (list[int]): A list of integer values specifying the mask labels to include in the output.  
                Only voxels in the mask with values matching these thresholds will be included.  

        Returns:  
            np.array: A 1D numpy array containing the fMRI data values corresponding to the voxels in the mask  
                that match the specified thresholds. If no voxels match, an assertion error is raised.  

        Raises:  
            AssertionError: If no voxels in the mask match the specified thresholds, an error is raised with a  
                descriptive message.  

        Example:  
            >>> fmri_data = np.random.rand(64, 64, 36)  # Example 3D fMRI data  
            >>> fmri_mask = np.random.randint(0, 5, size=(64, 64, 36))  # Example mask with labels 0-4  
            >>> thresholds = [1, 2]  # Include only voxels with mask values 1 or 2  
            >>> masked_data = __masking_fmri_to_array__(fmri_data, fmri_mask, thresholds)  
            >>> print(masked_data.shape)  # Output: (N,), where N is the number of matching voxels  
        """  
        fmri_mask = fmri_mask.astype(np.int16)
        mask_bool = np.isin(fmri_mask, thresholds)
        masked_data = fmri_data[mask_bool] if np.any(mask_bool) else None
        assert masked_data is not None, f'No voxels in thresholds={thresholds} found in fmri_mask.'
        return masked_data
    
    fmri_mask = rois_datapoint.fmri_mask
    thresholds = rois_datapoint.thresholds
    labels_string = rois_datapoint.labels_string

    # if done, return
    method_done_path = join_paths(nsd_subject_saved_dir_path, '_'.join([mask_fMRI.__name__, labels_string, 'done']))
    if os.path.exists(method_done_path):
        return None
    
    assert os.path.exists(run_files_path), f'run_files_path={run_files_path} does not exist, please run step1_run.sh first.'
    run_files = read_json_file(run_files_path)
    trial_path_list = [join_paths(run_files[tag], d) for tag in ['test', 'train'] for d in os.listdir(run_files[tag])]

    for trial_path in tqdm(trial_path_list, desc='Masking fMRI', leave=True):
        fmri_path = get_items_in_list_via_substrs(os.listdir(trial_path), functional_space, 'fmri')
        assert len(fmri_path) == 1, f'Found multiple fMRI paths: {fmri_path}'
        fmri_path = fmri_path[0]
        assert os.path.exists(join_paths(trial_path, fmri_path)), f'fmir_path={join_paths(trial_path, fmri_path)} does not exist.'
        _, fmri = read_nii_file(join_paths(trial_path, fmri_path))
        masked_fmri = __masking_fmri_to_array__(fmri, fmri_mask, thresholds)
        np.save(join_paths(trial_path, f'{functional_space}_{labels_string}.npy'), masked_fmri)

    # write done
    with open(method_done_path, 'w') as f:
        f.write('done')

    return None

blip2_dataset_point = namedtuple('blip2_dataset_point', 
                                ['trial', 'image', 'masked_fmri', 'image_embedding'])
# blip2_dataset_point = namedtuple('blip2_dataset_point', 
#                                 ['trial', 'image', 'masked_fmri', 'image_embedding', 'prompt', 'caption'])
# blipdiffusion_dataset_point = namedtuple('blipdiffusion_dataset_point',
#                                         ['trial', 'image', 'masked_fmri', 'image_embedding', 'caption_embedding'])
blipdiffusion_dataset_point = namedtuple('blipdiffusion_dataset_point',
                                        ['trial', 'image', 'masked_fmri', 'category', 'caption'])

class NSD_Dataset(Dataset):
    """
    load proprocessed data from hdf5 file
    """
    def __init__(self, functional_space : str, embedding_space : str, 
                 labels_string : str, set_name : str) -> None:
        super().__init__()
        self.functional_space = functional_space
        self.embedding_space = embedding_space
        self.labels_string = labels_string

        assert set_name in ['train', 'test'], f'set_name={set_name} is not supported, must be one of [train, test].'
        self.set_name = set_name

        assert os.path.exists(run_files_path), f'run_files_path={run_files_path} does not exist, please run step1_run.sh first.'
        run_files = read_json_file(run_files_path)
        self.set_files = [join_paths(run_files[set_name], x) for x in os.listdir(run_files[set_name])]

    def __get_the_only_items_in_list_via_substrs__(self, item_list : list[str], *substrs : str) -> str:
        items_list = get_items_in_list_via_substrs(item_list, *substrs)
        assert len(items_list) == 1, f'Found multiple items: {items_list}'
        return items_list[0]

    def __preprocess_masked_fmri__(self, masked_fmri : np.array) -> torch.Tensor:
        if self.labels_string == 'ventral':
            # max: max=32767.0, min=10649.0
            # min: max=-8565.0, min=-32768.0
            # return masked_fmri / 1e4
            masked_fmri = (masked_fmri-277.7413027138099) / 1349.5317774050227
        else: # whole_cortex: early_midventral_midlateral_midparietal_ventral_lateral_parietal
            masked_fmri = (masked_fmri-240.39239099458842) / 939.6683912182586
        # masked_fmri = np.clip(masked_fmri, -3, 3)
        return masked_fmri
            

    def __preprocess_image_embedding__(self, image_embedding : np.array) -> np.array:
        if self.embedding_space == 'blip2':
            # max: max=18.239103317260742, min=9.160021781921387
            # min: max=-20.08559226989746, min=-25.672496795654297
            # image_embedding = np.round(image_embedding).astype(np.int16)
            # image_embedding = np.clip(image_embedding, -1, 1)
            image_embedding = image_embedding[0]
            image_embedding = np.where(image_embedding < -0.5, 0, 
                              np.where((image_embedding >= -0.5) & (image_embedding < 0), 1, 
                              np.where((image_embedding >= 0) & (image_embedding <= 0.5), 2, 3)))
            # image_embedding = np.where(image_embedding < -0.5, 0, 
            #                   np.where((image_embedding >= -0.5) & (image_embedding < -0.25), 1, 
            #                   np.where((image_embedding >= -0.25) & (image_embedding < 0), 2, 
            #                   np.where((image_embedding >= 0) & (image_embedding < 0.25), 3, 
            #                   np.where((image_embedding >= 0.25) & (image_embedding < 0.5), 4, 5)))))
            image_embedding = np.eye(4)[image_embedding] 
            return image_embedding # [-1, 1]
    
    def __getitem__(self, index) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        dir_path = self.set_files[index]
        file_path_list = [join_paths(dir_path, x) for x in os.listdir(dir_path)]

        # trial
        trial = os.path.basename(dir_path)

        # image 
        image_path = self.__get_the_only_items_in_list_via_substrs__(file_path_list, 'image', 'png')
        image = np.array(Image.open(image_path).convert('RGB'))

        # masked_fmri
        # shape : {
        #     'func1mm_ventral'    : (44656),
        #     'func1pt8mm_ventral' : (7604 ),
        #     'func1mm_whole_cortex'    : (161510),
        # }
        masked_fmri_path = self.__get_the_only_items_in_list_via_substrs__(file_path_list, '_'.join([self.functional_space, self.labels_string]), 'npy')
        masked_fmri = np.load(masked_fmri_path, allow_pickle=True)
        masked_fmri = self.__preprocess_masked_fmri__(masked_fmri)

        # caption and category
        strings = read_json_file(self.__get_the_only_items_in_list_via_substrs__(file_path_list, 'strings', 'json'))
        caption = strings['blip2_caption']
        category = strings['category_string']
        
        # blip2: image_embedding
        if self.embedding_space == 'blip2':
            # blip2: shape=(677, 1408)
            image_embedding_path = self.__get_the_only_items_in_list_via_substrs__(file_path_list, self.embedding_space, 'image_embedding', 'npy')
            image_embedding = np.load(image_embedding_path, allow_pickle=True)
            image_embedding = self.__preprocess_image_embedding__(image_embedding)
        # bipldiffusion: image_embedding, caption_embedding
        elif self.embedding_space == 'blipdiffusion':
            # blipdiffusion: shape=(16, 768)
            image_embedding_path = self.__get_the_only_items_in_list_via_substrs__(file_path_list, self.embedding_space, 'image_embedding', 'npy')
            image_embedding = np.load(image_embedding_path, allow_pickle=True)
            # blipdiffusion: shape=(58, 768)
            caption_embedding_path = self.__get_the_only_items_in_list_via_substrs__(file_path_list, self.embedding_space, 'caption_embedding', 'npy')
            caption_embedding = np.load(caption_embedding_path, allow_pickle=True)

        # np.array -> torch.Tensor, and return 
        image = torch.tensor(image, dtype=torch.int16)
        masked_fmri = torch.tensor(masked_fmri, dtype=torch.float32)
        image_embedding = torch.tensor(image_embedding, dtype=torch.float32)
        if self.embedding_space == 'blip2': 
            return blip2_dataset_point(trial, image, masked_fmri, image_embedding)
        elif self.embedding_space == 'blipdiffusion': 
            caption_embedding = torch.tensor(caption_embedding, dtype=torch.float32)
            # return blipdiffusion_dataset_point(trial, image, masked_fmri, image_embedding, caption_embedding)
            return blipdiffusion_dataset_point(trial, image, masked_fmri, category, caption)
        
    def __len__(self) -> int:
        return len(self.set_files)
    

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