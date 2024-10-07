import os
import gc
import h5py
import copy
import torch
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import namedtuple 
from torch.utils.data import Dataset

from models import device, load_blip_models
from utils import run_files_path, join_paths, read_nii_file, check_and_make_dirs, read_json_file, get_file_size

os.environ['TOKENIZERS_PARALLELISM'] = 'false' 

########################
###### NSD Dataset #####
######################## 

def fetch_nsd_rois_and_labels(subj_path : str, rois_setup : namedtuple) -> tuple[np.ndarray, str]:
    """  
    """
    derived_type = rois_setup.derived_type
    roi_name = rois_setup.roi_name
    thresholds = rois_setup.thresholds

    rois_path = join_paths(subj_path, 'ROIs')
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
    _, mask_data = read_nii_file([x for x in files_path_list if '.nii.gz' in x and all(sub not in x for sub in ['lh.', 'rh.'])][0])

    label_tags = read_json_file(json_path)
    label_tags = {int(key) : value for key, value in label_tags.items()}
    labels = [label_tags[threshold] for threshold in thresholds] if not len(thresholds) == 0 else [value for key, value in label_tags.items() if key > 0]
    labels_string = '_'.join(labels)

    thresholds = thresholds if not len(thresholds) == 0 else list(range(1, int(np.max(mask_data))+1))

    return mask_data, thresholds, labels_string

def make_nsd_dataset(subj_path : str, mask_data : np.ndarray, thresholds : list[int], labels_string : str
                     ) -> tuple[any, any]:
    def __masking_fmri_to_array__(fmri_data : np.ndarray, 
                              mask_data : np.ndarray, thresholds : list[int]) -> np.ndarray:
        """  
        """ 
        mask_data = mask_data.astype(np.int16)
        mask_bool = np.isin(mask_data, thresholds)
        masked_data = fmri_data[mask_bool] if np.any(mask_bool) else None
        assert masked_data is not None, f'No voxels in thresholds={thresholds} found in mask_data.'
        return masked_data
    
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
    
    regions_saved_dir_path = join_paths(subj_path, labels_string)
    check_and_make_dirs(regions_saved_dir_path)

    assert os.path.exists(run_files_path), f'run_files_path={run_files_path} does not exist, please run step1_run.sh first.'
    run_files = read_json_file(join_paths(run_files_path))
    train_hdf5_path = run_files['train']['hdf5']
    train_json_path = run_files['train']['json']
    test_hdf5_path  = run_files['test']['hdf5']
    test_json_path  = run_files['test']['json']
    uncond_embedding_path = run_files['uncond_embedding_path']
    causal_attention_mask_path = run_files['causal_attention_mask_path']
    
    for tag, hdf5_path, json_path in zip(['test', 'train'], [test_hdf5_path, train_hdf5_path], [test_json_path, train_json_path]):
        set_saved_dir_path = join_paths(regions_saved_dir_path, tag)
        check_and_make_dirs(set_saved_dir_path)
        all_done_path = join_paths(set_saved_dir_path, 'all_done')
        if os.path.exists(all_done_path):
            continue
        # load blip model
        blip2_feature_extractor, bfe_vis_processors, _ = load_blip_models(mode = 'feature')
        # In json, index:string_path
        strings_path = read_json_file(json_path)
        # In hdf5, index:{image ,  fmri ,  hidden_states ,  causal_attention_mask}
        uint8_max = np.iinfo(np.uint8).max
        reshape_a, reshape_b = -99, -99
        with h5py.File(hdf5_path, 'r') as file:
            for index in tqdm(file, desc=f'process {tag}', leave=True):
                # dir
                idx_dir_path = join_paths(set_saved_dir_path, index)
                check_and_make_dirs(idx_dir_path)
                # check if already done
                done_path = join_paths(idx_dir_path, 'done')
                if os.path.exists(done_path):
                    continue
                # strings
                shutil.copy(src=strings_path[index], dst=join_paths(idx_dir_path, 'strings.json'))
                # blip output
                hidden_states = file[index]['hidden_states'][:]
                np.save(file=join_paths(idx_dir_path, 'blip_hidden_states.npy'), arr=hidden_states)
                # coco image
                image = file[index]['image'][:]
                Image.fromarray(image).save(fp=join_paths(idx_dir_path, 'coco_image.png'))
                # mask fmri -> embedding
                fmri_data = file[index]['fmri'][:]
                masked_fmri = __masking_fmri_to_array__(fmri_data=fmri_data, mask_data=mask_data, thresholds=thresholds)
                np.save(file=join_paths(idx_dir_path, 'original_masked_fmri.npy'), arr=masked_fmri)
                min_val, max_val = masked_fmri.min(), masked_fmri.max()
                masked_fmri = (masked_fmri - min_val) / (max_val - min_val) # to 0~1
                masked_fmri = (masked_fmri*uint8_max).astype(np.uint8) # to 0~255
                if max(reshape_a, reshape_b) <= 0:
                    reshape_a, reshape_b = __find_factors__(masked_fmri.shape[0])
                    if min(reshape_a, reshape_b) == 1:
                        masked_fmri = np.append(masked_fmri, 0)
                        reshape_a, reshape_b = __find_factors__(masked_fmri.shape[0])
                masked_fmri = masked_fmri.reshape(1, reshape_a, reshape_b) # (K) -> (1, a, b)
                masked_fmri = masked_fmri.repeat(3, axis=0) # (1, a, b) -> (3, a, b)
                masked_fmri = np.transpose(masked_fmri, (1, 2, 0)) # (3, a, b) -> (a, b, 3)
                masked_fmri = Image.fromarray(masked_fmri)
                masked_fmri = bfe_vis_processors['eval'](masked_fmri).cpu().numpy()
                masked_fmri = torch.from_numpy(masked_fmri).unsqueeze(0).to(device)
                masked_embedding = blip2_feature_extractor.extract_features({'image' : masked_fmri}, mode='image').image_embeds # (1, 32, 768)
                assert masked_embedding.shape == (1, 32, 768), f'{masked_embedding.shape} != (1, 32, 768)'
                masked_embedding = masked_embedding[-1].squeeze().cpu().numpy() # (32, 768)
                np.save(file=join_paths(idx_dir_path, 'blip_masked_embedding.npy'), arr=masked_embedding)
                # Done flag
                with open(done_path, 'w') as f:
                    f.write('Done')

        # delete the loaded models
        del blip2_feature_extractor, bfe_vis_processors
        torch.cuda.empty_cache()
        gc.collect() 

        with open(all_done_path, 'w') as f:
            f.write('Done')
    
    return uncond_embedding_path, causal_attention_mask_path, regions_saved_dir_path

DataPoint = namedtuple('DataPoint', ['index', 'image', 'blip_masked_embedding', 'hidden_states_image',
                                     'hidden_states_caption_fixed', 'hidden_states_caption_variable', 
                                     'strings_json_path'
                                    ])  
class NSD_Dataset(Dataset):
    """
    load proprocessed data from hdf5 file
    """
    def __init__(self, root_dir : str) -> None:
        super().__init__()
        assert os.path.exists(root_dir), f'{root_dir} does not exist.'
        self.dirs = {int(d) : os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))}

    def __split_and_concat__(self, array : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        array_1, array_2, arrayr_3 = np.split(array, [2, 18]) # BLIP decides, 77 = 2+16+59
        image_embedding = array_2
        text_embedding = np.concatenate((array_1, arrayr_3), axis=0)
        return image_embedding, text_embedding

    def __split_caption_embedding__(self, array : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert array.shape == (61, 768), f'{array.shape} != (61, 768)'
        array_0 = array[0]
        array_1 = array[1]
        array_60 = array[60]
        array_fixed = np.stack([array_0, array_1, array_60], axis=0)
        array_variable = array[2:60]
        return array_fixed, array_variable
        
    def __getitem__(self, index) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        dir_path = self.dirs[index]
        image = np.array(Image.open(os.path.join(dir_path, 'coco_image.png')))
        blip_hidden_states = np.load(os.path.join(dir_path, 'blip_hidden_states.npy'), allow_pickle=True)
        hidden_states_image, hidden_states_caption = self.__split_and_concat__(np.squeeze(blip_hidden_states))
        hidden_states_caption_fixed, hidden_states_caption_variable = self.__split_caption_embedding__(hidden_states_caption)
        blip_masked_embedding = np.squeeze(np.load(os.path.join(dir_path, 'blip_masked_embedding.npy'), allow_pickle=True))
        
        blip_masked_embedding, _ = self.__split_and_concat__(blip_masked_embedding)
        
        strings_json_path = os.path.join(dir_path, 'strings.json')

        image = torch.tensor(image, dtype=torch.float32)                                 # (425, 425, 3)
        blip_masked_embedding = torch.tensor(blip_masked_embedding, dtype=torch.float32) # (77, 768)
        hidden_states_image = torch.tensor(hidden_states_image, dtype=torch.float32)     # (16, 768)
        hidden_states_caption_fixed = torch.tensor(hidden_states_caption_fixed, dtype=torch.float32) # (3, 768)
        hidden_states_caption_variable = torch.tensor(hidden_states_caption_variable, dtype=torch.float32) # (58, 768)

        return DataPoint(index, image, blip_masked_embedding, hidden_states_image, hidden_states_caption_fixed, 
                         hidden_states_caption_variable, strings_json_path)

    def __len__(self) -> int:
        return len(self.dirs)
    

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