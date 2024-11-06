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
from utils import run_files_path, join_paths, read_nii_file, check_and_make_dirs, read_json_file, BLIP_Prior_Tools

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
                     ) -> tuple[np.array, np.array, np.array, np.array, str]:
    def __masking_fmri_to_array__(fmri_data : np.ndarray, 
                              mask_data : np.ndarray, thresholds : list[int]) -> np.ndarray:
        """  
        """ 
        mask_data = mask_data.astype(np.int16)
        mask_bool = np.isin(mask_data, thresholds)
        masked_data = fmri_data[mask_bool] if np.any(mask_bool) else None
        assert masked_data is not None, f'No voxels in thresholds={thresholds} found in mask_data.'
        return masked_data
    
    regions_saved_dir_path = join_paths(subj_path, labels_string)
    check_and_make_dirs(regions_saved_dir_path)

    assert os.path.exists(run_files_path), f'run_files_path={run_files_path} does not exist, please run step1_run.sh first.'
    run_files = read_json_file(join_paths(run_files_path))
    train_hdf5_path = run_files['train']['hdf5']
    train_json_path = run_files['train']['json']
    test_hdf5_path  = run_files['test']['hdf5']
    test_json_path  = run_files['test']['json']
    uncond_embedding_path = run_files['uncond_embedding_path']
    position_embeddings_path = run_files['position_embeddings_path']
    causal_attention_mask_path = run_files['causal_attention_mask_path']
    null_sample_hidden_states_path = run_files['null_sample_hidden_states_path']

    uncond_embedding = np.load(uncond_embedding_path, allow_pickle=True)
    assert uncond_embedding.shape == (1, 77, 768), f'uncond_embedding.shape={uncond_embedding.shape} != (1, 77, 768)'
    position_embeddings = np.squeeze(np.load(position_embeddings_path, allow_pickle=True))
    assert position_embeddings.shape == (77, 768), f'position_embeddings.shape={position_embeddings.shape} != (77, 768)'
    causal_attention_mask = np.load(causal_attention_mask_path, allow_pickle=True)
    assert causal_attention_mask.shape == (1, 1, 77, 77), f'causal_attention_mask.shape={causal_attention_mask.shape} != (1, 1, 77, 77)'
    null_sample_hidden_states = np.load(null_sample_hidden_states_path, allow_pickle=True)
    assert null_sample_hidden_states.shape == (1, 77, 768), f'null_sample_hidden_states.shape={null_sample_hidden_states.shape}!= (1, 77, 768)'
    
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
        # In hdf5, index:{image ,  fmri ,  hidden_states}
        uint8_max = np.iinfo(np.uint8).max
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
                hidden_states = np.squeeze(file[index]['hidden_states'][:]) # (77, 768)
                assert hidden_states.shape == position_embeddings.shape, f'hidden_states.shape={hidden_states.shape} != position_embeddings.shape={position_embeddings.shape}'
                hidden_states_minus_position_embedding = hidden_states - position_embeddings
                np.save(file=join_paths(idx_dir_path, 'hidden_states_minus_position_embedding.npy'), arr=hidden_states_minus_position_embedding)
                # coco image
                image = file[index]['image'][:]
                Image.fromarray(image).save(join_paths(idx_dir_path, 'coco_image.png'))
                # mask fmri -> embedding
                fmri_data = file[index]['fmri'][:]
                masked_fmri = __masking_fmri_to_array__(fmri_data=fmri_data, mask_data=mask_data, thresholds=thresholds)
                np.save(file=join_paths(idx_dir_path, 'masked_fmri.npy'), arr=masked_fmri)
                min_val, max_val = masked_fmri.min(), masked_fmri.max()
                masked_fmri = (masked_fmri - min_val) / (max_val - min_val) # to 0~1
                masked_fmri = (masked_fmri*uint8_max).astype(np.uint8) # to 0~255
                masked_fmri = masked_fmri.reshape(1, masked_fmri.shape[0], 1) # (K) -> (1, K, 1)
                masked_fmri = masked_fmri.repeat(3, axis=0) # (1, K, 1) -> (3, K, 1)
                masked_fmri = np.transpose(masked_fmri, (1, 2, 0)) # (3, K, 1) -> (K, 1, 3)
                masked_fmri = Image.fromarray(masked_fmri)
                masked_fmri = bfe_vis_processors['eval'](masked_fmri).unsqueeze(0).to(device)
                masked_fmri_embedding = blip2_feature_extractor.extract_features({'image' : masked_fmri}, mode='image').image_embeds # (1, 32, 768)
                masked_fmri_embedding = masked_fmri_embedding.squeeze().cpu().numpy()
                assert masked_fmri_embedding.shape == (32, 768), f'{masked_fmri_embedding.shape} != (32, 768)'
                np.save(file=join_paths(idx_dir_path, 'masked_fmri_embedding.npy'), arr=masked_fmri_embedding)
                # Done flag
                with open(done_path, 'w') as f:
                    f.write('Done')

        # delete the loaded models
        del blip2_feature_extractor, bfe_vis_processors
        torch.cuda.empty_cache()
        gc.collect() 

        with open(all_done_path, 'w') as f:
            f.write('Done')
    
    return namedtuple('dataPoint', ['uncond_embedding', 'position_embeddings', 'causal_attention_mask', 
                                    'null_sample_hidden_states', 'regions_saved_dir_path'
                                   ])(uncond_embedding, position_embeddings, causal_attention_mask, 
                                      null_sample_hidden_states, regions_saved_dir_path
                                   )
     

NSD_Dataset_DataPoint = namedtuple('NSD_Dataset_DataPoint', 
                                ['index', 'coco_image', 'masked_fmri', 'masked_fmri_embedding',
                                 'blip_image_embedding', 
                                 'blip_caption_embedding_fixed', 'blip_caption_embedding_variable', 
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

    def __getitem__(self, index) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        dir_path = self.dirs[index]
        # read data
        coco_image = np.array(Image.open(os.path.join(dir_path, 'coco_image.png')))
        blip_hidden_states = np.load(os.path.join(dir_path, 'hidden_states_minus_position_embedding.npy'), allow_pickle=True)
        blip_image_embedding, blip_caption_embedding = BLIP_Prior_Tools.split_and_concat(np.squeeze(blip_hidden_states))
        blip_caption_embedding_fixed, blip_caption_embedding_variable = BLIP_Prior_Tools.split_caption_embedding(blip_caption_embedding)
        masked_fmri = np.load(os.path.join(dir_path, 'masked_fmri.npy'), allow_pickle=True)
        masked_fmri -= np.iinfo(np.int16).min # [-32768, 32767] -> [0, 65535]
        masked_fmri_embedding = np.load(os.path.join(dir_path, 'masked_fmri_embedding.npy'), allow_pickle=True)
        strings_json_path = os.path.join(dir_path, 'strings.json')
        
        # around
        blip_image_embedding = np.clip(blip_image_embedding, -2.1, 2.1)
        # blip_image_embedding = np.around(blip_image_embedding, 1) # max=4.3, min=-5.6
        # blip_image_embedding = np.clip(blip_image_embedding, -2.1, 2.1) # max=2.1, min=-2.1
        # blip_image_embedding += 2.1 # max=4.2, min=0
        # blip_image_embedding = (blip_image_embedding*10 + 1).astype(np.uint8) # max=43, min=1
        # assert blip_image_embedding.max() <= 42, f'{blip_image_embedding.max()} > 42'
        # # one-hot encoding
        # blip_image_embedding = np.eye(42+1, dtype=np.uint8)[blip_image_embedding] 

        # ndarray -> tensor
        coco_image = torch.tensor(coco_image, dtype=torch.float32)                     # (425, 425, 3)
        masked_fmri = torch.tensor(masked_fmri, dtype=torch.int32)                     # (K,)
        masked_fmri_embedding = torch.tensor(masked_fmri_embedding, dtype=torch.float32) # (32, 768)
        blip_image_embedding = torch.tensor(blip_image_embedding, dtype=torch.float32)   # (16, 768)
        blip_caption_embedding_fixed = torch.tensor(blip_caption_embedding_fixed, dtype=torch.float32)       # (3, 768)
        blip_caption_embedding_variable = torch.tensor(blip_caption_embedding_variable, dtype=torch.float32) # (58, 768)

        return NSD_Dataset_DataPoint(index, coco_image, masked_fmri, masked_fmri_embedding, blip_image_embedding, 
                         blip_caption_embedding_fixed, blip_caption_embedding_variable, strings_json_path)

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