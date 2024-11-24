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
from utils import run_files_path, join_paths, read_nii_file, check_and_make_dirs, read_json_file, BLIP_Prior_Tools, get_items_in_list_via_substrs

os.environ['TOKENIZERS_PARALLELISM'] = 'false' 

########################
###### NSD Dataset #####
######################## 

def fetch_nsd_rois_and_labels(functional_space : str, rois_setup : namedtuple) -> tuple[np.array, str]:
    """  
    """
    derived_type = rois_setup.derived_type
    roi_name = rois_setup.roi_name
    thresholds = rois_setup.thresholds

    assert os.path.exists(run_files_path), f'run_files_path={run_files_path} does not exist, please run step1_run.sh first.'
    rois_path = read_json_file(run_files_path)[functional_space]['rois_dir_path']
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

def make_nsd_dataset(subj_path : str, functional_space : str, mask_data : np.array, 
                     thresholds : list[int], labels_string : str, fmri2embedding : bool = False 
                     ) -> tuple[np.array, np.array, np.array, np.array, np.array, str, str]:
    def __masking_fmri_to_array__(fmri_data : np.array, mask_data : np.array, thresholds : list[int]) -> np.array:
        """  
        """ 
        mask_data = mask_data.astype(np.int16)
        mask_bool = np.isin(mask_data, thresholds)
        masked_data = fmri_data[mask_bool] if np.any(mask_bool) else None
        assert masked_data is not None, f'No voxels in thresholds={thresholds} found in mask_data.'
        return masked_data
    
    # Save data as differnet ROIs(labels_string)
    regions_saved_dir_path = join_paths(subj_path, functional_space, labels_string)
    check_and_make_dirs(regions_saved_dir_path)

    assert os.path.exists(run_files_path), f'run_files_path={run_files_path} does not exist, please run step1_run.sh first.'
    run_files = read_json_file(join_paths(run_files_path))
    uncond_embedding_path = run_files['uncond_embedding_path']
    position_embeddings_path = run_files['position_embeddings_path']
    causal_attention_mask_path = run_files['causal_attention_mask_path']
    null_sample_hidden_states_path = run_files['null_sample_hidden_states_path']
    caption_embedding_fixed_path = run_files['caption_embedding_fixed_path']
    nullI_nullC_image_path = run_files['nullI_nullC_image_path']

    uncond_embedding = np.load(uncond_embedding_path, allow_pickle=True)
    assert uncond_embedding.shape == (1, 77, 768), f'uncond_embedding.shape={uncond_embedding.shape} != (1, 77, 768)'
    position_embeddings = np.squeeze(np.load(position_embeddings_path, allow_pickle=True))
    assert position_embeddings.shape == (77, 768), f'position_embeddings.shape={position_embeddings.shape} != (77, 768)'
    causal_attention_mask = np.load(causal_attention_mask_path, allow_pickle=True)
    assert causal_attention_mask.shape == (1, 1, 77, 77), f'causal_attention_mask.shape={causal_attention_mask.shape} != (1, 1, 77, 77)'
    null_sample_hidden_states = np.load(null_sample_hidden_states_path, allow_pickle=True)
    assert null_sample_hidden_states.shape == (77, 768), f'null_sample_hidden_states.shape={null_sample_hidden_states.shape}!= (77, 768)'
    caption_embedding_fixed = np.load(caption_embedding_fixed_path, allow_pickle=True)
    assert caption_embedding_fixed.shape == (3, 768), f'caption_embedding_fixed.shape={caption_embedding_fixed.shape} != (3, 768)'

    for tag in ['test', 'train']:
        json_path = run_files[functional_space][tag]['json']
        dirs_path = run_files[functional_space][tag]['dirs']
        strings_path = read_json_file(json_path) # index:string_path
        
        set_saved_dir_path = join_paths(regions_saved_dir_path, tag)
        check_and_make_dirs(set_saved_dir_path)

        # check all done, if exists, skip
        all_done_path = join_paths(set_saved_dir_path, '_'.join([str(fmri2embedding), 'all_done']))
        if os.path.exists(all_done_path):
            continue

        # load blip model
        if fmri2embedding:
            blip2_feature_extractor, bfe_vis_processors, _ = load_blip_models(mode = 'feature')

        # Process fMRI, copy blipcation, cococaptions, coco_image, strings and done
        for dir_path in tqdm(os.listdir(dirs_path), desc=f'process {tag}', leave=True):
            if os.path.isdir(join_paths(dirs_path, dir_path)):
                index = dir_path
                # src dir 
                src_dir_path = join_paths(dirs_path, dir_path)
                assert os.path.exists(src_dir_path), f'{src_dir_path} does not exist'
                # files path
                files_path = [x for x in os.listdir(src_dir_path)]
                fmri_path       = get_items_in_list_via_substrs(files_path, 'fmri')[0]
                done_path       = get_items_in_list_via_substrs(files_path, 'done')[0]
                coco_image_path = get_items_in_list_via_substrs(files_path, 'coco_image')[0]
                blipcaption_path  = get_items_in_list_via_substrs(files_path, 'blipcaption')[0]
                cococaptions_path = get_items_in_list_via_substrs(files_path, 'cococaptions')[0]
                # dst dir
                idx_dir_path = join_paths(set_saved_dir_path, index)
                check_and_make_dirs(idx_dir_path)
                
                # check done, if exists, skip
                done_path = join_paths(idx_dir_path, '_'.join([str(fmri2embedding), 'done']))
                if os.path.exists(done_path):
                    continue

                # [coco image, blip caption, coco captions]
                for basename in [coco_image_path, blipcaption_path, cococaptions_path]:
                    src_path = join_paths(src_dir_path, basename)
                    dst_path = join_paths(idx_dir_path, basename)
                    if not os.path.exists(dst_path):
                        shutil.copy(src_path, dst_path)
                # strings
                if not os.path.exists(join_paths(idx_dir_path, 'strings.json')):
                    shutil.copy(src=strings_path[index], dst=join_paths(idx_dir_path, 'strings.json'))
                # fMRI -> maked fMRI
                fmri_data = np.load(join_paths(src_dir_path, fmri_path), allow_pickle=True)
                masked_fmri = __masking_fmri_to_array__(fmri_data=fmri_data, mask_data=mask_data, thresholds=thresholds)
                np.save(file=join_paths(idx_dir_path, 'masked_fmri.npy'), arr=masked_fmri)
                if fmri2embedding:
                    # masked fMRI -> embedding (32, 768)
                    masked_fmri = (masked_fmri - np.iinfo(np.int16).min) / (np.iinfo(np.int16).max - np.iinfo(np.int16).min) # to 0~1
                    masked_fmri = (masked_fmri*np.iinfo(np.uint8).max).astype(np.uint8) # to 0~255
                    masked_fmri = masked_fmri.reshape(1, masked_fmri.shape[0], 1) # (K) -> (1, K, 1)
                    masked_fmri = masked_fmri.repeat(3, axis=0) # (1, K, 1) -> (3, K, 1)
                    masked_fmri = np.transpose(masked_fmri, (1, 2, 0)) # (3, K, 1) -> (K, 1, 3)
                    masked_fmri = Image.fromarray(masked_fmri)
                    masked_fmri = bfe_vis_processors['eval'](masked_fmri).unsqueeze(0).to(device)
                    masked_fmri_embedding = blip2_feature_extractor.extract_features({'image' : masked_fmri}, mode='image').image_embeds # (1, 32, 768)
                    masked_fmri_embedding = masked_fmri_embedding.squeeze().cpu().numpy()
                    assert masked_fmri_embedding.shape == (32, 768), f'{masked_fmri_embedding.shape} != (32, 768)'
                    np.save(file=join_paths(idx_dir_path, 'masked_fmri_embedding.npy'), arr=masked_fmri_embedding)
                
                # write done
                with open(done_path, 'w') as f:
                    f.write('Done')

        # delete the loaded models
        del blip2_feature_extractor, bfe_vis_processors
        torch.cuda.empty_cache()
        gc.collect() 

        # write all_done
        with open(all_done_path, 'w') as f:
            f.write('Done')
    
    return namedtuple('dataPoint', ['uncond_embedding', 'position_embeddings', 'causal_attention_mask', 
                                    'null_sample_hidden_states', 'caption_embedding_fixed',
                                    'regions_saved_dir_path', 'nullI_nullC_image_path'
                                   ])(uncond_embedding, position_embeddings, causal_attention_mask, 
                                      null_sample_hidden_states, caption_embedding_fixed,
                                      regions_saved_dir_path, nullI_nullC_image_path
                                   )

dataset_point = namedtuple('dataset_point', ['index', 'coco_image', 'masked_fmri', 'strings_json_path',
                                             'image_embedding', 'caption_embedding_variable'])
class NSD_Dataset(Dataset):
    """
    load proprocessed data from hdf5 file
    """
    def __init__(self, root_dir : str, caption_type : str) -> None:
        super().__init__()
        assert os.path.exists(root_dir), f'{root_dir} does not exist.'
        self.dirs = {int(d) : os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))}
        assert caption_type in ['blip', 'coco', 'b', 'c'], f'caption_type={caption_type} is not supported, must be one of [blip, coco, b, c].'
        self.embedding_dir_basename = 'blipcaption' if caption_type in ['blip', 'b'] else 'cococaptions'

    def __process_masked_fmri__(self, masked_fmri : np.array, 
                                clip_min : int = np.iinfo(np.int16).min, 
                                clip_max : int = np.iinfo(np.int16).max) -> np.array:
        masked_fmri = np.clip(masked_fmri, clip_min, clip_max)
        masked_fmri /= np.abs(clip_min)
        return masked_fmri
    
    def __process_blip_image_embedding__(self, blip_image_embedding : np.array) -> np.array:
        blip_image_embedding = np.clip(blip_image_embedding, -2.1, 2.1)
        # blip_image_embedding = np.around(blip_image_embedding, 1) # max=4.3, min=-5.6
        # blip_image_embedding = np.clip(blip_image_embedding, -2.1, 2.1) # max=2.1, min=-2.1
        # blip_image_embedding += 2.1 # max=4.2, min=0
        # blip_image_embedding = (blip_image_embedding*10).astype(np.uint8) # max=42, min=0
        # assert blip_image_embedding.max() <= 42, f'{blip_image_embedding.max()} > 42'
        # # one-hot encoding
        # blip_image_embedding = np.eye(42+1, dtype=np.uint8)[blip_image_embedding] 
        return blip_image_embedding
    
    def __getitem__(self, index) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        dir_path = self.dirs[index]
        
        '''
        original image
        coco_image.shape = (425, 425, 3)
        '''
        coco_image = np.array(Image.open(os.path.join(dir_path, 'coco_image.png')))
        
        '''
        input tensor: Brain Space
        masked_fmri.shape = (k,)
        func1mm-early: k=34509
        masked_fmri_embedding.shape = (32, 768)
        '''
        masked_fmri = np.load(os.path.join(dir_path, 'masked_fmri.npy'), allow_pickle=True)
        masked_fmri = self.__process_masked_fmri__(masked_fmri, clip_min=-2846, clip_max=4117)
        # masked_fmri_embedding = np.load(os.path.join(dir_path, 'masked_fmri_embedding.npy'), allow_pickle=True)

        '''
        target embedding: BLIP Space
        blip_image_embedding.shape = (16, 768)
        blip_caption_embedding_variable.shape = (58, 768)
        '''
        embedding_dir_path = join_paths(dir_path, self.embedding_dir_basename)
        image_embedding = np.load(join_paths(embedding_dir_path, 'image_embedding.npy'), allow_pickle=True)
        image_embedding = self.__process_blip_image_embedding__(image_embedding)
        caption_embedding_variable = np.load(join_paths(embedding_dir_path, 'caption_embedding_variable.npy'), allow_pickle=True)
        
        '''
        strings: json file
        '''
        strings_json_path = os.path.join(dir_path, 'strings.json')
        
        # ndarray -> tensor
        coco_image = torch.tensor(coco_image, dtype=torch.float32)
        masked_fmri = torch.tensor(masked_fmri, dtype=torch.float32)                   
        image_embedding = torch.tensor(image_embedding, dtype=torch.float32)  
        caption_embedding_variable = torch.tensor(caption_embedding_variable, dtype=torch.float32)

        # Return data point
        return dataset_point(index, coco_image, masked_fmri, strings_json_path,
                             image_embedding, caption_embedding_variable)

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