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

from config import configs_dict
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
                np.save(file=join_paths(idx_dir_path, 'fmri.npy'), arr=fmri_data)
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

def generate_null_blip_images_via_embedding(regions_saved_dir_path : str,
                                            uncond_embedding : np.ndarray,
                                            position_embeddings : np.ndarray,
                                            causal_attention_mask : np.ndarray,
                                            null_img_embedding : np.ndarray,
                                            null_cap_embedding : np.ndarray) -> None:
    '''
    '''
    uncond_embedding      = torch.from_numpy(uncond_embedding).to(device)
    causal_attention_mask = torch.from_numpy(causal_attention_mask).to(device)
    blip_diffusion_model, _, _ = load_blip_models(mode = 'diffusion')
    test_train_dir_path_list = [join_paths(regions_saved_dir_path, x) for x in os.listdir(regions_saved_dir_path) if os.path.isdir(join_paths(regions_saved_dir_path, x))]
    ### test
    test_train_dir_path_list = [test_train_dir_path_list[0]]
    ### test
    samples_dir_path_list = [join_paths(test_train_dir_path, x) for test_train_dir_path in test_train_dir_path_list for x in os.listdir(test_train_dir_path) if os.path.isdir(join_paths(test_train_dir_path, x))]
    for cnt, dir_path in enumerate(samples_dir_path_list):  
        print(f"Now is generating {dir_path}, processed {cnt+1}/{len(samples_dir_path_list)}")
        coco_matrix = np.array(Image.open(join_paths(dir_path, 'coco_image.png')))
        hidden_states_minus_position_embedding = np.load(join_paths(dir_path, 'hidden_states_minus_position_embedding.npy'), allow_pickle=True)
        assert hidden_states_minus_position_embedding.shape == position_embeddings.shape, f'hidden_states_minus_position_embedding.shape={hidden_states_minus_position_embedding.shape} != position_embeddings.shape={position_embeddings.shape}'
        hidden_states = hidden_states_minus_position_embedding + position_embeddings
        blip_image_embedding, blip_caption_embedding = BLIP_Prior_Tools.split_and_concat(np.squeeze(hidden_states))
        blip_caption_embedding_fixed, blip_caption_embedding_variable = BLIP_Prior_Tools.split_caption_embedding(blip_caption_embedding)
        np.save(file=join_paths(dir_path, 'blip_image_embedding.npy'), arr=blip_image_embedding)
        np.save(file=join_paths(dir_path, 'blip_caption_embedding_fixed.npy'), arr=blip_caption_embedding_fixed)
        np.save(file=join_paths(dir_path, 'blip_caption_embedding_variable.npy'), arr=blip_caption_embedding_variable)
        hidden_state_dict = {
            'nullI+nullC' : BLIP_Prior_Tools.concatenate_embeddings(img_emb=null_img_embedding, txt_emb=null_cap_embedding),
            'nullI+blipC' : BLIP_Prior_Tools.concatenate_embeddings(img_emb=null_img_embedding, txt_emb=blip_caption_embedding),
            'blipI+nullC' : BLIP_Prior_Tools.concatenate_embeddings(img_emb=blip_image_embedding, txt_emb=null_cap_embedding),
            'blipI+blipC' : BLIP_Prior_Tools.concatenate_embeddings(img_emb=blip_image_embedding, txt_emb=blip_caption_embedding),
        }
        for tag, hidden_state in hidden_state_dict.items():
            image_path = join_paths(dir_path, f'{tag}.png')
            if not os.path.exists(image_path):
                hidden_state = torch.from_numpy(hidden_state).unsqueeze(0).to(device)
                generated_image = blip_diffusion_model.generate_image_via_embedding(
                                        uncond_embedding=uncond_embedding,
                                        hidden_states=hidden_state,
                                        causal_attention_mask=causal_attention_mask,
                                        seed=configs_dict['blip_diffusion']['iter_seed'],
                                        guidance_scale=configs_dict['blip_diffusion']['guidance_scale'],
                                        height=coco_matrix.shape[0],
                                        width=coco_matrix.shape[1],
                                        num_inference_steps=configs_dict['blip_diffusion']['num_inference_steps']//10,
                                    )
                generated_image.convert('RGB').save(image_path)
    del blip_diffusion_model


image_datapoint = namedtuple('image_datapoint',
                            ['index', 'coco_image', 'input_tensor',
                             'target_embedding', 'strings_json_path'])
caption_datapoint = namedtuple('caption_datapoint',
                               ['index', 'coco_image', 'input_tensor', 'target_embedding',
                                'blip_caption_embedding_fixed', 'strings_json_path'])
class NSD_Dataset(Dataset):
    """
    load proprocessed data from hdf5 file
    """
    def __init__(self, root_dir : str, tower_name : str) -> None:
        super().__init__()
        assert os.path.exists(root_dir), f'{root_dir} does not exist.'
        self.dirs = {int(d) : os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))}
        self.tower_name = tower_name
    
    def __process_masked_fmri__(self, masked_fmri : np.ndarray) -> np.ndarray:
        # early
        masked_fmri = np.clip(masked_fmri, -1564, 2697)
        # masked_fmri += 1564
        # masked_fmri /= (2697+1564) # [0, 1]
        # masked_fmri *= 2 # [0, 2]
        # masked_fmri -= 1 # [-1, 1]
        masked_fmri /= 1564
        # masked_fmri -= np.iinfo(np.int16).min # [-32768, 32767] -> [0, 65535]
        return masked_fmri
    
    def __process_blip_image_embedding__(self, blip_image_embedding : np.ndarray) -> np.ndarray:
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
        fmri.shape = (81, 104, 83), fmri.min = -32768, fmri.max = 32767
        masked_fmri.shape = (k,)
        masked_fmri_embedding.shape = (32, 768)
        '''
        # fmri = np.load(os.path.join(dir_path, 'fmri.npy'), allow_pickle=True)
        masked_fmri = np.load(os.path.join(dir_path, 'masked_fmri.npy'), allow_pickle=True)
        masked_fmri = self.__process_masked_fmri__(masked_fmri)
        # masked_fmri_embedding = np.load(os.path.join(dir_path, 'masked_fmri_embedding.npy'), allow_pickle=True)
        input_tensor = masked_fmri

        '''
        target embedding: BLIP Space
        blip_image_embedding.shape = (16, 768)
        blip_caption_embedding_fixed.shape = (3, 768)
        blip_caption_embedding_variable.shape = (58, 768)
        '''
        if self.tower_name == 'i':
            blip_image_embedding = np.load(os.path.join(dir_path, 'blip_image_embedding.npy'), allow_pickle=True)
            target_embedding = self.__process_blip_image_embedding__(blip_image_embedding)
        elif self.tower_name == 'c':
            blip_caption_embedding_fixed = np.load(os.path.join(dir_path, 'blip_caption_embedding_fixed.npy'), allow_pickle=True)
            blip_caption_embedding_variable = np.load(os.path.join(dir_path, 'blip_caption_embedding_variable.npy'), allow_pickle=True)
            target_embedding = blip_caption_embedding_variable
        
        '''
        strings: json file
        '''
        strings_json_path = os.path.join(dir_path, 'strings.json')
        
        # ndarray -> tensor
        coco_image = torch.tensor(coco_image, dtype=torch.float32)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)                   
        target_embedding = torch.tensor(target_embedding, dtype=torch.float32)  
        if self.tower_name == 'c':
            blip_caption_embedding_fixed = torch.tensor(blip_caption_embedding_fixed, dtype=torch.float32)       # (3, 768)

        # Return data point
        if self.tower_name == 'i':
            return image_datapoint(index, coco_image, input_tensor, 
                                  target_embedding, strings_json_path)
        elif self.tower_name == 'c':
            return caption_datapoint(index, coco_image, input_tensor, target_embedding, 
                                  blip_caption_embedding_fixed, strings_json_path)

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