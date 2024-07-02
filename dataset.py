import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils import BraVO_saved_dir_path
from utils import join_paths, read_nii_file, read_json_file

class BLIP2_Dataset(Dataset):
    """
    load the image and captions and then process them via BLIP-2
    """
    def __init__(self, image_rgb : np.ndarray, captions_list : list[str], vis_processors : dict, txt_processors : dict, device : torch.device) -> None:
        super().__init__()
        self.images_list = [vis_processors["eval"](Image.fromarray(image_rgb)).to(device)] * len(captions_list)
        self.captions_list = [txt_processors["eval"](caption) for caption in captions_list]
    
    def __getitem__(self, index) -> dict[torch.Tensor, str]:
        return {"image": self.images_list[index], "text_input": self.captions_list[index]}
    
    def __len__(self) -> int:
        assert len(self.images_list) == len(self.captions_list)
        return len(self.images_list)


class NSD_Dataset(Dataset):
    def __init__(self, root_dir : str = BraVO_saved_dir_path, subj_id : int = None, mode : str = None) -> None:
        super().__init__()
        assert 1<= subj_id <= 8, print(f'Invalid subj_id={subj_id}. Please choose from 1 to 8.')
        assert mode in ['train', 'test'], print(f'Invalid mode={mode}. Please choose from "train" or "test".')
        self.dir_path = os.path.join(root_dir, f'subj{str(subj_id).zfill(2)}_pairs', mode)
        self.trials_path_list = [join_paths(self.dir_path, x) for x in os.listdir(self.dir_path)]
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, tuple, torch.Tensor, torch.Tensor, torch.Tensor]:
        trial_path = self.trials_path_list[index]
        fmri_path = join_paths(trial_path, 'fmri.nii.gz')
        image_path = join_paths(trial_path, 'image.png')
        info_dict = read_json_file(join_paths(trial_path, 'info.json'))
        isold = info_dict['ISOLD']
        captions_list = info_dict['caption'] # natural language captions
        embeddings_dict = np.load(join_paths(trial_path, 'embedding.npz'))
        print(type(embeddings_dict))
        for x in embeddings_dict:
            print(x)
        image_embedding = embeddings_dict['image']
        caption_embedding = embeddings_dict['caption']
        multimodal_embedding = embeddings_dict['multimodal']
        fmri_header, fmri_data = read_nii_file(fmri_path)
        image_data = cv2.imread(image_path)
        fmri_data = fmri_data.astype(np.int16)
        return fmri_data, image_data, isold, captions_list, image_embedding, caption_embedding, multimodal_embedding
    
    def __len__(self) -> int:
        return  len(self.trials_path_list)
