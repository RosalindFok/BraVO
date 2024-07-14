import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils import BraVO_saved_dir_path
from utils import join_paths, read_nii_file

__all__ = ['BLIP2_Dataset', 'NSD_Dataset']

class BLIP2_Dataset(Dataset):
    """
    load the image and captions and then process them via BLIP-2
    """
    def __init__(self, image_rgb : np.ndarray, captions_list : list[str], vis_processors : dict, txt_processors : dict, device : torch.device) -> None:
        super().__init__()
        self.images_list = [vis_processors["eval"](image_rgb).to(device)] * len(captions_list)
        self.captions_list = [txt_processors["eval"](caption) for caption in captions_list]
    
    def __getitem__(self, index) -> dict[torch.Tensor, str]:
        return {"image": self.images_list[index], "text_input": self.captions_list[index]}
    
    def __len__(self) -> int:
        assert len(self.images_list) == len(self.captions_list)
        return len(self.images_list)

def make_paths_dict(subj_id : int, mode : str) -> dict[str, dict[str, str]]:
    """
    Generates a dictionary of file paths for a specific subject and mode.

    Args:
        subj_id (int): Subject identifier, an integer where zero-padding is applied if necessary.
        mode (str): The mode or category under which the data is stored.

    Returns:
        dict[str, dict[str, str]]: A nested dictionary where each key is an index corresponding 
        to a trial and each value is another dictionary containing paths to files ('fmri', 
        'image', 'embedding', and 'strings') associated with that trial.

    Raises:
        AssertionError: If the directory path constructed from the given parameters does not exist.

    Example:
        paths_dict = make_paths_dict(1, 'test')
        print(paths_dict)
        {
            0: {'fmri': 'path/to/fmri.nii.gz', 'image': 'path/to/image.png', 
                'embedding': 'path/to/embedding.npz', 'strings': 'path/to/strings.json'},
            ...
        }
    """
    dir_path = join_paths(BraVO_saved_dir_path, f'subj{str(subj_id).zfill(2)}_pairs', mode)
    assert os.path.exists(dir_path), print(f'dir_path={dir_path} does not exist.')
    trial_paths_list = [join_paths(dir_path, x) for x in os.listdir(dir_path)]
    trial_paths_dict = {}
    for index, trail_path in enumerate(trial_paths_list):
        trial_paths_dict[index] = {
            'fmri' : join_paths(trail_path, 'fmri.nii.gz'),
            'image' : join_paths(trail_path, 'image.png'),
            'embedding' : join_paths(trail_path, 'embedding.npz'),
            'strings' : join_paths(trail_path,'strings.json')
        }
    return trial_paths_dict


class NSD_Dataset(Dataset):
    """
    load proprocessed data
    """
    def __init__(self, trial_paths_dict : dict[str, dict[str, str]]) -> None:
        super().__init__()
        self.trial_paths_dict = trial_paths_dict
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # paths
        fmri_path = self.trial_paths_dict[index]['fmri']
        image_path = self.trial_paths_dict[index]['image']
        embedding_path = self.trial_paths_dict[index]['embedding']

        # data
        fmri_header, fmri_data = read_nii_file(fmri_path)
        image_data =  np.array(Image.open(image_path))
        embeddings_dict = np.load(embedding_path)
        image_embedding = embeddings_dict['image_embedding']
        caption_embedding = embeddings_dict['captions_embedding']
        multimodal_embedding = embeddings_dict['multimodal_embedding']

        return index, fmri_data, image_data, image_embedding, caption_embedding, multimodal_embedding
    
    def __len__(self) -> int:
        return  len(self.trial_paths_dict)
