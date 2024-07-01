import torch
import numpy as np
from lavis.models import load_model_and_preprocess

from utils import bert_base_uncased_dir_path

__all__ = ['device', 
           'BLIP2_Tools']

def _setup_device_() -> torch.device:
    """
    Set up and return the available torch device.

    Returns:
        torch.device: A torch.device object representing the device,
        choosing GPU or CPU based on the availability of CUDA.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device = {device if not torch.cuda.is_available() else torch.cuda.get_device_name(torch.cuda.current_device())}')
    torch.cuda.init() if torch.cuda.is_available() else None
    return device

device = _setup_device_()

class BLIP2_Tools():
    def __init__(self) -> None:
        super().__init__()
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device,
                bert_base_uncased_dir_path = bert_base_uncased_dir_path
            )
    
    def multimodal_features(self, image_rgb : np.ndarray, caption : str) -> None:
        image = self.vis_processors["eval"](image_rgb).unsqueeze(0).to(device)
        text = self.txt_processors["eval"](caption) 
        sample = {"image": image, "text_input": [text]}
        features_multimodal = self.model.extract_features(sample)
        print(features_multimodal.multimodal_embeds.shape)
        # torch.Size([1, 32, 768]), 32 is the number of queries

    # @staticmethod
    # def unimodal_features(self, mode : str = None) -> None:
    #     features = self.model.extract_features(self.sample, mode=mode)
        
    #     print(features.image_embeds.shape)
    #     # torch.Size([1, 32, 768])
    #     print(features.text_embeds.shape)
    #     # torch.Size([1, 12, 768])

    # @staticmethod
    # def get_similarity(self, features_image, features_text) -> float:  
    #     similarity = (features_image.image_embeds_proj @ features_text.text_embeds_proj[:,0,:].t()).max()
    #     print(similarity)
    #     # tensor([[0.3642]])
