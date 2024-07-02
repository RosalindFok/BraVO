import torch
import numpy as np
import torch.nn as nn
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

# Load BLIP-2 Module
BLIP2_model, vis_processors, txt_processors = load_model_and_preprocess(
                name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device,
                bert_base_uncased_dir_path = bert_base_uncased_dir_path
            )
    # @staticmethod
    # def get_similarity(self, features_image, features_text) -> float:  
    #     similarity = (features_image.image_embeds_proj @ features_text.text_embeds_proj[:,0,:].t()).max()
    #     print(similarity)
    #     # tensor([[0.3642]])


class BraVO_Encoder(nn.Module):
    """
    Map the embedding of image or caption, into the brain activity.
    """
    def __init__(self) -> None:
        super().__init__()

    def brainnetome():
        pass

    def brain_functional_network():
        pass

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = x 
        return x
    
class BraVO_Decoder(nn.Module):
    """
    Map the brain activity into the embedding of image or caption.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = x 
        return x