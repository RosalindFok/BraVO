import time
import torch
import torch.nn as nn
from utils import bert_base_uncased_dir_path
from lavis.models import load_model_and_preprocess

__all__ = [
    'device', 
    'load_blip_models',
    'BraVO_Encoder', 'BraVO_Decoder'
]

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
def load_blip_models(mode : str) -> tuple[torch.nn.Module, dict, dict]:
    start_time = time.time()
    if mode == 'encoder':
        model, vis_processors, txt_processors = load_model_and_preprocess(
                name="blip2_feature_extractor", 
                model_type="coco", # Go to blip2_qformer.py to see model types
                is_eval=True, 
                device=device,
                bert_base_uncased_dir_path = bert_base_uncased_dir_path
            )
    elif mode == 'diffusion':
        model, vis_processors, txt_processors = load_model_and_preprocess(
                name="blip_diffusion", # class BlipDiffusion(BaseModel)
                model_type="base", 
                is_eval=True, 
                device=device,
                bert_base_uncased_dir_path = bert_base_uncased_dir_path
            )
    else:
        raise ValueError(f"Invalid mode: {mode}.")  
    
    end_time = time.time()
    print(f'It took {end_time - start_time:.2f} seconds to load the BLIP-2 model {mode}.')
    return model, vis_processors, txt_processors

class Average_Embedding(nn.Module):
    """
    Input: caption_embedding or multimodal_embedding, whose shape is (batch_size, k, 768), k = number of captions of the corresponding image.
    Output: the average embedding
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # k = 1
        # k >= 1
        x = x.mean(dim=1)

        return x


class MaxSimilarty_Embedding(nn.Module):
    """
    Input: image_embedding, whose shape is (batch_size, 768).
    Input: caption_embedding or multimodal_embedding, whose shape is (batch_size, k, 768), k = number of captions of the corresponding image.
    Output: the embedding with the highest similarity with the image_embedding.
    """
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # k = 1
        # similarity = (features_image.image_embeds_proj @ features_text.text_embeds_proj[:,0,:].t()).max()

        # k >= 1
        
        return x
    

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