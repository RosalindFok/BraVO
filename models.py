import torch
import numpy as np
from PIL import Image
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
    '''
    lavis-BLIP2: https://github.com/salesforce/LAVIS/tree/main/projects/blip2
    '''
    @staticmethod
    def blip2_encoder(mode : str = None, image_rgb : np.ndarray = None, caption : str = None) -> torch.Tensor:
        assert mode in ['multimodal', 'm', 'image', 'i', 'text', 't'], print(f'Invalid mode: {mode}. Please choose from [multimodal, image, text, m, i, t].') 
        # Load BLIP-2
        model, vis_processors, txt_processors = load_model_and_preprocess(
                name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device,
                bert_base_uncased_dir_path = bert_base_uncased_dir_path
            )
        image = vis_processors["eval"](Image.fromarray(image_rgb)).unsqueeze(0).to(device) if image_rgb is not None else None
        text = txt_processors["eval"](caption) if caption is not None else None
        sample = {"image": image, "text_input": [text]}
        
        if mode in ['multimodal', 'm']:
            assert sample["image"] is not None and sample["text_input"] is not None, print(f'Please provide both image and text inputs.')
            embedding = model.extract_features(sample).multimodal_embeds
        elif mode in ['image', 'i']:
            assert sample["image"] is not None, print(f'Please provide an image input.')
            embedding = model.extract_features(sample, mode='image').image_embeds
        elif mode in ['text', 't']:
            assert sample["text_input"] is not None, print(f'Please provide a text input.')
            embedding = model.extract_features(sample, mode='text').text_embeds
        else:
            raise ValueError(f'Invalid mode: {mode}. Please choose from [multimodal, image, text, m, i, t].') 
        
        embedding = torch.squeeze(embedding).mean(dim=0)
        return embedding # dim=768

    # @staticmethod
    # def get_similarity(self, features_image, features_text) -> float:  
    #     similarity = (features_image.image_embeds_proj @ features_text.text_embeds_proj[:,0,:].t()).max()
    #     print(similarity)
    #     # tensor([[0.3642]])
