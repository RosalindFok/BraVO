import time
import torch  
import platform
import numpy as np
import torch.nn as nn 

__all__ = [
    'device', 'get_GPU_memory_usage'
    'load_blip_models',
    'BraVO_Decoder'
]

########################
###### Set divece ######
########################

num_workers = 6 if platform.system() == 'Linux' else 0

def _setup_device_() -> list[torch.device]:
    """
    """
    if torch.cuda.is_available():
        torch.cuda.init()  
        device_count = torch.cuda.device_count()  
        print(f'Number of GPUs available: {device_count}')  
        devices = []  # List to hold all available devices  
        for device_id in range(device_count):  
            device = torch.device(f'cuda:{device_id}')  
            devices.append(device)  
            device_name = torch.cuda.get_device_name(device_id)  
            print(f'Device {device_id}: {device_name}')  
        torch.cuda.set_device(devices[0])
    else:
        devices = [torch.device('cpu')]
        print('Device: CPU, no CUDA device available') 
    return devices  

devices = _setup_device_()
device = devices[0]
devices_num = len(devices)

def get_GPU_memory_usage() -> tuple[float, float]:
    if torch.cuda.is_available():  
        current_device = torch.cuda.current_device()  
        mem_reserved = torch.cuda.memory_reserved(current_device) / (1024 ** 3)    # GB  
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)  # GB  
        return total_memory, mem_reserved

########################
###### Load BLIPs  #####
########################

def load_blip_models(mode : str, device : torch.device = device, is_eval : bool = True) -> tuple[nn.Module, dict, dict]:
    from lavis.models import load_model_and_preprocess
    start_time = time.time()
    if mode == 'feature':
        model, vis_processors, txt_processors = load_model_and_preprocess(
                name='blip2_feature_extractor', # class Blip2Qformer(Blip2Base)
                model_type='coco', 
                is_eval=is_eval, 
                device=device
            )
    elif mode == 'diffusion':
        model, vis_processors, txt_processors = load_model_and_preprocess(
                name='blip_diffusion', # class BlipDiffusion(BaseModel)
                model_type="base", 
                is_eval=is_eval, 
                device=device
            )
    elif mode == 'matching':
        model, vis_processors, txt_processors = load_model_and_preprocess(
                name='blip2_image_text_matching', # class Blip2ITM(Blip2Qformer)
                model_type='coco', 
                is_eval=is_eval,
                device=device 
            )
    elif mode == 'caption':
        model, vis_processors, txt_processors = load_model_and_preprocess(
                name='blip2_t5', # blip2_models.blip2_t5.Blip2T5
                model_type='caption_coco_flant5xl', # pretrain_flant5xl, caption_coco_flant5xl, pretrain_flant5xxl
                is_eval=is_eval, 
                device=device
            )
    else:
        raise ValueError(f'Invalid mode: {mode}.')  
    
    # Multi-GPUs
    # if devices_num > 1:
    #     model = nn.DataParallel(model)
    model = model.module if hasattr(model, 'module') else model
    end_time = time.time()
    print(f'It took {end_time - start_time:.2f} seconds to load the BLIP {mode} model.')
    return model, vis_processors, txt_processors



########################
######Brain Decoder#####
######################## 
# class ScaledTanh(nn.Module):
#     def __init__(self, pos_scale : float = 1.0, neg_scale : float = 1.0) -> None:
#         super().__init__()
#         self.pos_scale = pos_scale
#         self.neg_scale = neg_scale
#         self.tanh = nn.Tanh()

#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         return self.tanh(x) * torch.where(x > 0, self.pos_scale, self.neg_scale)

       
class BraVO_Decoder(nn.Module):
    def __init__(self,  input_shape : torch.Size, 
                        image_embedding_shape : torch.Size,
                        caption_embedding_shape : torch.Size = None,
    ) -> None:
        super().__init__()
        self.image_embedding_shape = image_embedding_shape
        if caption_embedding_shape is None: # blip2
            self.net = nn.Sequential(
                nn.Linear(input_shape[0], input_shape[0]//14),
                nn.Tanh(),
                nn.Linear(input_shape[0]//14, image_embedding_shape[0]*image_embedding_shape[1]),
                nn.Unflatten(dim=1, unflattened_size=image_embedding_shape),
                nn.Softmax(dim=-1)
            )
            # self.softmax = nn.Softmax(dim=-1)
        else: # blipdiffusion
            pass

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x