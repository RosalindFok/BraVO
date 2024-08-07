import time
import torch  
import torch.nn as nn  

__all__ = [
    'device', 
    'load_blip_models',
    'BraVO_Decoder'
]

########################
###### Set divece ######
########################

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



########################
###### Load BLIPs  #####
########################

def load_blip_models(mode : str) -> tuple[nn.Module, dict, dict]:
    from lavis.models import load_model_and_preprocess
    start_time = time.time()
    if mode == 'encoder':
        model, vis_processors, txt_processors = load_model_and_preprocess(
                name='blip2_feature_extractor', # class Blip2Qformer(Blip2Base)
                model_type='coco', 
                is_eval=True, 
                device=device
            )
    elif mode == 'diffusion':
        model, vis_processors, txt_processors = load_model_and_preprocess(
                name='blip_diffusion', # class BlipDiffusion(BaseModel)
                model_type="base", 
                is_eval=True, 
                device=device
            )
    elif mode == 'matching':
        model, vis_processors, txt_processors = load_model_and_preprocess(
                name='blip2_image_text_matching', # class Blip2ITM(Blip2Qformer)
                model_type='coco', 
                is_eval=True,
                device=device 
            )
    else:
        raise ValueError(f"Invalid mode: {mode}.")  
    
    end_time = time.time()
    print(f'It took {end_time - start_time:.2f} seconds to load the BLIP-2 model {mode}.')
    return model, vis_processors, txt_processors



########################
######Brain Encoder#####
########################  

# class BraVO_Encoder(nn.Module):
#     """
#     Map the embedding of image + caption + category, into the whole brain activity.
#     """
#     def __init__(self, input_shape, output_shape):
#         super().__init__()
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         self.activate = nn.ReLU(inplace=True)
#         self.prob = nn.Sigmoid()
#         self.linear = nn.Linear(input_shape[2], output_shape[1]*output_shape[2])
#         self.conv = nn.Conv2d(in_channels=input_shape[0]*input_shape[1], out_channels=output_shape[0], kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         # reshape the input embedding: [batch_size, 2, 77, 768]->[batch_size, 154, 768]->[batch_size*154, 768]
#         x = x.permute(0, 2, 1, 3).reshape(x.size(0), -1, x.size(-1)) 
#         batch_size, dim0, dim1 = x.shape
#         x = x.view(batch_size*dim0, dim1)

#         # Linear layer: [batch_size*154, 768]->[batch_size, 154, 186, 148]
#         x = self.linear(x)
#         x = self.activate(x)
#         output_dim = x.shape[-1]
#         x = x.view(batch_size, dim0, output_dim)
#         x = x.reshape(x.size(0), x.size(1), self.output_shape[1], self.output_shape[2])

#         # Conv layer: [batch_size, 154, 186, 148]->[batch_size, 145, 186, 148]
#         x = self.conv(x)

#         # normalize
#         x = self.prob(x)

#         return x



########################
######Brain Decoder#####
######################## 

class BraVO_Decoder(nn.Module):
    """
    Map the brain activity into the embedding of image or caption.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.prob = nn.Sigmoid()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = x 
        return x