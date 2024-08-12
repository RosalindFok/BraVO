import time
import torch  
import torch.nn as nn  

__all__ = [
    'device', 'get_GPU_memory_usage'
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

def get_GPU_memory_usage() -> tuple[float, float]:
    if torch.cuda.is_available():  
        current_device = torch.cuda.current_device()  
        mem_reserved = torch.cuda.memory_reserved(current_device) / (1024 ** 3)    # GB  
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)  # GB  
        return total_memory, mem_reserved

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
######Brain Decoder#####
######################## 

# class BraVO_Decoder(nn.Module):
#     """
#     Map the brain activity into the embedding of image or caption.
#     """
#     def __init__(self, input_shape : torch.Size, output_shape : torch.Size) -> None:
#         super().__init__()
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         self.linear = nn.Linear(input_shape[0], output_shape[0]*output_shape[1])
#         self.conv_1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
#         self.conv_2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
#         self.activate = nn.ReLU(inplace=True)

#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         x = self.linear(x)
#         x = self.activate(x)
#         x = x.view(x.size(0), *self.output_shape)
#         x = x.unsqueeze(1)
#         x = self.conv_1(x)
#         x = self.activate(x)
#         x = self.conv_2(x)
#         x = self.activate(x)
#         x = x.squeeze()
#         return x
    
class BraVO_Decoder(nn.Module):
    """
    VAE: Map the brain activity into the embedding of image or caption.
    """
    def __init__(self, input_shape : torch.Size, output_shape : torch.Size, 
                 input_mean : float, input_std: float,
                 priori_mean : float, priori_std: float
                 ) -> None:
        super().__init__()
        self.input_mean = input_mean
        self.input_std = input_std
        self.priori_mean = priori_mean
        self.priori_std = priori_std

        z_dim = input_shape[0] // 36
        self.output_shape = output_shape
        
        self.Encoder = nn.Sequential(
            nn.Linear(input_shape[0], z_dim*6),
            nn.Tanh(),
        )

        self.Decoder = nn.Sequential(
            nn.Linear(z_dim, z_dim*6),
            nn.Tanh(),
            nn.Linear(z_dim*6, z_dim*36),
            nn.Tanh(),
            nn.Linear(z_dim*36, output_shape[0]*output_shape[1]),
            nn.Sigmoid(),
        )

        self.fc_mean = nn.Linear(z_dim*6, z_dim)
        self.fc_log_var = nn.Linear(z_dim*6, z_dim)

    def forward(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = (x - self.input_mean) / self.input_std

        x = self.Encoder(x)

        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)

        # reparameterization
        epsilon = torch.randn_like(z_log_var)
        x = torch.exp(0.5 * z_log_var) * epsilon + z_mean

        x = self.Decoder(x)

        mean_x = torch.mean(x, dim=1, keepdim=True)
        std_x = torch.std(x, dim=1, keepdim=True)
        x = ((x - mean_x) / std_x) * self.priori_std + self.priori_mean

        x = x.view(x.size(0), *self.output_shape)

        return x, z_mean, z_log_var
