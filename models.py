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

def get_GPU_memory_usage() -> tuple[float, float]:
    if torch.cuda.is_available():  
        current_device = torch.cuda.current_device()  
        mem_reserved = torch.cuda.memory_reserved(current_device) / (1024 ** 3)    # GB  
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)  # GB  
        return total_memory, mem_reserved

########################
###### Load BLIPs  #####
########################

def load_blip_models(mode : str, device : torch.device = device) -> tuple[nn.Module, dict, dict]:
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

# class Bijection_ND_CDF(nn.Module):
#     """
#     N(miu, sigma) <--> CDF[0, 1] 
#     """    
#     def __init__(self, forward_mean  : float = 0.0, forward_std  : float = 1.0, 
#                        backward_mean : float = 0.0, backward_std : float = 1.0) -> None:
#         super().__init__()
#         self.forward_mean = forward_mean
#         self.forward_std = forward_std
#         self.backward_mean = backward_mean
#         self.backward_std = backward_std

#     # Map the normal distribution to a Cumulative Distribution Function between [0,1]
#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         # Standardize the tensor    
#         x = (x - self.forward_mean) / self.forward_std
#         # Compute the CDF of the standard normal distribution
#         x = 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device))))
#         return x

#     # Map the Cumulative Distribution Function between [0,1] to a normal distribution ~ N(miu, sigma)
#     def backward(self, x : torch.Tensor) -> torch.Tensor:
#         # Inverse CDF (Probit function)  
#         x = torch.distributions.Normal(0, 1).icdf(x)  
#         # Transform to new normal distribution with priori_mean and priori_std  
#         x = self.backward_mean + self.backward_std * x  
#         return x 
    
class BraVO_Decoder(nn.Module):
    """
    """
    def __init__(self, input_shape : torch.Size, output_shape : torch.Size) -> None:
        super().__init__()
        self.input_shape = input_shape[0]
        self.output_shape = output_shape[0]*output_shape[1]
        self.latend_dim = 4096 # 2**12

        self.encoder = nn.Sequential(
            nn.Linear(self.input_shape, self.latend_dim),
            nn.Tanh(),
        )

        self.mean_layer = nn.Linear(self.latend_dim, self.latend_dim//4)
        self.log_var_layer = nn.Linear(self.latend_dim, self.latend_dim//4)

        self.decoder = nn.Sequential(
            nn.Linear(self.latend_dim//4, self.latend_dim),
            nn.Tanh(),
            nn.Linear(self.latend_dim, self.output_shape),
            nn.Tanh(),
        )
        

    def reparameterization(self, mean : torch.Tensor, log_var : torch.Tensor) -> torch.Tensor:
        epsilon = torch.randn_like(log_var).to(mean.device)
        return torch.exp(0.5 * log_var) * epsilon + mean
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        x = self.encoder(x)
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        x = self.reparameterization(mean, log_var)
        x = self.decoder(x)
        return x, mean, log_var  

# class BraVO_Decoder(nn.Module):
#     def __init__(self, input_shape : torch.Size, output_shape : torch.Size) -> None:
#         super().__init__()
#         self.input_shape = input_shape[0]
#         self.output_dim= output_shape[0]
#         self.output_channel = output_shape[-1]
#         self.start_feature_map_size = 256#int(self.input_shape**0.5)

#         self.activate = nn.LeakyReLU(negative_slope=0.1)
#         self.linear = nn.Linear(self.input_shape, self.start_feature_map_size**2)
#         # output_dim=(input_dim−1)×stride−2×padding+dilation×(kernel_size−1)+output_padding+1
#         self.convt = nn.ConvTranspose2d(in_channels=1, out_channels=self.output_channel, 
#                                         kernel_size=4, stride=2, 
#                                         padding=1, #output_padding=1
#                                     )
#         self.upsample = nn.Upsample(size=(self.output_dim, self.output_dim), mode='bicubic', align_corners=False)
#         mid_channel_in_convs = 9
#         self.convs = nn.Sequential(
#             nn.Conv2d(in_channels=self.output_channel, out_channels=mid_channel_in_convs, kernel_size=3, stride=1, padding=1),
#             # nn.BatchNorm2d(num_features=mid_channel_in_convs),
#             self.activate,
#             nn.Conv2d(in_channels=mid_channel_in_convs, out_channels=self.output_channel, kernel_size=3, stride=1, padding=1),
#         )

#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         x = self.linear(x)
#         x = self.activate(x)
#         x = x.view(x.size(0), 1, self.start_feature_map_size, self.start_feature_map_size)
#         x = self.convt(x)
#         x = self.activate(x)
#         x = self.upsample(x)
#         x = self.activate(x)
#         x = self.convs(x)
#         x = x.permute(0, 2, 3, 1)  
#         x = (x-torch.min(x))/(torch.max(x)-torch.min(x)) * 255
#         return x      

