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

class Bijection_ND_CDF(nn.Module):
    """
    N(miu, sigma) <--> CDF[0, 1] 
    """    
    def __init__(self, forward_mean  : float = 0.0, forward_std  : float = 1.0, 
                       backward_mean : float = 0.0, backward_std : float = 1.0) -> None:
        super().__init__()
        self.forward_mean = forward_mean
        self.forward_std = forward_std
        self.backward_mean = backward_mean
        self.backward_std = backward_std

    # Map the normal distribution to a Cumulative Distribution Function between [0,1]
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Standardize the tensor    
        x = (x - self.forward_mean) / self.forward_std
        # Compute the CDF of the standard normal distribution
        x = 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device))))
        return x

    # Map the Cumulative Distribution Function between [0,1] to a normal distribution ~ N(miu, sigma)
    def backward(self, x : torch.Tensor) -> torch.Tensor:
        # Inverse CDF (Probit function)  
        x = torch.distributions.Normal(0, 1).icdf(x)  
        # Transform to new normal distribution with priori_mean and priori_std  
        x = self.backward_mean + self.backward_std * x  
        return x 
    
class BraVO_Decoder(nn.Module):
    """
    VAE: Map the brain activity into the embedding of image or caption.
    """
    def __init__(self, input_shape : torch.Size, output_shape : torch.Size) -> None:
        super().__init__()

        z_dim = input_shape[0] // 36
        self.output_shape = output_shape
        
        self.Encoder = nn.Sequential(
            nn.Linear(input_shape[0], z_dim*6),
            nn.ReLU(),
        )

        self.Decoder = nn.Sequential(
            nn.Linear(z_dim, z_dim*6),
            nn.ReLU(),
            nn.Linear(z_dim*6, z_dim*36),
            nn.ReLU(),
            nn.Linear(z_dim*36, output_shape[0]*output_shape[1]),
            nn.Sigmoid(),
        )

        self.fc_mean = nn.Linear(z_dim*6, z_dim)
        self.fc_log_var = nn.Linear(z_dim*6, z_dim)
    
    def reparameterization(self, mean : torch.Tensor, log_var : torch.Tensor) -> torch.Tensor:
        epsilon = torch.randn_like(log_var)
        return torch.exp(0.5 * log_var) * epsilon + mean

    def forward(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.Encoder(x)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        x = self.reparameterization(mean=z_mean, log_var=z_log_var)
        x = self.Decoder(x)
        x = x.view(x.size(0), *self.output_shape)
        return x, z_mean, z_log_var
