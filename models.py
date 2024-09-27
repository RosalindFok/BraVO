import time
import torch  
import platform
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
    elif mode == 'caption':
        model, vis_processors, txt_processors = load_model_and_preprocess(
                name='blip2_t5', # blip2_models.blip2_t5.Blip2T5
                model_type='caption_coco_flant5xl', # pretrain_flant5xl, caption_coco_flant5xl, pretrain_flant5xxl
                is_eval=True, 
                device=device
            )
    else:
        raise ValueError(f'Invalid mode: {mode}.')  
    
    # Multi-GPUs
    if devices_num > 1:
        model = nn.DataParallel(model)
    model = model.module if hasattr(model, 'module') else model
    end_time = time.time()
    print(f'It took {end_time - start_time:.2f} seconds to load the BLIP-2 model {mode}.')
    return model, vis_processors, txt_processors



########################
######Brain Decoder#####
######################## 
class Conv_Twice(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=in_channels , out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1)//2), # stride = 1
            nn.BatchNorm1d(out_channels),
            nn.Hardtanh(min_val=-3, max_val=3),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1)//2), # stride = 1
            nn.BatchNorm1d(out_channels),
            nn.Hardtanh(min_val=-3, max_val=3)
        )
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        return x
    
class Down(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3) -> None:
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.MaxPool1d(2),
            Conv_Twice(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.down_sample(x)  
        return x

class Up(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3) -> None:
        super().__init__()
        self.up_sample = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2)
        self.convs =  Conv_Twice(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        
    def forward(self, x1 : torch.Tensor, x2 : torch.Tensor) -> torch.Tensor:
        x1 = self.up_sample(x1)
        assert x1.shape == x2.shape, f'x1.shape={x1.shape} != x2.shape={x2.shape}.'
        x = torch.cat((x1, x2), dim=1)  
        x = self.convs(x)
        return x

class Caption_Decoder(nn.Module):
    """
    """
    def __init__(self, input_shape : torch.Size, output_shape : torch.Size) -> None:
        super().__init__()
        self.input_layer = nn.Conv1d(in_channels=input_shape[0], out_channels=128, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm1d(128) 
        self.dw1 = Down(in_channels=128, out_channels=256)
        self.dw2 = Down(in_channels=256, out_channels=512)
        self.dw3 = Down(in_channels=512, out_channels=1024)
        self.dw4 = Down(in_channels=1024, out_channels=2048)
        encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8, dim_feedforward=4096, batch_first=True)  
        self.bottleneck = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.up1 = Up(in_channels=2048, out_channels=1024)
        self.up2 = Up(in_channels=1024, out_channels=512)
        self.up3 = Up(in_channels=512, out_channels=256)
        self.up4 = Up(in_channels=256, out_channels=128)
        self.output_layer = nn.Conv1d(in_channels=128, out_channels=output_shape[0], kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        # Encoder
        x1 = self.input_bn(self.input_layer(x))
        x2 = self.dw1(x1)
        x3 = self.dw2(x2)
        x4 = self.dw3(x3)
        x5 = self.dw4(x4)

        # Bottleneck
        x5 = x5.permute(0, 2, 1) # (N, C, L) ->(N, L, C) 
        x5 = self.bottleneck(x5)
        x5 = x5.permute(0, 2, 1) # (N, L, C) ->(N, C, L)

        # Decoder
        y4 = self.up1(x5, x4)
        y3 = self.up2(y4, x3)
        y2 = self.up3(y3, x2)
        y1 = self.up4(y2, x1)

        # Output
        y = self.output_layer(y1)

        return y


class Image_Decoder(nn.Module):
    """
    """
    def __init__(self, input_shape : torch.Size, output_shape : torch.Size) -> None:
        super().__init__()
        self.input_layer = nn.Conv1d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm1d(32) 
        self.dw1 = Down(in_channels=32 , out_channels=64 , kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.dw2 = Down(in_channels=64 , out_channels=128 , kernel_size=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.dw3 = Down(in_channels=128 , out_channels=256, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(256)
        self.dw4 = Down(in_channels=256, out_channels=512, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=output_shape[0], dim_feedforward=1024, batch_first=True)  
        self.bottleneck = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.up1 = Up(in_channels=512, out_channels=256, kernel_size=3)
        self.up2 = Up(in_channels=256, out_channels=128, kernel_size=3)
        self.up3 = Up(in_channels=128, out_channels=64 , kernel_size=3)
        self.up4 = Up(in_channels=64 , out_channels=32 , kernel_size=3)
        self.output_layer = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=output_shape[0], kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        # Encoder
        x1 = self.input_layer(x)
        x2 = self.dw1(x1)
        x2 = self.bn1(x2)
        x3 = self.dw2(x2)
        x3 = self.bn2(x3)
        x4 = self.dw3(x3)
        x4 = self.bn3(x4)
        x5 = self.dw4(x4)
        x5 = self.bn4(x5)

        # Bottleneck
        x5 = x5.permute(0, 2, 1) # (N, C, L) ->(N, L, C) 
        x5 = self.bottleneck(x5)
        x5 = x5.permute(0, 2, 1) # (N, L, C) ->(N, C, L)
        x5 = self.bn4(x5)

        # Decoder
        y4 = self.up1(x5, x4)
        y4 = self.bn3(y4)
        y3 = self.up2(y4, x3)
        y3 = self.bn2(y3)
        y2 = self.up3(y3, x2)
        y2 = self.bn1(y2)
        y1 = self.up4(y2, x1)
        y1 = self.input_bn(y1)

        # Output
        y = self.output_layer(y1)

        return y
