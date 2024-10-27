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
    if mode == 'feature':
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

class Conv_Twice_1d(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=in_channels , out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1)//2), # stride = 1
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1)//2), # stride = 1
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        return x
    
class Down_1d(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3) -> None:
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.MaxPool1d(2),
            Conv_Twice_1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.down_sample(x)  
        return x

class Up_1d(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3) -> None:
        super().__init__()
        self.up_sample = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2)
        self.convs =  Conv_Twice_1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        
    def forward(self, x1 : torch.Tensor, x2 : torch.Tensor) -> torch.Tensor:
        x1 = self.up_sample(x1)
        assert x1.shape == x2.shape, f'x1.shape={x1.shape} != x2.shape={x2.shape}.'
        x = torch.cat((x1, x2), dim=1)  
        x = self.convs(x)
        return x

class Caption_Decoder(nn.Module):
    """
# text max = 0.16060367, min = -0.104751
    """
    def __init__(self, input_shape : torch.Size, output_shape : torch.Size) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.ConvTranspose1d(in_channels=input_shape[0], out_channels=64, kernel_size=27, padding=13),
            nn.Tanh(),
        )

        self.dw1 = Down_1d(in_channels=64, out_channels=128, kernel_size=27)
        self.dw2 = Down_1d(in_channels=128, out_channels=256, kernel_size=27)
        self.dw3 = Down_1d(in_channels=256, out_channels=512, kernel_size=27)
        self.dw4 = Down_1d(in_channels=512, out_channels=1024, kernel_size=27)
        self.bottleneck = nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(d_model=1024, nhead=16, dim_feedforward=2048, batch_first=True),
                            num_layers=8
                        )
        self.up1 = Up_1d(in_channels=1024, out_channels=512, kernel_size=27)
        self.up2 = Up_1d(in_channels=512, out_channels=256, kernel_size=27)
        self.up3 = Up_1d(in_channels=256, out_channels=128, kernel_size=27)
        self.up4 = Up_1d(in_channels=128, out_channels=64, kernel_size=27)
        self.output_layer = nn.Sequential(
                nn.Conv1d(in_channels=64, out_channels=output_shape[0], kernel_size=27, padding=13),
                nn.Hardtanh(min_val=-0.104751, max_val=0.16060367),
                # nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x1 = self.input_layer(x)  
        # Encoder
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

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class Conv_Twice_2d(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1)//2), # stride = 1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1)//2), # stride = 1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        return x
    
class Down_2d(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3) -> None:
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2),
            Conv_Twice_2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.down_sample(x)  
        return x

class Up_2d(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3) -> None:
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2)
        self.convs =  Conv_Twice_2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        
    def forward(self, x1 : torch.Tensor, x2 : torch.Tensor) -> torch.Tensor:
        x1 = self.up_sample(x1)
        # padding x1
        diff_height = x2.size(2) - x1.size(2)  
        diff_width = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, 
                                (diff_width  // 2, diff_width  - diff_width  // 2, 
                                 diff_height // 2, diff_height - diff_height // 2
                                )
                            )
        assert x1.shape == x2.shape, f'x1.shape={x1.shape}!= x2.shape={x2.shape}.'
        x = torch.cat((x1, x2), dim=1)  
        x = self.convs(x)
        return x

class Image_Decoder(nn.Module):
    """
    """
    def __init__(self, input_shape : torch.Size, output_shape : torch.Size) -> None:
        super().__init__()
        # self.input_layer = nn.Sequential(
        #     nn.Conv1d(in_channels=input_shape[0], out_channels=64, kernel_size=27, padding=13),
        #     # nn.Hardtanh(min_val=-5.607545375823975, max_val=4.3020148277282715),
        #     nn.Tanh()
        # )
        
        # self.dw1 = Down_1d(in_channels=64 , out_channels=128 , kernel_size=27)
        # self.dw2 = Down_1d(in_channels=128 , out_channels=256 , kernel_size=27)
        # self.dw3 = Down_1d(in_channels=256 , out_channels=512, kernel_size=27)
        # self.dw4 = Down_1d(in_channels=512, out_channels=1024, kernel_size=27)
        # self.bottleneck = nn.TransformerEncoder(
        #                     nn.TransformerEncoderLayer(d_model=1024, nhead=16, dim_feedforward=2048, batch_first=True), 
        #                     num_layers=8
        #                 )
        # self.up1 = Up_1d(in_channels=1024, out_channels=512, kernel_size=27)
        # self.up2 = Up_1d(in_channels=512, out_channels=256, kernel_size=27)
        # self.up3 = Up_1d(in_channels=256, out_channels=128 , kernel_size=27)
        # self.up4 = Up_1d(in_channels=128 , out_channels=64 , kernel_size=27)
        # self.output_layer = nn.Sequential(
        #     nn.Conv1d(in_channels=64, out_channels=output_shape[0], kernel_size=27, padding=13),
        #     nn.Hardtanh(min_val=-5.607545375823975, max_val=4.3020148277282715),
        #     # nn.Sigmoid(),
        # )

        self.embedding_layer = nn.Sequential(
            nn.Embedding(num_embeddings=torch.iinfo(torch.uint16).max+1, embedding_dim=output_shape[-1]),
            nn.Tanh(),
            nn.Conv1d(in_channels=input_shape[0], out_channels=2048, kernel_size=27, padding=13),
            nn.Tanh(),
            nn.Conv1d(in_channels=2048, out_channels=output_shape[1], kernel_size=27, padding=13),
        )
        self.to_3d_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=27, padding=13),
            nn.Tanh(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=27, padding=13),
            nn.Tanh(),
            nn.Conv2d(in_channels=8, out_channels=output_shape[0], kernel_size=27, padding=13),
        )

        self.dw1 = Down_2d(in_channels=output_shape[0]*1, out_channels=output_shape[0]*2 , kernel_size=27)
        self.dw2 = Down_2d(in_channels=output_shape[0]*2, out_channels=output_shape[0]*4 , kernel_size=27)
        self.dw3 = Down_2d(in_channels=output_shape[0]*4, out_channels=output_shape[0]*8 , kernel_size=27)
        self.dw4 = Down_2d(in_channels=output_shape[0]*8, out_channels=output_shape[0]*16, kernel_size=27)
        # self.bottleneck = nn.Conv2d(in_channels=output_shape[0]*8, out_channels=output_shape[0]*8, kernel_size=27, padding=13)
        # self.bottleneck = nn.TransformerEncoder(
        #                     nn.TransformerEncoderLayer(d_model=output_shape[0]*8*(int(output_shape[1]/2**3)), nhead=6, dim_feedforward=256, batch_first=True), 
        #                     num_layers=1
        #                 )
        self.up1 = Up_2d(in_channels=output_shape[0]*16, out_channels=output_shape[0]*8, kernel_size=27)
        self.up2 = Up_2d(in_channels=output_shape[0]*8 , out_channels=output_shape[0]*4, kernel_size=27)
        self.up3 = Up_2d(in_channels=output_shape[0]*4 , out_channels=output_shape[0]*2, kernel_size=27)
        self.up4 = Up_2d(in_channels=output_shape[0]*2 , out_channels=output_shape[0]*1, kernel_size=27)
        self.output_layer = nn.Softmax(-1)

        # self.input_layer = nn.Sequential(
        #     nn.Linear(input_shape[0], output_shape[0]*output_shape[1]*output_shape[2]),
        #     nn.ReLU()
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        x = x.int()
        x = self.embedding_layer(x) # (B, 5917) -> (B, 768, 100)
        x = x.unsqueeze(1)          # (B, 768, 100) -> (B, 1, 768, 100)
        x1 = self.to_3d_layer(x)    # (B, 1, 768, 100) -> (B, 16, 768, 100)

        # Encoder
        x2 = self.dw1(x1)
        x3 = self.dw2(x2)
        x4 = self.dw3(x3)
        x5 = self.dw4(x4)

        # Bottleneck
        # (B,C1,C2,L) = x4.shape
        # x4 = x4.view(B, C1*C2, L)
        # x4 = x4.permute(0, 2, 1) # (N, C, L) ->(N, L, C) 
        # x4 = x3 # self.bottleneck(x4)
        # x4 = x4.permute(0, 2, 1) # (N, L, C) ->(N, C, L)
        # x4 = x4.view(B, C1, C2, L)

        # Decoder
        y4 = self.up1(x5, x4)
        y3 = self.up2(y4, x3)
        y2 = self.up3(y3, x2)
        y1 = self.up4(y2, x1)

        # Output
        y = self.output_layer(y1)

        return y

