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
    
# class Conv_Twice_1d(nn.Module):
#     def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3) -> None:
#         super().__init__()
#         self.convs = nn.Sequential(
#             nn.Conv1d(in_channels=in_channels , out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1)//2), # stride = 1
#             nn.BatchNorm1d(out_channels),
#             nn.Tanh(),
#             nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1)//2), # stride = 1
#             nn.BatchNorm1d(out_channels),
#             nn.Tanh(),
#         )

#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         x = self.convs(x)
#         return x
    
# class Down_1d(nn.Module):
#     def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3) -> None:
#         super().__init__()
#         self.down_sample = nn.Sequential(
#             nn.MaxPool1d(2),
#             Conv_Twice_1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
#         )

#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         x = self.down_sample(x)  
#         return x

# class Up_1d(nn.Module):
#     def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3) -> None:
#         super().__init__()
#         self.up_sample = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2)
#         self.convs =  Conv_Twice_1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        
#     def forward(self, x1 : torch.Tensor, x2 : torch.Tensor) -> torch.Tensor:
#         x1 = self.up_sample(x1)
#         assert x1.shape == x2.shape, f'x1.shape={x1.shape} != x2.shape={x2.shape}.'
#         x = torch.cat((x1, x2), dim=1)  
#         x = self.convs(x)
#         return x

# class Caption_Decoder(nn.Module):
#     """
# # text max = 0.16060367, min = -0.104751
#     """
#     def __init__(self, input_shape : torch.Size, output_shape : torch.Size) -> None:
#         super().__init__()
#         self.input_layer = nn.Sequential(
#             nn.ConvTranspose1d(in_channels=input_shape[0], out_channels=64, kernel_size=27, padding=13),
#             nn.Tanh(),
#         )

#         self.dw1 = Down_1d(in_channels=64, out_channels=128, kernel_size=27)
#         self.dw2 = Down_1d(in_channels=128, out_channels=256, kernel_size=27)
#         self.dw3 = Down_1d(in_channels=256, out_channels=512, kernel_size=27)
#         self.dw4 = Down_1d(in_channels=512, out_channels=1024, kernel_size=27)
#         self.bottleneck = nn.TransformerEncoder(
#                             nn.TransformerEncoderLayer(d_model=1024, nhead=16, dim_feedforward=2048, batch_first=True),
#                             num_layers=8
#                         )
#         self.up1 = Up_1d(in_channels=1024, out_channels=512, kernel_size=27)
#         self.up2 = Up_1d(in_channels=512, out_channels=256, kernel_size=27)
#         self.up3 = Up_1d(in_channels=256, out_channels=128, kernel_size=27)
#         self.up4 = Up_1d(in_channels=128, out_channels=64, kernel_size=27)
#         self.output_layer = nn.Sequential(
#                 nn.Conv1d(in_channels=64, out_channels=output_shape[0], kernel_size=27, padding=13),
#                 # nn.Hardtanh(min_val=-0.104751, max_val=0.16060367),
#                 nn.Tanh(),
#             )

#     def forward(self, x: torch.Tensor) -> torch.Tensor: 
#         x1 = self.input_layer(x)  
#         # Encoder
#         x2 = self.dw1(x1)
#         x3 = self.dw2(x2)
#         x4 = self.dw3(x3)
#         x5 = self.dw4(x4)

#         # Bottleneck
#         x5 = x5.permute(0, 2, 1) # (N, C, L) ->(N, L, C) 
#         x5 = self.bottleneck(x5)
#         x5 = x5.permute(0, 2, 1) # (N, L, C) ->(N, C, L)

#         # Decoder
#         y4 = self.up1(x5, x4)
#         y3 = self.up2(y4, x3)
#         y2 = self.up3(y3, x2)
#         y1 = self.up4(y2, x1)

#         # Output
#         y = self.output_layer(y1)
#         return y

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# class Conv_Twice_2d(nn.Module):
#     def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3) -> None:
#         super().__init__()
#         self.convs = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels , out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1)//2), # stride = 1
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1)//2), # stride = 1
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#         )

#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         x = self.convs(x)
#         return x
    
# class Down_2d(nn.Module):
#     def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3) -> None:
#         super().__init__()
#         self.down_sample = nn.Sequential(
#             nn.MaxPool2d(2),
#             Conv_Twice_2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
#         )

#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         x = self.down_sample(x)  
#         return x

# class Up_2d(nn.Module):
#     def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3) -> None:
#         super().__init__()
#         self.up_sample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2)
#         self.convs =  Conv_Twice_2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        
#     def forward(self, x1 : torch.Tensor, x2 : torch.Tensor) -> torch.Tensor:
#         x1 = self.up_sample(x1)
#         # padding x1
#         diff_height = x2.size(2) - x1.size(2)  
#         diff_width = x2.size(3) - x1.size(3)
#         x1 = nn.functional.pad(x1, 
#                                 (diff_width  // 2, diff_width  - diff_width  // 2, 
#                                  diff_height // 2, diff_height - diff_height // 2
#                                 )
#                             )
#         assert x1.shape == x2.shape, f'x1.shape={x1.shape}!= x2.shape={x2.shape}.'
#         x = torch.cat((x1, x2), dim=1)  
#         x = self.convs(x)
#         return x

# class Image_Decoder(nn.Module):
#     """
#     """
#     def __init__(self, input_shape : torch.Size, output_shape : torch.Size) -> None:
#         super().__init__()

#         self.embedding_layer = nn.Sequential(
#             nn.Embedding(num_embeddings=torch.iinfo(torch.uint16).max+1, embedding_dim=output_shape[-1]),
#             nn.Tanh(),
#             nn.Conv1d(in_channels=input_shape[0], out_channels=2048, kernel_size=27, padding=13),
#             nn.Tanh(),
#             nn.Conv1d(in_channels=2048, out_channels=output_shape[1], kernel_size=27, padding=13),
#             nn.Sigmoid()
#         )
#         self.to_3d_layer = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=4, kernel_size=27, padding=13),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=4, out_channels=8, kernel_size=27, padding=13),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8, out_channels=output_shape[0], kernel_size=27, padding=13),
#         )

#         self.up_down = nn.Sequential(
#             nn.Conv2d(in_channels=output_shape[0]*1, out_channels=output_shape[0]*2, kernel_size=27, padding=13),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=output_shape[0]*2, out_channels=output_shape[0]*4, kernel_size=27, padding=13),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=output_shape[0]*4, out_channels=output_shape[0]*8, kernel_size=27, padding=13),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=output_shape[0]*8, out_channels=output_shape[0]*4, kernel_size=27, padding=13),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=output_shape[0]*4, out_channels=output_shape[0]*2, kernel_size=27, padding=13),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=output_shape[0]*2, out_channels=output_shape[0]*1, kernel_size=27, padding=13),
#         )
#         # self.dw1 = Down_2d(in_channels=output_shape[0]*1, out_channels=output_shape[0]*2 , kernel_size=27)
#         # self.dw2 = Down_2d(in_channels=output_shape[0]*2, out_channels=output_shape[0]*4 , kernel_size=27)
#         # self.dw3 = Down_2d(in_channels=output_shape[0]*4, out_channels=output_shape[0]*8 , kernel_size=27)
#         # self.dw4 = Down_2d(in_channels=output_shape[0]*8, out_channels=output_shape[0]*16, kernel_size=27)

#         # self.up1 = Up_2d(in_channels=output_shape[0]*16, out_channels=output_shape[0]*8, kernel_size=27)
#         # self.up2 = Up_2d(in_channels=output_shape[0]*8 , out_channels=output_shape[0]*4, kernel_size=27)
#         # self.up3 = Up_2d(in_channels=output_shape[0]*4 , out_channels=output_shape[0]*2, kernel_size=27)
#         # self.up4 = Up_2d(in_channels=output_shape[0]*2 , out_channels=output_shape[0]*1, kernel_size=27)
#         self.output_layer = nn.Softmax(-1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:  
#         x = x.int()
#         x = self.embedding_layer(x) # (B, 5917) -> (B, 768, 100)
#         x = x.unsqueeze(1)          # (B, 768, 100) -> (B, 1, 768, 100)
#         x1 = self.to_3d_layer(x)    # (B, 1, 768, 100) -> (B, 16, 768, 100)

#         # # Encoder
#         # x2 = self.dw1(x1)
#         # x3 = self.dw2(x2)
#         # x4 = self.dw3(x3)
#         # x5 = self.dw4(x4)

#         # # Decoder
#         # y4 = self.up1(x5, x4)
#         # y3 = self.up2(y4, x3)
#         # y2 = self.up3(y3, x2)
#         # y1 = self.up4(y2, x1)

#         # # Output
#         # y = self.output_layer(y1)
#         y = self.up_down(x1)
#         y = self.output_layer(y)
#         return y

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# def get_attn_pad_mask(seq_q, seq_k):
#     '''
#     Padding, because of unequal in source_len and target_len.

#     parameters:
#     seq_q: [batch, seq_len]
#     seq_k: [batch, seq_len]

#     return:
#     mask: [batch, len_q, len_k]


#     '''
#     batch, len_q = seq_q.size()
#     batch, len_k = seq_k.size()
#     # # we define index of PAD is 0, if tensor equals (zero) PAD tokens
#     # pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # [batch, 1, len_k]
#     # No need to pad: all False
#     pad_attn_mask = torch.zeros_like(seq_k, dtype=torch.bool).unsqueeze(1)

#     return pad_attn_mask.expand(batch, len_q, len_k) # [batch, len_q, len_k]

# class PositionalEncoding(nn.Module):
#     def __init__(self, max_len : int, d_model : int, p_drop : float = 0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(p=p_drop)
      
#         positional_encoding = torch.zeros(max_len, d_model) # [max_len, d_model]
#         position = torch.arange(0, max_len).float().unsqueeze(1) # [max_len, 1]

#         div_term = torch.exp(torch.arange(0, d_model, 2).float() *
#                              (-torch.log(torch.Tensor([10000])) / d_model)) # [max_len / 2]

#         positional_encoding[:, 0::2] = torch.sin(position * div_term) # even
#         positional_encoding[:, 1::2] = torch.cos(position * div_term) # odd

#         # [max_len, d_model] -> [1, max_len, d_model] -> [max_len, 1, d_model]
#         positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)

#         # register pe to buffer and require no grads
#         self.register_buffer('pe', positional_encoding)

#     def forward(self, x):
#         # x: [seq_len, batch, d_model]
#         # we can add positional encoding to x directly, and ignore other dimension
#         x = x + self.pe[:x.size(0), ...]
#         return self.dropout(x)

# class ScaledDotProductAttention(nn.Module):
#     def __init__(self, d_k : int):
#         super().__init__()
#         self.d_k = torch.tensor(d_k)

#     def forward(self, Q, K, V, attn_mask):
#         '''
#         Q: [batch, n_heads, len_q, d_k]
#         K: [batch, n_heads, len_k, d_k]
#         V: [batch, n_heads, len_v, d_v]
#         attn_mask: [batch, n_heads, seq_len, seq_len]
#         '''
#         scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(self.d_k) # [batch, n_heads, len_q, len_k]
#         scores.masked_fill_(attn_mask, -1e9)

#         attn = nn.Softmax(dim=-1)(scores) # [batch, n_heads, len_q, len_k]
#         prob = torch.matmul(attn, V) # [batch, n_heads, len_q, d_v]
#         return prob, attn
  
# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model : int, n_heads : int, d_k : int = 64, d_v : int = 64):
#         super().__init__()
#         # do not use more instance to implement multihead attention
#         # it can be complete in one matrix
#         self.n_heads = n_heads
#         self.d_k = d_k
#         self.d_v = d_v

#         # we can't use bias because there is no bias term in formular
#         self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
#         self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
#         self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
#         self.fc = nn.Linear(d_v * n_heads, d_model, bias=False)
#         self.layer_norm = nn.LayerNorm(d_model)

#     def forward(self, input_Q, input_K, input_V, attn_mask):
#         '''
#         To make sure multihead attention can be used both in encoder and decoder,
#         we use Q, K, V respectively.
#         input_Q: [batch, len_q, d_model]
#         input_K: [batch, len_k, d_model]
#         input_V: [batch, len_v, d_model]
#         '''
#         residual, batch = input_Q, input_Q.size(0)

#         # [batch, len_q, d_model] -- matmul W_Q --> [batch, len_q, d_q * n_heads] -- view -->
#         # [batch, len_q, n_heads, d_k,] -- transpose --> [batch, n_heads, len_q, d_k]

#         Q = self.W_Q(input_Q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch, n_heads, len_q, d_k]
#         K = self.W_K(input_K).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch, n_heads, len_k, d_k]
#         V = self.W_V(input_V).view(batch, -1, self.n_heads, self.d_v).transpose(1, 2) # [batch, n_heads, len_v, d_v]

#         attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # [batch, n_heads, seq_len, seq_len]

#         # prob: [batch, n_heads, len_q, d_v] attn: [batch, n_heads, len_q, len_k]
#         prob, attn = ScaledDotProductAttention(d_k=self.d_k)(Q, K, V, attn_mask)

#         prob = prob.transpose(1, 2).contiguous() # [batch, len_q, n_heads, d_v]
#         prob = prob.view(batch, -1, self.n_heads * self.d_v).contiguous() # [batch, len_q, n_heads * d_v]

#         output = self.fc(prob) # [batch, len_q, d_model]

#         return self.layer_norm(residual + output), attn

# class FeedForwardNetwork(nn.Module):
#     '''
#     Using nn.Conv1d replace nn.Linear to implements FFN.
#     '''
#     def __init__(self, d_model : int, p_drop : float = 0.1):
#         super().__init__()
#         # self.ff1 = nn.Linear(d_model, d_ff)
#         # self.ff2 = nn.Linear(d_ff, d_model)
#         d_ff = d_model * 4
#         self.ff1 = nn.Conv1d(d_model, d_ff, 1)
#         self.ff2 = nn.Conv1d(d_ff, d_model, 1)
#         self.relu = nn.ReLU()

#         self.dropout = nn.Dropout(p=p_drop)
#         self.layer_norm = nn.LayerNorm(d_model)

#     def forward(self, x):
#         # x: [batch, seq_len, d_model]
#         residual = x
#         x = x.transpose(1, 2) # [batch, d_model, seq_len]
#         x = self.ff1(x)
#         x = self.relu(x)
#         x = self.ff2(x)
#         x = x.transpose(1, 2) # [batch, seq_len, d_model]
#         return self.layer_norm(residual + x)
    
# class EncoderLayer(nn.Module):
#     def __init__(self, d_model : int, n_heads : int):
#         super().__init__()
#         self.encoder_self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
#         self.ffn = FeedForwardNetwork(d_model=d_model)

#     def forward(self, encoder_input, encoder_pad_mask):
#         '''
#         encoder_input: [batch, source_len, d_model]
#         encoder_pad_mask: [batch, n_heads, source_len, source_len]

#         encoder_output: [batch, source_len, d_model]
#         attn: [batch, n_heads, source_len, source_len]
#         '''
#         encoder_output, attn = self.encoder_self_attn(encoder_input, encoder_input, encoder_input, encoder_pad_mask)
#         encoder_output = self.ffn(encoder_output) # [batch, source_len, d_model]

#         return encoder_output, attn

# class Encoder(nn.Module):
#     """
#     """
#     def __init__(self, input_vocabulary_size : int, max_len : int, 
#                  d_model : int, n_heads : int, n_layers : int) -> None:
#         super().__init__()
#         self.input_vocabulary_size = input_vocabulary_size
#         self.d_model = d_model
#         self.embedding_layer = nn.Embedding(num_embeddings=input_vocabulary_size, embedding_dim=d_model)
#         self.positional_embedding = PositionalEncoding(max_len=max_len, d_model=d_model)
#         self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, n_heads=n_heads) for layer in range(n_layers)])
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         embedding = self.embedding_layer(x) # (B, seq_len) -> (B, seq_len, d_model)
#         embedding = self.positional_embedding(embedding.transpose(0, 1)).transpose(0, 1) # (B, seq_len, d_model)
#         encoder_self_attn_mask = get_attn_pad_mask(x, x) # [batch, source_len, source_len]
#         encoder_self_attns = []
#         for layer in self.layers:
#             # encoder_output: [batch, source_len, d_model]
#             # encoder_self_attn: [batch, n_heads, source_len, source_len]
#             encoder_output, attn = layer(embedding, encoder_self_attn_mask) 
#             encoder_self_attns.append(attn)
#         # encoder_output: [batch, source_len, d_model]
#         # encoder_self_attns: [n_layers, batch, n_heads, source_len, source_len]
#         return encoder_output, encoder_self_attns
    
# class Image_Decoder(nn.Module):
#     """
#     """
#     def __init__(self, input_shape : torch.Size, output_shape : torch.Size,
#                  d_model : int = 16, n_heads : int = 8, n_layers : int = 6) -> None:
#         super().__init__()
#         input_vocabulary_size = torch.iinfo(torch.uint16).max+1
#         output_vocabulary_size = output_shape[-1]
#         self.encoder = Encoder(input_vocabulary_size=input_vocabulary_size, max_len=input_shape[0], d_model=d_model, n_heads=n_heads, n_layers=n_layers)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         encoder_output, _ = self.encoder(x)
#         return encoder_output

# class PositionalEncoding(nn.Module):  
#     def __init__(self, d_model : int, max_len : int, dropout : float = 0.1):  
#         super().__init__()  
#         self.dropout = nn.Dropout(p=dropout)  
        
#         pe = torch.zeros(max_len, d_model)  # (max_len, d_model)  
#         position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)  
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))  # (d_model/2)  
        
#         pe[:, 0::2] = torch.sin(position * div_term)  # even
#         pe[:, 1::2] = torch.cos(position * div_term)  # odd  
#         pe = pe.unsqueeze(0)  # (1, max_len, d_model)  
#         self.register_buffer('pe', pe)  
        
#     def forward(self, x):  
#         x = x + self.pe[:, :x.size(1), :]  
#         return self.dropout(x) 
    
# class Image_Decoder(nn.Module):
#     def __init__(self, input_shape : torch.Size, output_shape : torch.Size, 
#                  d_model : int = 16, nhead : int = 4, num_encoder_layers : int = 6) -> None:
#         super().__init__()
#         input_vocabulary_size = torch.iinfo(torch.uint16).max+1
#         self.output_vocabulary_size = 43 #output_shape[-1]
#         self.d_model = d_model  
#         self.output_shape = output_shape
#         self.input_embedding = nn.Embedding(input_vocabulary_size, d_model)  
#         self.positional_encoding_input = PositionalEncoding(d_model, max_len=input_shape[0])  
        
#         self.transformer_encoder = nn.TransformerEncoder(  
#             nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True),  
#             num_layers=num_encoder_layers  
#         )

#         self.fc_out = nn.Linear(d_model, self.output_vocabulary_size)  
#         self.output_embedding = nn.Embedding(self.output_vocabulary_size, d_model)  
#         self.output_seq_len = output_shape[0]*output_shape[1]  
        
#         self.output_vocab_embeddings = nn.Parameter(torch.randn(self.output_vocabulary_size, output_shape[1])) 
        
#         # self.softmax = nn.Softmax(dim=-1)
#         self.output_layer = nn.Sequential(
#             nn.Conv2d(in_channels=output_shape[0], out_channels=output_shape[0]*2, kernel_size=9, padding=4),
#             nn.Tanh(),
#             nn.Conv2d(in_channels=output_shape[0]*2, out_channels=output_shape[0]*4, kernel_size=9, padding=4),
#             nn.Tanh(),
#             nn.Conv2d(in_channels=output_shape[0]*4, out_channels=output_shape[0]*2, kernel_size=9, padding=4),
#             nn.Tanh(),
#             nn.Conv2d(in_channels=output_shape[0]*2, out_channels=output_shape[0], kernel_size=9, padding=4),
#             nn.Softmax(dim=-1)
#         )
#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         x = self.input_embedding(x)           # (B, input_seq_len) -> (B, input_seq_len, d_model)
#         x = self.positional_encoding_input(x) # (B, input_seq_len, d_model)
#         x = self.transformer_encoder(x)       # (B, input_seq_len, d_model)
#         # 简单地将编码器的输出平均池化，得到句子的表示  
#         encoded = torch.mean(x, dim=1) # (B, d_model)
#         encoded = encoded.unsqueeze(1).repeat(1, self.output_seq_len, 1) # (B, output_seq_len, d_model)
#         # 映射到输出词汇表  
#         logits = self.fc_out(encoded) # (B, output_seq_len, output_vocabulary_size)
#         logits = logits.view(-1, *self.output_shape, self.output_vocabulary_size)

#         probs = self.output_layer(logits)
#         _, predicted_tokens = torch.max(probs, dim=-1)
#         predicted_tokens = predicted_tokens.to(torch.float32)
#         return predicted_tokens

class ScaledTanh(nn.Module):
    def __init__(self, scale : float) -> None:
        super().__init__()
        self.scale = scale
        self.tanh = nn.Tanh()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.scale * self.tanh(x)

# class Caption_Decoder(nn.Module):
#     def __init__(self, input_shape : torch.Size, output_shape : torch.Size) -> None:
#         super().__init__()
#         self.input_embedding = nn.Embedding(torch.iinfo(torch.uint16).max+1, output_shape[1])
#         self.convs = nn.Sequential(
#             nn.Conv1d(in_channels=input_shape[0], out_channels=4096, kernel_size=27, padding=13),
#             ScaledTanh(scale=0.6),
#             nn.Conv1d(in_channels=4096, out_channels=2048, kernel_size=27, padding=13),
#             ScaledTanh(scale=0.6),
#             nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=27, padding=13),
#             ScaledTanh(scale=0.6),
#             nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=27, padding=13),
#             ScaledTanh(scale=0.6),
#             nn.Conv1d(in_channels=512, out_channels=output_shape[0], kernel_size=27, padding=13)
#         )
#         self.clip = nn.Hardtanh(min_val=-0.104751, max_val=0.16060367)

#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         x = self.input_embedding(x)
#         x = self.convs(x)
#         x = self.clip(x)
#         return x

# class Caption_Decoder(nn.Module):
#     def __init__(self, input_shape : torch.Size, output_shape : torch.Size) -> None:
#         super().__init__()
#         self.output_shape = output_shape
#         self.mlp = nn.Linear(in_features=input_shape[0], out_features=output_shape[0]*output_shape[1])
#         self.clip = nn.Hardtanh(min_val=-0.104751, max_val=0.16060367)

#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         x = x.float()
#         x /= torch.iinfo(torch.uint16).max
#         x /= 5
#         x -= 0.1 # [-0.1, 0.1]
#         x = self.mlp(x)
#         x = self.clip(x)
#         x = x.view(-1, *self.output_shape)
#         return x
    
# class Image_Decoder(nn.Module):
#     def __init__(self, input_shape : torch.Size, output_shape : torch.Size):
#         super().__init__()
#         self.input_embedding = nn.Embedding(torch.iinfo(torch.uint16).max+1, output_shape[1])
#         self.convs = nn.Sequential(
#             nn.Conv1d(in_channels=input_shape[0], out_channels=1024, kernel_size=27, padding=13),
#             ScaledTanh(scale=2.0),
#             nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=27, padding=13),
#             ScaledTanh(scale=2.0),
#             nn.Conv1d(in_channels=128, out_channels=output_shape[0], kernel_size=27, padding=13),
#         )
#         self.clip = nn.Hardtanh(-2.1, 2.1) 

#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         x = self.input_embedding(x)
#         x = self.convs(x)
#         x = self.clip(x)
#         return x

# class Image_Decoder(nn.Module):
#     def __init__(self, input_shape : torch.Size, output_shape : torch.Size) -> None:
#         super().__init__()
#         self.output_shape = output_shape
#         self.mlp = nn.Linear(in_features=input_shape[0], out_features=output_shape[0]*output_shape[1])
#         self.clip = nn.Hardtanh(min_val=-2.1, max_val=2.1)
#         # self.softmax = nn.Softmax(dim=-1)
    
#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         x = x.float()
#         x /= torch.iinfo(torch.uint16).max
#         x *= 5 
#         x -= 2.5  # [-2.5, 2.5]
#         x = self.mlp(x)
#         x = self.clip(x)
#         x = x.view(-1, *self.output_shape)
#         return x

# class BraVO_Decoder(nn.Module):
#     def __init__(self, input_shape : torch.Size, output_shape : torch.Size, tower_name : str) -> None:
#         super().__init__()
#         self.output_shape = output_shape
#         self.linear = nn.Linear(in_features=input_shape[0], 
#                                 out_features=output_shape[0]*output_shape[1])
#         if tower_name == 'i':
#             # self.clip = nn.Hardtanh(min_val=-2.1, max_val=2.1)
#             self.clip = nn.Sigmoid()

#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         # x *= 3 # [-3, 3]
#         x = self.linear(x)
#         x = self.clip(x)
#         x = x.view(-1, *self.output_shape)
#         return x

''' input blip_embedding + fMRI; output image'''
class BraVO_Decoder(nn.Module):
    def __init__(self, input_shape : torch.Size, target_embedding_shape : torch.Size, 
                 target_image_shape : torch.Size,
                 uncond_embedding : torch.Tensor,
                 position_embeddings : torch.Tensor,
                 causal_attention_mask : torch.Tensor,
                 caption_embedding_fixed : torch.Tensor,
                 sets : dict[str, any]
                 ) -> None:
        super().__init__()
        self.target_embedding_shape = target_embedding_shape
        self.image_shape = target_image_shape

        self.uncond_embedding = uncond_embedding
        self.position_embeddings = position_embeddings
        self.causal_attention_mask = causal_attention_mask
        self.caption_embedding_fixed = caption_embedding_fixed
        self.sets = sets
        
        self.linear = nn.Linear(in_features=input_shape[0], 
                                out_features=self.target_embedding_shape[0]*self.target_embedding_shape[1])
        self.scaled_tanh = ScaledTanh(scale=2.1)

        self.blip_diffusion_model, _, _ = load_blip_models(mode='diffusion', is_eval=False)
        # self.blip_diffusion_model.eval()
        # for param in self.blip_diffusion_model.parameters():  
            # param.requires_grad = False  
    
    def __concat_caption_embedding__(self, fixed_tensor : torch.Tensor, variable_tensor : torch.Tensor) -> torch.Tensor:
        prefix = fixed_tensor[:2].unsqueeze(0).expand(variable_tensor.size(0), -1, -1)   
        suffix = fixed_tensor[2:].unsqueeze(0).expand(variable_tensor.size(0), -1, -1)  
        caption_embedding = torch.cat([prefix, variable_tensor, suffix], dim=1)
        return caption_embedding
    
    def __concat_hidden_states__(self, image_tensor : torch.Tensor, caption_tensor : torch.Tensor) -> torch.Tensor:
        hidden_states = torch.cat([caption_tensor[:,:2,:], image_tensor, caption_tensor[:,2:,:]], dim=1)
        hidden_states += self.position_embeddings.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
        return hidden_states
    
        
    def forward(self, masked_fmri : torch.Tensor, prompt_embedding : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # predicted_embedding : image, prompt_embedding : caption variable
        predicted_embedding = self.scaled_tanh(self.linear(masked_fmri))
        image_embedding   = predicted_embedding.view(-1, *self.target_embedding_shape)
        caption_embedding = self.__concat_caption_embedding__(fixed_tensor=self.caption_embedding_fixed, variable_tensor=prompt_embedding)
        hidden_states = self.__concat_hidden_states__(image_tensor=image_embedding, caption_tensor=caption_embedding)
        images = self.blip_diffusion_model.generate_image_array_via_embedding_trainable(
                                uncond_embedding=self.uncond_embedding,
                                hidden_states=hidden_states,
                                causal_attention_mask=self.causal_attention_mask,
                                seed=self.sets['iter_seed'],
                                guidance_scale=self.sets['guidance_scale'],
                                height=self.image_shape[0],
                                width=self.image_shape[1],
                                num_inference_steps=self.sets['num_inference_steps']//10,
                            )
        images = torch.tensor(np.array(images), device=device).view(masked_fmri.shape[0], *self.image_shape)
        return images, image_embedding
