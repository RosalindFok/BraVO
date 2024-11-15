import torch
import torch.nn as nn
import torch.nn.functional as F      

__all__ = ['Decoder_loss']

# class Decoder_loss(nn.modules.loss._Loss):
#     def __init__(self, w1 : float =1.0, w2 : float = 1.0, w3 : float = 1.0) -> None:
#         super().__init__()
#         self.w1 = w1 # MAE Loss
#         self.w2 = w2 # MSE Loss
#         self.w3 = w3 # COS Loss

#     def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
#         input = input.reshape(input.shape[0], -1)
#         target = target.reshape(target.shape[0], -1)
#         assert input.shape == target.shape, f'Input and target shapes do not match: {input.shape} vs {target.shape}'
#         # Check input and target tensors
#         assert not torch.isnan(input).any() and not torch.isinf(input).any(), f'Input tensor contains nan or inf values'
#         assert not torch.isnan(target).any() and not torch.isinf(target).any(), f'Target tensor contains nan or inf values'
#         # MAE Loss
#         maeloss = nn.L1Loss()(input, target)
#         assert not torch.isnan(maeloss), f'MAE loss is nan: {maeloss}'
#         # MSE Loss
#         mseloss = nn.MSELoss()(input, target)
#         assert not torch.isnan(mseloss), f'MSE loss is nan: {mseloss}'
#         # COS Loss
#         cosloss = nn.CosineEmbeddingLoss()(input, target, torch.ones([input.shape[0]], device=input.device))
#         assert not torch.isnan(cosloss), f'Cosine loss is nan: {cosloss}'

#         return self.w1 * maeloss + self.w2 * mseloss + self.w3 * cosloss

class Decoder_loss(nn.modules.loss._Loss):
    def __init__(self, w1 : float =1.0, w2 : float = 1.0, w3 : float = 1.0) -> None:
        super().__init__()
    
    def forward(self, input : torch.Tensor, target : torch.Tensor, temperature=0.07) -> torch.Tensor:
        input = input.reshape(input.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        input = F.normalize(input, dim=1)  
        target = F.normalize(target, dim=1)  

        logits = torch.matmul(input, target.T) / temperature  

        batch_size = input.size(0)  
        labels = torch.arange(batch_size).to(input.device)  

        loss_i2t = F.cross_entropy(logits, labels)  
        loss_t2i = F.cross_entropy(logits.T, labels)  
        loss = (loss_i2t + loss_t2i) / 2  
        return loss 