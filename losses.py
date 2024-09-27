import torch
import torch.nn as nn

__all__ = ['VAE_loss']

class Decoder_loss(nn.modules.loss._Loss):
    def __init__(self, w1 : float =1.0, w2 : float = 1.0, w3 : float = 1.0) -> None:
        super().__init__()
        self.w1 = w1 # MAE Loss
        self.w2 = w2 # MSE Loss
        self.w3 = w3 # Cosine Loss

    def forward(self, input, target) -> torch.Tensor:
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        # Check input and target tensors
        assert not torch.isnan(input).any() and not torch.isinf(input).any(), f'Input tensor contains nan or inf values'
        assert not torch.isnan(target).any() and not torch.isinf(target).any(), f'Target tensor contains nan or inf values'
        # MAE Loss
        maeloss = nn.L1Loss()(input, target)
        assert not torch.isnan(maeloss), f'MAE loss is nan: {maeloss}'
        # MSE Loss
        mseloss = nn.MSELoss()(input, target)
        assert not torch.isnan(mseloss), f'MSE loss is nan: {mseloss}'
        # COS Loss
        cosloss = nn.CosineEmbeddingLoss()(input, target, torch.ones([input.shape[0]], device=input.device))
        assert not torch.isnan(cosloss), f'Cosine loss is nan: {cosloss}'
        return self.w1 * maeloss + self.w2 * mseloss + self.w3 * cosloss