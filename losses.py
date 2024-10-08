import torch
import torch.nn as nn

__all__ = ['VAE_loss']

class Decoder_loss(nn.modules.loss._Loss):
    def __init__(self, w1 : float =1.0, w2 : float = 1.0, w3 : float = 1.0) -> None:
        super().__init__()
        self.w1 = w1 # MAE Loss
        self.w2 = w2 # MSE Loss
        self.w3 = w3 # COS Loss

    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        input = input.reshape(input.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        assert input.shape == target.shape, f'Input and target shapes do not match: {input.shape} vs {target.shape}'
        # Check input and target tensors
        assert not torch.isnan(input).any() and not torch.isinf(input).any(), f'Input tensor contains nan or inf values'
        assert not torch.isnan(target).any() and not torch.isinf(target).any(), f'Target tensor contains nan or inf values'
        # MAE Loss
        maeloss = nn.L1Loss()(input, target)
        assert not torch.isnan(maeloss), f'MAE loss is nan: {maeloss}'
        # MSE Loss
        input_01 = torch.where(input > 0, torch.tensor(1., dtype=input.dtype, device=input.device), torch.tensor(0., dtype=torch.float64, device=input.device))
        target_01 = torch.where(target > 0, torch.tensor(1., dtype=target.dtype, device=target.device), torch.tensor(0., dtype=torch.float64, device=target.device))
        mseloss = nn.MSELoss()(input_01, target_01)
        assert not torch.isnan(mseloss), f'MSE loss is nan: {mseloss}'
        # COS Loss
        cosloss = nn.CosineEmbeddingLoss()(input, target, torch.ones([input.shape[0]], device=input.device))
        assert not torch.isnan(cosloss), f'Cosine loss is nan: {cosloss}'

        return self.w1 * maeloss + self.w2 * mseloss + self.w3 * cosloss