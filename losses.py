# import torch
# import torch.nn as nn

# __all__ = ['VAE_loss']

# class VAE_loss(nn.modules.loss._Loss):
#     def __init__(self, w_mse : float, w_kld : float):
#         super().__init__()
#         self.w_mse = float(w_mse)
#         self.w_kld = float(w_kld)

#     def forward(self, input, target, mean, log_var):
#         mseloss = nn.MSELoss()(input, target)
#         KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
#         return self.w_mse * mseloss + self.w_kld * KLD