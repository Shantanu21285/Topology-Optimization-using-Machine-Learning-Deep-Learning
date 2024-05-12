import torch
import torch.nn as nn

class CustomLossFN(nn.Module):
    def __init__(self, vol_coeff = 1.0):
        super(CustomLossFN, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.vol_coeff = vol_coeff

    def forward(self, output, target):
        bce_loss = self.bce(output, target)
        mse_loss = self.mse(output, target)
        return bce_loss + self.vol_coeff * mse_loss