import torch
from torch import nn
import torch.nn.functional as F


class ResonancePeaksLoss(nn.Module):
    def __init__(self, w_amp=10.0, w_grad=5.0, w_wass=2.0, w_sam=1.0):
        super().__init__()
        self.w_amp = w_amp
        self.w_grad = w_grad
        self.w_wass = w_wass
        self.w_sam = w_sam

        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-8)

    def get_derivatives(self, x):
        return x[:, 1:] - x[:, :-1]
    
    def weighted_mse(self, input, target, weight):
        return torch.mean(weight * (input - target) ** 2)


    def wasserstein_loss(self, input, target):
        # implemented from https://github.com/TakaraResearch/Pytorch-1D-Wasserstein-Statistical-Loss
        
        # compute cumulative distribution functions
        cdf_tensor_a = torch.cumsum(input,dim=-1)
        cdf_tensor_b = torch.cumsum(target,dim=-1)
        # distance
        return torch.mean(torch.abs(cdf_tensor_a-cdf_tensor_b))

    def forward(self, y_pred, y_true):
        # # loss v1
        # weights = 1.0 + 5.0 * (1.0 - y_true)

        # loss v2 - exponential weighting to give priority to the most important peaks
        peaks_importance = (1.0 - y_true)**3
        max_vals, _ = torch.max(peaks_importance, dim=1, keepdim=True)
        peaks_importance = peaks_importance / (max_vals + 1e-6) # [0,1] range
        weights = 1.0 + (self.w_amp * peaks_importance)

        loss_amp = self.weighted_mse(y_pred, y_true, weights)

        # first derivatives
        grad_pred = self.get_derivatives(y_pred)
        grad_true = self.get_derivatives(y_true)
        loss_grad = torch.mean((grad_pred - grad_true)**2)

        # create probability density functions
        wass_pred = F.relu(1.0 - y_pred) + 1e-6
        wass_true = F.relu(1.0 - y_true) + 1e-6
        # PDFs must be normalised
        wass_pred = wass_pred / torch.sum(wass_pred, dim=1, keepdim=True)
        wass_true = wass_true / torch.sum(wass_true, dim=1, keepdim=True)
        loss_wass = self.wasserstein_loss(wass_pred, wass_true)

        # loss v2 - cosine similarity loss
        # spectral angle mapper
        loss_sam = torch.mean(1.0 - self.cosine(y_pred, y_true))

        return loss_amp + \
               (self.w_grad * loss_grad) + \
               (self.w_wass * loss_wass) + \
               (self.w_sam * loss_sam)
    