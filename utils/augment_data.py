import torch
import torch.nn as nn
import torch.nn.functional as F

from training_utils import unsqueeze_2d_tensor, squeeze_3d_tensor

class RandomGaussianBlur1D(nn.Module):

    # kernel_size must be even
    # more sigma more blur
    # p is the probability to apply blur
    def __init__(self, kernel_size=5, sigma_range=(0.5, 2.0), p=0.5):
        super(RandomGaussianBlur1D, self).__init__()
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.p = p
        
        self.padding = kernel_size // 2

    # kernel
    def get_gaussian_kernel(self, sigma, device):
        k_half = self.kernel_size // 2
        x = torch.arange(-k_half, k_half + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, -1)

    # random sigma value
    def get_random_sigma(self):
        return torch.empty(1).uniform_(self.sigma_range[0], self.sigma_range[1]).item()

    def forward(self, x):
        # randomly apply blur only in training (no test or val)
        if not self.training:
            return x
        if torch.rand(1).item() > self.p:
            return x
            
        x = unsqueeze_2d_tensor(x)
            
        batch_size = x.shape[0]
        device = x.device
        
        # random sigma in range
        sigma = self.get_random_sigma()
        
        kernel = self.get_gaussian_kernel(sigma, device)
        x_blurred = F.conv1d(x, kernel, padding=self.padding) # kernel
        
        x_blurred = squeeze_3d_tensor(x_blurred)
            
        return x_blurred