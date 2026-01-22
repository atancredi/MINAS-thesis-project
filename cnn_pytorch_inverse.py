import torch
import torch.nn as nn
import torch.nn.functional as F

class PeakAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(PeakAttentionBlock, self).__init__()
        self.conv_mask = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [Batch, Channels, Length]
        # attention_mask: [Batch, 1, Length]
        attention_mask = self.sigmoid(self.conv_mask(x))
        
        return x * attention_mask

class ResidualBlockCNN(nn.Module):
    def __init__(self, channels):
        super(ResidualBlockCNN, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.gelu(out)

class InverseCNN(nn.Module):
    def __init__(self, input_points=81, output_geom_dim=4):
        super(InverseCNN, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        
        self.attention = PeakAttentionBlock(in_channels=32)
        
        self.res_blocks = nn.Sequential(
            ResidualBlockCNN(32),
            nn.MaxPool1d(2), # 81 -> 40
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            ResidualBlockCNN(64),
            ResidualBlockCNN(64),
            nn.MaxPool1d(2) # 40 -> 20
        )
        
        flat_dim = 20 * 64 
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_geom_dim),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        feat = self.initial_conv(x)
        feat_weighted = self.attention(feat)
        
        deep_feat = self.res_blocks(feat_weighted)
        
        geom = self.regressor(deep_feat)
        return geom