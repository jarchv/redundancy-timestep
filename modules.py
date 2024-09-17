import torch
import torch.nn as nn
import torch.nn.functional as F

def GroupNorm(in_channels):
    return nn.GroupNorm(num_channels=in_channels, num_groups=32, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, stride=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)
    
    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, 
        in_channels, 
        out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = GroupNorm(in_channels)
        self.swish = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

        self.norm2 = GroupNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,padding=1)

        if in_channels != out_channels:
            self.conv_up = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = x                                       # h: (B,C,H,W)
        h = self.norm1(h)                           
        h = self.swish(h)
        h = self.conv1(h)                           # h: (B,N,H,W)
        
        
        h = self.norm2(h)                           
        h = self.swish(h)
        h = self.conv2(h)                           # h: (B,N,H,W)

        if self.in_channels != self.out_channels:
            x = self.conv_up(x)            # x: (B,N,H,W)
        return x + h            

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        self.norm = GroupNorm(in_channels)
        self.q    = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.k    = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.v    = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)

        self.swish = nn.SiLU(inplace=True)

    def forward(self, x):
        h = x
        h = self.norm(h)
        h = self.swish(h)

        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        # Compute Attention Map
        B, C, H, W = q.shape
        q = q.reshape(B, C, -1)                     # q: (B,C,H*W)
        q = q.permute(0, 2, 1)                      # q: (B,H*W,C)
        
        k = k.reshape(B, C, -1)                     # k: (B,C,H*W)
        w = torch.bmm(q, k)                         # w: (B,H*W,H*W)
        w = w / (C ** 0.5)
        w = F.softmax(w, dim=-1)
        
        # Compute Context Vector
        v = v.reshape(B, C, -1)
        w = w.permute(0, 2, 1)                      # w: (B,H*W,H*W)
        c = torch.bmm(v, w)                         # c: (B,C,H*W)
        c = c.reshape(B, C, H, W)                   # c: (B,C,H,W)
        c = self.proj(c)                            # c: (B,C,H,W)

        return x + c