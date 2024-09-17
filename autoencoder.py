import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, 
        in_channels, 
        out_channels,
        num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.swish = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,padding=1)

    def forward(self, x):
        h = x
        h = self.conv1(self.swish(self.norm1(h)))                           
        h = self.conv2(self.swish(self.norm2(h)))
        return x + h

class ConvBlock(nn.Module):
    def __init__(self, 
        in_channels, 
        out_channels,
        num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.swish = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv1(self.swish(self.norm1(x)))    
        
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=0)
    
    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, num_downsamples):
        super(Encoder, self).__init__()
        layers = [nn.Conv2d(in_channels, hid_channels, 3, stride=1, padding=1)]

        in_ch  = hid_channels
        out_ch = hid_channels * 2

        for _ in range(num_downsamples):
            layers.append(ResidualBlock(in_ch, in_ch))
            layers.append(Downsample(in_ch, out_ch))
            in_ch = out_ch
            out_ch *= 2
        layers.append(ConvBlock(in_ch, out_channels))
        self.model = nn.ModuleList(layers)
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, num_upsamples):
        super(Decoder, self).__init__()
    
        layers = [nn.Conv2d(in_channels, hid_channels * 2 ** num_upsamples, 3, stride=1, padding=1)]    

        in_ch  = hid_channels * 2 ** num_upsamples
        out_ch = hid_channels * 2 ** (num_upsamples - 1)

        for _ in range(num_upsamples):
            layers.append(ResidualBlock(in_ch, in_ch))
            layers.append(Upsample(in_ch, out_ch))
            in_ch = out_ch
            out_ch = out_ch // 2
        layers.append(ConvBlock(in_ch, out_channels))
        self.model = nn.ModuleList(layers)
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

def from_pretrained(path):
    epoch = int(path.split('-')[-1].split('.')[0])

    print('\nLoading "AutoEncoder@epoch[{:d}]"...'.format(epoch), end='')
    checkpoint = torch.load(path)
    args = checkpoint['args']
    state_dict = checkpoint['model']

    model = Autoencoder(args)
    """
    new = {}
    for key, value in state_dict.items():
        new[key[key.find('.')+1:]] = value

    checkpoint_new = {
        "model": new,
        "opt"  : checkpoint['opt'],
        "args" : args
    }

    torch.save(checkpoint_new, "vae-models/checkpoints/autoencoder.pt")
    """
    model.load_state_dict(state_dict)
    print("Done.")
    return model

class Autoencoder(nn.Module):
    def __init__(self, args):
        in_channels = args.in_channels
        latent_channels = args.latent_channels
        hid_channels = args.hid_channels
        num_downsamples = args.num_downsamples
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_channels, latent_channels, hid_channels, num_downsamples)
        self.decoder = Decoder(latent_channels, in_channels, hid_channels, num_downsamples)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def loss(self, x):
        x_hat = self(x)
        return F.l1_loss(x_hat, x)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
if __name__ == '__main__':
    model = Autoencoder(3, 4, 64, 3)
    x = torch.randn(1,3,64,64)
    z = model.encode(x)
    print(z.shape)
    x_recon = model.decode(z)
    print(x_recon.shape)