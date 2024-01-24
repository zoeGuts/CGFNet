import torch
import torch.nn as nn
import torch.nn.functional as F


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)  


class Enhanced_Channel_Attenion(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(Enhanced_Channel_Attenion, self).__init__()
        self.gate_channels = gate_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.down_conv = nn.Conv2d(1, 1, kernel_size=(2, 1))
        self.mlp = nn.Sequential(
            Flatten(),  # [b, c]
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        
    def forward(self, x):
        r=x
        b, c, _, _=x.size()

        x_avg = self.avg_pool(x).view(self.avg_pool(x).size(0), -1).unsqueeze(axis=1).unsqueeze(axis=1)
        x_max = self.max_pool(x).view(self.max_pool(x).size(0), -1).unsqueeze(axis=1).unsqueeze(axis=1)
        x = torch.cat((x_avg, x_max), dim=2)
        x = self.down_conv(x)
    
        x = self.mlp(x)
        x=torch.sigmoid(x)
        
        return x


class ChannelPool(nn.Module):
    def forward(self, x, k):   # k is stride of local maxpool and avgpool
        avg_pool = F.avg_pool2d(x, k, stride=k)
        max_pool = F.max_pool2d(x, k, stride=k)
        return torch.cat( (torch.max(max_pool,1)[0].unsqueeze(1), torch.mean(avg_pool,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=3, dilation=3, relu=False)

    def forward(self, x):
        r = x
        b, c, h, w = x.size()

        x_compress = self.compress(x, 2)
        x_out = self.spatial(x_compress)
        x_out = F.interpolate(x_out, size=[h, w], mode='nearest')
        x = torch.sigmoid(x_out) 
     
        return x


class MBLA(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(MBLA, self).__init__()
        self.ChannelGate = Enhanced_Channel_Attenion(gate_channels, reduction_ratio)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out1 = self.ChannelGate(x).unsqueeze(axis=2).unsqueeze(axis=2)

        if not self.no_spatial:
            x_out2 = self.SpatialGate(x)

        x_out = torch.mul(x_out1, x_out2)
        x_out = torch.sigmoid(x_out)

        return x_out
