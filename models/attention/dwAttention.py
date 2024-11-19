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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DWConv_BN_RELU(nn.Module):
    def __init__(self, channels, kernel_size=3, relu=True, stride=1, padding=1):
        super(DWConv_BN_RELU, self).__init__()

        self.dwconv = nn.Conv2d(channels, channels, groups=channels, stride=stride,
                                kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = relu

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class AGAB(nn.Module):
    def __init__(self, gate_channels, spatial_size,
                 iType='Conv1_ReLU_Conv1',
                 reduction_ratio=1,
                 is_sigmoid=True):
        super(AGAB, self).__init__()
        
        self.iType = iType
        self.is_sigmoid = is_sigmoid
        
        self.downop = DWConv_BN_RELU(gate_channels, kernel_size=spatial_size, relu=False, stride=1, padding=0)

        # =>[b, c, 1, 1]
        if iType == 'Conv1_ReLU_Conv1': 
            self.mlp = nn.Sequential(
                nn.Conv2d(gate_channels, gate_channels // reduction_ratio, kernel_size=1, padding=0, bias=False),
                nn.ReLU(),
                nn.Conv2d(gate_channels // reduction_ratio, gate_channels, kernel_size=1, padding=0, bias=False)
            )
        elif iType == 'Conv1_BN_ReLU_Conv1':  
            self.mlp = nn.Sequential(
                nn.Conv2d(gate_channels, gate_channels // reduction_ratio, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(gate_channels // reduction_ratio),
                nn.ReLU(),
                nn.Conv2d(gate_channels // reduction_ratio, gate_channels, kernel_size=1, padding=0, bias=False)
            )
        elif iType == 'None':
            self.mlp = Identity()
    
    def forward(self, x):
        attn = self.downop(x)  # [b, c, 1, 1]
        attn = self.mlp(attn)
        # Multi-scale information fusion 
        if self.is_sigmoid:   
           attn = torch.sigmoid(attn)
        
        return attn 


class GGAM(nn.Module):
    def __init__(self, gate_channels, spatial_size,
                 reduction_ratio=1,
                 groups=64,
                 sa_kernel_size=3,
                 is_sigmoid=True,
                 no_spatial=False):
        super(GGAM, self).__init__()
        
        self.is_sigmoid = is_sigmoid
        self.no_spatial = no_spatial
        self.dwattn = AGAB(gate_channels, spatial_size, 
                            iType='Conv1_ReLU_Conv1', 
                            reduction_ratio=reduction_ratio, 
                            is_sigmoid=is_sigmoid)
        if not no_spatial:
            self.groups = groups
            group_channels = gate_channels // groups
            self.channel_pool = ChannelPool()
            
            self.dwconv = nn.Conv2d(2, 1, groups=1, stride=1,
                    kernel_size=sa_kernel_size, padding=(sa_kernel_size-1) // 2, bias=False)
            self.bn = nn.BatchNorm2d(1)
            
            self.wght = nn.parameter.Parameter(torch.zeros(1, groups, 1, 1))
            self.bias = nn.parameter.Parameter(torch.ones(1, groups, 1, 1))
         
    def forward(self, x):
        b, c, h, w = x.shape
        
        cattn = self.dwattn(x)
        if not self.is_sigmoid:
            cattn = torch.sigmoid(cattn)
        x_cattn = x * cattn
        if self.no_spatial:
            return x_cattn, cattn
        
        # spatial attention
        x = x.reshape(b * self.groups, -1, h, w)
        sattn = self.bn(self.dwconv(self.channel_pool(x)))
        sattn = sattn.reshape(b, -1, h, w)
        sattn = sattn * self.wght + self.bias
        sattn = sattn.reshape(b * self.groups, -1, h, w)
        sattn = torch.sigmoid(sattn)
        x_sattn = x * sattn
        x_sattn = x_sattn.reshape(b, c, h, w)
            
        out = (x_cattn + x_sattn) * 0.5

        return out, cattn 
