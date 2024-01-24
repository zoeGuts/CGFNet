import torch
import torch.nn as nn
from torch.nn import init

from models.attention.cbam import CBAM
from models.attention.mbla import MBLA
from models.attention.dwAttention import *
from Lofunction import caLofunction_v3 as caLofunction


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=16):
        super(SqueezeBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = self.avg_pool(x).view(batch, channels)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)

        return out * x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, 
                 att_type=None,
                 reduction_ratio=1,
                 spatial_size=None,
                 return_w=False,
                 method_version='original',
                 channel_groups=64,
                 sa_kernel_size=3,
                 no_spatial=False,
                 **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

        self.cbam = CBAM(out_channel) \
            if att_type == 'cbam' else None
        self.mbla = MBLA(out_channel, reduction_ratio=16) \
            if att_type == 'clam' else None
        if att_type == 'ours':
            cls_name = 'GGAM'
            if cls_name in globals():
                self.ours = globals()[cls_name](out_channel, spatial_size, 
                                                reduction_ratio=reduction_ratio,
                                                groups=channel_groups,
                                                sa_kernel_size=sa_kernel_size,
                                                no_spatial=no_spatial)
                if not no_spatial:
                    print("NOW RUNNING: %s | reduction_ratio => %d, groups => %d, saks => %d" % \
                        (cls_name, reduction_ratio, channel_groups, sa_kernel_size))
                else:
                    print("No spatial! Juct only channel attention. reduction_ratio => %d" % reduction_ratio)
            else: 
                self.ours = None
                print("!!!ERROR!!!Don't find this method: %s" % cls_name)
        else:
            self.ours = None
        self.return_w = return_w

    def forward(self, x, attn=None):
        residual = x
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if not self.cbam is None:
            out = self.cbam(out)

        if not self.mbla is None:
            w = self.mbla(out)
            out = torch.mul(out, w)
            
        # [b, c, w, h] -> [b, c, 1, 1]
        if not self.ours is None:
            out, w = self.ours(out)

        out = torch.add(out, residual)
        out = self.relu(out)
        if self.return_w and self.ours is not None:
            return out, w
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, 
                 groups=1, width_per_group=64, 
                 att_type=None, 
                 reduction_ratio=1,
                 spatial_size=None,
                 return_w=False,
                 method_version='original',
                 channel_groups=64,
                 sa_kernel_size=3,
                 no_spatial=False,
                 **kwargs):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        
        self.conv1 = nn.Conv2d(in_channel, width, 
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width,
                               groups=groups, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channel * self.expansion, 
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.cbam = CBAM(out_channel * self.expansion) \
            if att_type == 'cbam' else None
        self.mbla = MBLA(out_channel * self.expansion, reduction_ratio=2) \
            if att_type == 'clam' else None
            
        if att_type == 'ours':
            cls_name = 'GGAM'
            if cls_name in globals():
                self.ours = globals()[cls_name](out_channel, spatial_size, 
                                                reduction_ratio=reduction_ratio,
                                                groups=channel_groups,
                                                sa_kernel_size=sa_kernel_size,
                                                no_spatial=no_spatial)
                if not no_spatial:
                    print("NOW RUNNING: %s | reduction_ratio => %d, groups => %d, saks => %d" % \
                        (cls_name, reduction_ratio, channel_groups, sa_kernel_size))
                else:
                    print("No spatial! Juct only channel attention. reduction_ratio => %d" % reduction_ratio)
            else: 
                self.ours = None
                print("!!!ERROR!!!Don't find this method: %s" % cls_name)
        else:
            self.ours = None
        self.return_w = return_w

    def forward(self, x, attn=None):
        residual = x
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if not self.cbam is None:
            out = self.cbam(out)

        if not self.mbla is None:
            w = self.mbla(out)
            out = torch.mul(out, w)
        
        # [b, c, w, h] -> [b, c, 1, 1]
        if not self.ours is None:
            out, w = self.ours(out)
        
        out = torch.add(out, residual)
        out = self.relu(out)

        if self.return_w and self.ours is not None:
            return out, w
        else:
            return out


class ResNet(nn.Module):
    
    def __init__(self, 
                 block, 
                 blocks_num, 
                 num_classes=1000, 
                 groups=1,
                 width_per_group=64,
                 att_type=None,
                 reduction_ratio=1,
                 beta=1.0,
                 method_version='original',
                 channel_groups=64,
                 sa_kernel_size=3,
                 no_spatial=False,
                 no_lofct=False):
        super(ResNet, self).__init__()
        self.in_channel = 64
        spatial = [56, 28, 14, 7]
        
        self.groups = groups
        self.width_per_group = width_per_group
        
        # select the loss function
        self.att_type = att_type
        if self.att_type == 'ours':
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

            self.beta = beta
            self.reduction_ratio = reduction_ratio
            self.no_spatial = no_spatial
            
            self.channel_groups = channel_groups
            self.sa_kernel_size = sa_kernel_size
            self.method_version = method_version.upper()
            
            self.no_lofct = no_lofct
            if self.no_lofct:
                print("Just only cross entropy!")
            else:
                print("Cross entropy, and lofct, beta=> %f" % self.beta)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.conv1 = nn.Conv2d(3, self.in_channel, 
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64,  blocks_num[0], spatial_size=spatial[0], return_w=False)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2, spatial_size=spatial[1], return_w=False)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2, spatial_size=spatial[2], return_w=False)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2, spatial_size=spatial[3], return_w=True)
        
        self.avgpool = nn.AvgPool2d(7)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split(".")[-1] == 'bias':
                self.state_dict()[key][...] = 0
            elif key.split('.')[-1] == "weight":
                if "conv" in key.split('.')[-2]:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                elif "bn" in key.split('.')[-2]:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
                else:
                    print('not inited: %s' % key)

    def _make_layer(self, block, channel, block_num, stride=1, spatial_size=None, return_w=False):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, 
                            channel, 
                            stride=stride, 
                            downsample=downsample,
                            groups=self.groups,
                            width_per_group=self.width_per_group,
                            att_type=self.att_type,
                            reduction_ratio=self.reduction_ratio,
                            spatial_size=spatial_size,
                            return_w=return_w,
                            method_version=self.method_version,
                            channel_groups=self.channel_groups,
                            sa_kernel_size=self.sa_kernel_size,
                            no_spatial=self.no_spatial))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, 
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group, 
                                att_type=self.att_type,
                                reduction_ratio=self.reduction_ratio,
                                spatial_size=spatial_size,
                                return_w=return_w,
                                method_version=self.method_version,
                                channel_groups=self.channel_groups,
                                sa_kernel_size=self.sa_kernel_size,
                                no_spatial=self.no_spatial))

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
       
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        if self.att_type == 'ours':
            attn = None
            for alayer in self.layer4:
              x, attn = alayer(x, attn)
        else: 
            x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        if self.training:
            if self.att_type == 'ours' and not self.no_lofct and self.method_version != 'V3':
                loss = self.criterion(x, y) + caLofunction(attn=attn, y=y) * self.beta
            else:
                loss = self.criterion(x, y)
            return x, loss
        else:
            return x


def ResNet18(num_classes, att_type, reduction_ratio, beta=1.0, method_version='original', channel_groups=64, sa_kernel_size=3, no_spatial=False, no_lofct=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, att_type=att_type, 
                   reduction_ratio=reduction_ratio, beta=beta, method_version=method_version, channel_groups=channel_groups, 
                   sa_kernel_size=sa_kernel_size, no_spatial=no_spatial, no_lofct=no_lofct)
    return model


def ResNet34(num_classes, att_type, reduction_ratio, beta=1.0, method_version='original', channel_groups=64, sa_kernel_size=3, no_spatial=False, no_lofct=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, att_type=att_type, 
                   reduction_ratio=reduction_ratio, beta=beta, method_version=method_version, channel_groups=channel_groups, 
                   sa_kernel_size=sa_kernel_size, no_spatial=no_spatial, no_lofct=no_lofct)
    return model


def ResNet50(num_classes, att_type, reduction_ratio, beta=1.0, method_version='original', channel_groups=64, sa_kernel_size=3, no_spatial=False, no_lofct=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, att_type=att_type, 
                   reduction_ratio=reduction_ratio, beta=beta, method_version=method_version, channel_groups=channel_groups, 
                   sa_kernel_size=sa_kernel_size, no_spatial=no_spatial, no_lofct=no_lofct)
    return model


def ResNet101(num_classes, att_type, reduction_ratio, beta=1.0, method_version='original', channel_groups=64, sa_kernel_size=3, no_spatial=False, no_lofct=False):
    model = ResNet(BasicBlock, [3, 4, 23, 3], num_classes, 
                   att_type=att_type, reduction_ratio=reduction_ratio, beta=beta,
                   method_version=method_version, channel_groups=channel_groups, sa_kernel_size=sa_kernel_size, no_spatial=no_spatial, no_lofct=no_lofct)
    return model


