from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)
        
        
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
        
        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        #block = Bottleneck if depth >=44 else BasicBlock
        block = BasicBlock
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        
        x = self.layer3(x)  # 8x8
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class FFL_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, num_branches = 2, width_per_group=64):
        super(FFL_ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        # if depth <= 44:
        #     assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44'
        #     n = (depth - 2) // 6
        #     block = BasicBlock
        # else:
        #     assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 47, 56, 110, 1199'
        #     n = (depth - 2) // 9
        #     block = Bottleneck
        self.num_branches = num_branches
        # block = BasicBlock
       # block = Bottleneck if depth >= 44 else BasicBlock
        self.inplanes = 16
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)

        fix_inplanes=self.inplanes
        for bid in range(1, self.num_branches+1):
            setattr(self, f'layer3_{bid}', self._make_layer(block, 64, layers[2], stride=2))
            self.inplanes = fix_inplanes 
        
        self.avgpool = nn.AvgPool2d(8)
        for bid in range(1, self.num_branches+1):
            setattr(self, f'classfier3_{bid}', nn.Linear(64 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / layers[0]))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16

        fmap, out_list = [], []
        for bid in range(1, self.num_branches+1):
            x_3 = getattr(self, f'layer3_{bid}')(x)
            fmap.append(x_3)
            
            x_3 = self.avgpool(x_3)
            x_3 = x_3.view(x_3.size(0), -1)
            x_3 = getattr(self, f'classfier3_{bid}')(x_3)
            out_list.append(x_3)
        return out_list, fmap


class resnet_Fusion_module(nn.Module):
    def __init__(self, num_classes=10, channel=64, sptial=8, num_branches=2):
        super(resnet_Fusion_module, self).__init__()
        self.fc2   = nn.Linear(channel, num_classes)
        self.conv1 =  nn.Conv2d(channel*num_branches, channel*num_branches, kernel_size=3, stride=1, padding=1, groups=channel*num_branches, bias=False)
        self.bn1 = nn.BatchNorm2d(channel * num_branches)
        self.conv1_1 = nn.Conv2d(channel*num_branches, channel, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channel)

        self.sptial = sptial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.num_branches = num_branches
    

    def forward(self, fmap):
        input = torch.cat(fmap, 1)
        # from IPython import embed
        # embed()
        x = F.relu(self.bn1((self.conv1(input))))
        x = F.relu(self.bn1_1(self.conv1_1(x)))

        x = F.avg_pool2d(x, self.sptial)
        x = x.view(x.size(0), -1)

        out = self.fc2(x)
        return out




def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

def resnet_ffl(**kwargs):
    """
    Constructs a ResNet model.
    """
    return FFL_ResNet(**kwargs)


def resnet32_ffl(**kwargs):
    """
    Constructs a ResNet model.
    """
    return FFL_ResNet(BasicBlock, [5,5,5], **kwargs)


def resnet110_ffl(**kwargs):
    """
    Constructs a ResNet model.
    """
    return FFL_ResNet(Bottleneck,[12,12,12], **kwargs)


def wide_resnet20_8_ff1(**kwargs):
    """
    Constructs a ResNet model.
    """
    return FFL_ResNet(Bottleneck, [2,2,2],**kwargs)

# if __name__















