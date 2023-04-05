'''
VGG16 for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

'''

import torch
import torch.nn as nn
from copy import deepcopy

__all__ = ['vgg16', 'vgg19']

#cfg = {
#    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}


class Branch(nn.Module):
    def __init__(self, layer, fc) -> None:
        super().__init__()
        self.layer = layer
        self.fc = fc

    def forward(self, x):
        out = self.layer(x)
        out = out.view(out.size(0), -1)
        f = out  # [N, C]
        out = self.fc(out)
        return out, f


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, dropout = 0.0, ema=False, num_branches=2,  ):
        super(VGG, self).__init__()
        self.inplances = 64
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(True)
        self.layer1 = self._make_layers(128, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ema = ema
        self.num_branches = num_branches

        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4
        
        self.layer2 = self._make_layers(256, num_layer)
        self.layer3 = self._make_layers(512, num_layer)


        layer = self._make_layers(512, num_layer)
        fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(512, num_classes),
        )

        self.branch0 = Branch(layer, fc)
        for bid in range(1, self.num_branches):
            setattr(self, 'branch' + str(bid), deepcopy(self.branch0))

        self.en_fc = nn.Linear(512 * self.num_branches, num_classes)
        if self.ema:
            for param in self.parameters():
                param.detach_()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _make_layers(self, input, num_layer):    
        layers=[]
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)
    
    # def forward(self, x1, x2, x3):
    def forward(self, imgs):
        out_list, feat_list = [], []
        for bid in range(self.num_branches):
            ff = self.extract(imgs[:, bid, ...].contiguous())
            out, feat = getattr(self, f'branch{bid}')(ff) 
            out_list.append(out)
            feat_list.append(feat)
        
        if self.ema:
            return out_list
        else:
            en_feat = torch.cat(feat_list, dim=1)
            en_out = self.en_fc(en_feat)
            return out_list, en_out

    def extract(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x 
    
def vgg16_pcl(**kwargs):
    """
    Constructs a VGG16 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=16, **kwargs)
    return model
    
def vgg19_pcl(**kwargs):
    """
    Constructs a VGG19 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=19, **kwargs)
    return model
