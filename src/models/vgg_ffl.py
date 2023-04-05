'''
VGG16 for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['vgg16', 'vgg19']

#cfg = {
#    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}

class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, dropout = 0.0, KD= False, num_branches = 2):
        super(VGG, self).__init__()
        self.KD = KD
        self.inplances = 64
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(True)
        self.layer1 = self._make_layers(128, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.num_branches = num_branches
        
        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4
        
        self.layer2 = self._make_layers(256, num_layer)
        self.layer3 = self._make_layers(512, num_layer)
        
        for bid in range(1, self.num_branches+1):
            setattr(self, f'layer4_{bid}', self._make_layers(512, num_layer))
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(512, num_classes),
        )
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
    
    def forward(self, x):
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
        
        fmap, out_list = [], []
        for bid in range(1, self.num_branches+1):
            x4 = getattr(self, f'layer4_{bid}')(x)
            fmap.append(x4)
            
            x4 = x4.view(x.size(0), -1)
            x4 = self.classifier(x4)
            out_list.append(x4)
        return out_list, fmap


class vgg_Fusion_module(nn.Module):
    def __init__(self,num_classes, channel=512,sptial=1, num_branches=2):
        super(vgg_Fusion_module, self).__init__()
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
                
    def forward(self, fmap):
        input = torch.cat(fmap,1)
        x = F.relu(self.bn1((self.conv1(input))))
        x = F.relu(self.bn1_1(self.conv1_1(x)))

        x = F.avg_pool2d(x, self.sptial)
        x = x.view(x.size(0), -1)

        out = self.fc2(x)
        return out
    
    
def vgg16_ffl(pretrained=False, path=None, **kwargs):
    """
    Constructs a VGG16 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=16, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
    
def vgg19_ffl(pretrained=False, path=None, **kwargs):
    """
    Constructs a VGG19 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=19, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model



if  __name__ == '__main__':
    device = torch.device('cuda:3')
    model = vgg16_ffl().to(device)
    a = torch.rand(2, 3, 32, 32).to(device)
    
    outputs1, outputs2, fmap = model(a)
    print(outputs1.shape, outputs2.shape)
    
    fuse = Fusion_module(num_classes=10, channel=512, sptial=1).to(device)
    from IPython import embed
    embed()
    
    
    