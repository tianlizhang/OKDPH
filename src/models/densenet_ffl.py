'''
DenseNet for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
2. https://github.com/liuzhuang13/DenseNet
3. https://github.com/gpleiss/efficient_densenet_pytorch
4. Gao Huang, zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
Densely Connetcted Convolutional Networks. https://arxiv.org/abs/1608.06993

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
import math

__all__ = ['DenseNet', 'densenetd40k12', 'densenetd100k12', 'densenetd100k40', 'densenetd190k12']

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class FFL_DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=12, block_config=[16, 16, 16], compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False, KD = False, num_branches=2):

        super(FFL_DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7
        self.KD = KD
        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))
        self.num_branches = num_branches
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d_0' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
            else:
                for bid in range(1, self.num_branches):
                    self.features.add_module(f'denseblock{i+1}_{bid}', block)

        # Final batch norm
        for bid in range(self.num_branches):
            self.features.add_module(f'norm_final_{bid}', nn.BatchNorm2d(num_features))
        # Linear layer
        for bid in range(self.num_branches):
            setattr(self, f'classifier_{bid}', nn.Linear(num_features, num_classes))

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # embed()
        x1 = getattr(self.features, 'conv0')(x)
        x1 = getattr(self.features, 'denseblock1_0')(x1)
        x1 = getattr(self.features, 'transition1')(x1)
        x2 = getattr(self.features, 'denseblock2_0')(x1)
        x2 = getattr(self.features, 'transition2')(x2) # [B, 60, 8, 8]
        
        out_list, fmap = [], []
        for bid in range(self.num_branches):
            x3 = getattr(self.features, f'denseblock3_{bid}')(x2)
            x3 = getattr(self.features, f'norm_final_{bid}')(x3)
            fmap.append(x3)
            
            x3 = F.relu(x3, inplace=True)  
            x3 = F.avg_pool2d(x3, kernel_size=self.avgpool_size).view(x3.size(0), -1) # B x 132
            x3 = getattr(self, f'classifier_{bid}')(x3)
            out_list.append(x3)
        return out_list, fmap


class densenet_Fusion_module(nn.Module):
    def __init__(self, num_classes=10, channel=132, sptial=8, num_branches=2):
        super(densenet_Fusion_module, self).__init__()
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
    
    
def densenetd40k12_ffl(pretrained=False, path=None, **kwargs):
    """
    Constructs a densenetD40K12 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    
    model = FFL_DenseNet(growth_rate = 12, block_config = [6,6,6], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model

def densenetd100k12_ffl(pretrained=False, path=None, **kwargs):
    """
    Constructs a densenetD100K12 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    
    model = FFL_DenseNet(growth_rate = 12, block_config = [16,16,16], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model

def densenetd190k12_ffl(pretrained=False, path=None, **kwargs):
    """
    Constructs a densenetD190K12 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    
    model = FFL_DenseNet(growth_rate = 12, block_config = [31,31,31], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model

def densenetd100k40_ffl(pretrained=False, path=None, **kwargs):
    """
    Constructs a densenetD100K40 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    model = FFL_DenseNet(growth_rate = 40, block_config = [16,16,16], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


if  __name__ == '__main__':
    from IPython import embed
    
    device = torch.device('cuda:3')
    model = densenetd40k12_ffl().to(device)
    a = torch.rand(2, 3, 32, 32).to(device)
    outputs1, outputs2, fmap  = model(a)
    print(outputs1.shape, outputs2.shape)
    fuse = densenet_Fusion_module(num_classes=10, channel=132, sptial=8).to(device)
    out = fuse(fmap[0], fmap[1])
    embed()
    
    
    
    # embed()