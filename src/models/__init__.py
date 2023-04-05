from .resnet import resnet32, resnet110, wide_resnet20_8
from .resnetv2 import ResNet50
from .vgg import vgg19, vgg16

from .densenet import densenetd40k12, densenetd100k12, densenetd190k12, densenetd100k40
from .resnet_one import resnet32_one, resnet110_one, wide_resnet20_8_one
from .resnet_okddip import resnet32_okddip, resnet110_okddip, wide_resnet20_8_okddip
from .vgg_one import vgg16_one, vgg19_one
from .vgg_okddip import vgg16_okddip, vgg19_okddip

from .resnet_ffl import resnet32_ffl, resnet110_ffl, resnet_Fusion_module, wide_resnet20_8_ff1
from .vgg_ffl import vgg16_ffl, vgg19_ffl, vgg_Fusion_module
from .vgg_pcl import vgg16_pcl, vgg19_pcl
from .densenet_one import densenetd40k12_one
from .densenet_okddip import densenetd40k12_okddip
from .densenet_pcl import densenetd40k12_pcl
from.densenet_ffl import densenetd40k12_ffl, densenet_Fusion_module
from .resnet_pcl import resnet32_pcl, resnet110_pcl, wide_resnet20_8_pcl

from .resnet_ema import resnet32_ema, resnet110_ema, wide_resnet20_8_ema
from .vgg_ema import vgg16_ema
from .densenet_ema import densenetd40k12_ema
from torchvision.models import resnet18


model_dict = {
    'resnet18': resnet18, 
    'resnet50': ResNet50,
    # DML KDCL OKDPH
    'resnet32': resnet32,
    'resnet110': resnet110,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'wrn_20_8': wide_resnet20_8, 
    
    'densenet-40-12': densenetd40k12, 
    'densenet-100-12': densenetd100k12, 
    'densenet-190-12': densenetd190k12, 
    'densenet-100-40': densenetd100k40, 
    # ONE
    'resnet32_one': resnet32_one, 
    'resnet110_one': resnet110_one,
    'vgg16_one': vgg16_one, 
    'vgg19_one': vgg19_one,
    'densenet-40-12_one': densenetd40k12_one, 
    'wrn_20_8_one': wide_resnet20_8_one, 
    # OKDDip
    'resnet32_okddip': resnet32_okddip, 
    'resnet110_okddip': resnet110_okddip, 
    'vgg16_okddip': vgg16_okddip, 
    'vgg19_okddip': vgg19_okddip, 
    'densenet-40-12_okddip': densenetd40k12_okddip, 
    'wrn_20_8_okddip': wide_resnet20_8_okddip, 
    # PCL
    'resnet32_pcl': resnet32_pcl, 
    'resnet110_pcl': resnet110_pcl,
    'vgg16_pcl': vgg16_pcl, 
    'vgg19_pcl': vgg19_pcl, 
    'densenet-40-12_pcl': densenetd40k12_pcl, 
    'wrn_20_8_pcl': wide_resnet20_8_pcl, 
    # FFL
    'resnet_fm': resnet_Fusion_module, 
    'resnet32_ffl': resnet32_ffl, 
    'resnet110_ffl': resnet110_ffl, 
    'vgg16_ffl': vgg16_ffl, 
    'vgg19_ffl': vgg19_ffl, 
    'vgg_fm': vgg_Fusion_module, 
    'densenet-40-12_ffl': densenetd40k12_ffl, 
    'densenet_fm': densenet_Fusion_module, 
    'wrn_20_8_ffl': wide_resnet20_8_ff1, 
    # ema
    'resnet32_ema': resnet32_ema, 
    'resnet110_ema': resnet110_ema, 
    'wrn_20_8_ema': wide_resnet20_8_ema, 
    'vgg16_ema': vgg16_ema, 
    'densenet-40-12_ema': densenetd40k12_ema
    
}
