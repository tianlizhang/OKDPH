import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
import pickle as pkl
import numpy as np

from .cifar import CIFAR10, CIFAR100
from .auto_aug import Cutout, CIFAR10Policy
from .cifar_noisy import NoisyCIFAR
from IPython import embed


def load_dataset(args, trans_cur='', seed=None):
    if isinstance(trans_cur, list):
        train_transform = []
        for idx, trans_c in enumerate(trans_cur):
            if isinstance(seed, list):
                train_trans, test_transform = build_transform(args, trans_c, seed[idx])
            else:
                train_trans, test_transform = build_transform(args, trans_c, seed)
            train_transform.append(train_trans)
    else:
        train_transform, test_transform = build_transform(args, trans_cur, seed)

    if args.dataset == "cifar100":
        traindataset = CIFAR100(args.data_path, train=True, download=True,transform=train_transform)
        testdataset = CIFAR100(args.data_path, train=False, transform=test_transform)
        args.num_classes = 100

    elif args.dataset == "cifar10":
        traindataset = CIFAR10(args.data_path, train=True, download=True,transform=train_transform)
        testdataset = CIFAR10(args.data_path, train=False, transform=test_transform)
        args.num_classes = 10
    
    elif args.dataset == "cifar10_noisy":
        cifar_train = CIFAR10(args.data_path, train=True, download=True,transform=train_transform)
        cifar_test = CIFAR10(args.data_path, train=False, transform=test_transform)
        
        traindataset = NoisyCIFAR(cifar_train, args.noise_mean, args.noise_std)
        testdataset = NoisyCIFAR(cifar_test, args.noise_mean, args.noise_std)
        args.num_classes = 10
    
    elif args.dataset == "cifar100_noisy":
        cifar_train = CIFAR100(args.data_path, train=True, download=True,transform=train_transform)
        cifar_test = CIFAR100(args.data_path, train=False, transform=test_transform)
        
        traindataset = NoisyCIFAR(cifar_train, args.noise_mean, args.noise_std)
        testdataset = NoisyCIFAR(cifar_test, args.noise_mean, args.noise_std)
        args.num_classes = 100

    elif args.dataset == "cifar100_1":
        traindataset = CIFAR100(args.data_path, train=True, download=True,transform=train_transform,sample_id_path=args.sample_id_path)
        testdataset = CIFAR100(args.data_path, train=False, transform=test_transform)
        args.num_classes = 100

    elif args.dataset == "cifar100_10":
        traindataset = CIFAR100(args.data_path, train=True, download=True,transform=train_transform,sample_id_path=args.sample_id_path)
        testdataset = CIFAR100(args.data_path, train=False, transform=test_transform)
        args.num_classes = 100
    
    elif args.dataset == "cifar10_1":
        traindataset = CIFAR10(args.data_path, train=True, download=True,transform=train_transform,sample_id_path=args.sample_id_path)
        testdataset = CIFAR10(args.data_path, train=False, transform=test_transform)
        args.num_classes = 10

    elif args.dataset == "cifar10_10":
        traindataset = CIFAR10(args.data_path, train=True, download=True,transform=train_transform,sample_id_path=args.sample_id_path)
        testdataset = CIFAR10(args.data_path, train=False, transform=test_transform)
        args.num_classes = 10


    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.data_path, 'train')
        valdir = os.path.join(args.data_path, 'val')
        traindataset = datasets.ImageFolder(traindir, transform=train_transform)
        testdataset = datasets.ImageFolder(valdir, transform=test_transform)
        args.num_classes = 1000
        
    train_loader = DataLoader(traindataset, batch_size=args.bs, shuffle=True, num_workers=8)
    test_loader = DataLoader(testdataset, batch_size=args.bs, shuffle=False, num_workers=8)
    return train_loader, test_loader


def build_imagenet_transform(trans, seed):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if trans == 'hflip':
        train_transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    elif trans == 'hflip_seed':
        print(trans, seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    elif trans == 'augment':
        train_transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandAugment(),
            transforms.ToTensor(),
            normalize
        ])
    elif trans == 'auto_aug':
        train_transform=transforms.Compose([\
            transforms.RandomResizedCrop(224),
            transforms.AutoAugment(), 
            transforms.ToTensor(), 
            normalize,
        ])
    elif trans == 'cutout':
        train_transform=transforms.Compose([\
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            Cutout(n_holes=1, length=32), 
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    return train_transform, test_transform
        

def build_transform(args, trans='', seed=None):
    if 'cifar100_244' in args.dataset:
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        img_size = [224, 224]
    
    elif 'cifar100' in args.dataset:
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        img_size = [32, 32]

    elif 'cifar10_224' in args.dataset:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        img_size = [224, 224]

    elif 'cifar10' in args.dataset:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        img_size = [32, 32]
    
    elif 'imagenet' in args.dataset:
        print(trans)
        return build_imagenet_transform(trans, seed)
        
    
    print(trans)
    if trans == 'augment':
        train_transform=transforms.Compose([\
            transforms.Resize(img_size), 
            transforms.RandomCrop(img_size[0], padding=4),
            transforms.RandAugment(), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std),
        ])
    elif trans == 'hflip':
        train_transform=transforms.Compose([\
            transforms.Resize(img_size), 
            transforms.RandomCrop(img_size[0], padding=4),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std),
        ])
    elif trans == 'hflip_seed':
        print(trans, seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_transform=transforms.Compose([\
            transforms.Resize(img_size), 
            transforms.RandomCrop(img_size[0], padding=4),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std),
        ])
    elif trans == 'hflip+rot_seed':
        print(trans, seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_transform=transforms.Compose([\
            transforms.Resize(img_size), 
            transforms.RandomCrop(img_size[0], padding=4),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(15), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std),
        ])
        
    elif trans == 'vflip':
        train_transform=transforms.Compose([\
            transforms.Resize(img_size), 
            transforms.RandomCrop(img_size[0], padding=4),
            transforms.RandomVerticalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std),
        ])
    elif trans == 'rot':
        train_transform=transforms.Compose([\
            transforms.Resize(img_size), 
            transforms.RandomCrop(img_size[0], padding=4),
            transforms.RandomRotation(15), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std),
        ])

    elif trans == 'hflip+rot':
        train_transform=transforms.Compose([\
            transforms.Resize(img_size), 
            transforms.RandomCrop(img_size[0], padding=4),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(15), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std),
        ])
    elif trans == 'auto_aug':
        train_transform=transforms.Compose([\
            transforms.Resize(img_size), 
            transforms.RandomCrop(img_size[0], padding=4),
            transforms.AutoAugment(), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std),
        ])
    elif trans == 'cutout':
        train_transform=transforms.Compose([\
            transforms.Resize(img_size), 
            transforms.RandomCrop(img_size[0], padding=4),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            Cutout(n_holes=1, length=16), 
            transforms.Normalize(mean, std),
        ])
    elif trans == 'cifar10_policy':
        train_transform=transforms.Compose([\
            transforms.Resize(img_size), 
            transforms.RandomCrop(img_size[0], padding=4),
            transforms.RandomHorizontalFlip(), 
            CIFAR10Policy(), 
            transforms.ToTensor(), 
            Cutout(n_holes=1, length=16), 
            transforms.Normalize(mean, std),
        ])
    
    elif trans == 'pers':
        train_transform=transforms.Compose([\
            transforms.Resize(img_size), 
            transforms.RandomCrop(img_size[0], padding=4),
            transforms.RandomPerspective(), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std),
        ])
    
    elif trans == 'hflip+noise':
        train_transform=transforms.Compose([\
            transforms.Resize(img_size), 
            transforms.RandomCrop(img_size[0], padding=4),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std),
        ])
        
    else:
        train_transform=transforms.Compose([\
            transforms.Resize(img_size), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std),
        ])

    test_transform = transforms.Compose([\
        transforms.Resize(img_size), 
        transforms.ToTensor(), 
        transforms.Normalize(mean, std),
    ])
    return train_transform, test_transform