import logging
import os
import sys
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def set_random_seed(seed=None):
    seed = 42 if seed==None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_free_device(args):
    def get_free_gpu():
        import gpustat
        stats = gpustat.GPUStatCollection.new_query()
        GPU_usage = {stats[ii].entry['index']: stats[ii].entry['memory.used'] for ii in range(len(stats))}
        bestGPU = sorted(GPU_usage.items(), key=lambda item: item[1])[0][0]
        print(f"setGPU: Setting GPU to: {bestGPU}, GPU_usage: {GPU_usage}")
        return bestGPU
    try:
        gid = args.gid
        if gid<0:
            args.gid = get_free_gpu()
    except:
        args.gid = get_free_gpu()
    return torch.device(f'cuda:{args.gid}') if torch.cuda.is_available() else torch.device('cpu')


def set_logger(args):
    format_str = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=format_str, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
    
    if args.log:
        formatter = logging.Formatter(format_str, "%Y-%m-%d %H:%M:%S")
        try:
            log_dir = args.log_dir
        except:
            log_dir = '../log'
        cur_time = time.strftime("%m.%d-%H:%M")
        log_file = os.path.join(log_dir, cur_time + f'_{args.log}.txt')
        
        fh = logging.FileHandler(log_file)
        # fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    cmd_input = 'python ' + ' '.join(sys.argv)
    logger.info(cmd_input)
    logger.info(args)
    return logger


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.MaxPool2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.AdaptiveAvgPool2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


class Initializer:
    """
    net = Model()  # instantiate the model

    # to apply xavier_uniform:
    Initializer.initialize(model=net, initialization=init.xavier_uniform, gain=init.calculate_gain('relu'))

    # or maybe normal distribution:
    Initializer.initialize(model=net, initialization=init.normal, mean=0, std=0.2)
    """
    def __init__(self):
        pass

    @staticmethod
    def initialize(model, initialization, **kwargs):

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.Linear):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

        model.apply(weights_init)

def initModel(model, method="normal"):
    """
    method = ['constant', 'uniform', 'normal', 
            'xavier_uniform', 'xavier_normal',
            'kaiming_uniform','kaiming_normal', 
            'orthogonal']
    """
    if method == 'constant':
        Initializer.initialize(model, init.constant_, val=0.3)
    elif method == 'uniform':
        Initializer.initialize(model, init.uniform_, a=0, b=1)
    elif method == 'normal':
        Initializer.initialize(model, init.normal_)
    elif method == 'xavier_uniform':
        Initializer.initialize(model, init.xavier_uniform_, gain=init.calculate_gain('relu'))
    elif method == 'xavier_normal':
        Initializer.initialize(model, init.xavier_normal_, gain=init.calculate_gain('relu'))
    elif method == 'kaiming_uniform':
        Initializer.initialize(model, init.kaiming_uniform_, mode='fan_out', nonlinearity='relu')
    elif method == 'kaiming_normal':
        Initializer.initialize(model, init.kaiming_normal_, mode='fan_out', nonlinearity='relu')
    elif method == 'orthogonal':
        Initializer.initialize(model, init.orthogonal_, gain=1)
    elif method == 'sparse':
        Initializer.initialize(model, init.sparse_, sparsity=0.1)


def retrieve_name_ex(var):
    frame = sys._getframe(2)
    while(frame):
        for item in frame.f_locals.items():
            if (var is item[1]):
                return item[0]
        frame = frame.f_back
    return ""

def myout(*para, threshold=10):
    def get_mode(var):
        if isinstance(var, (list, dict, set)):
            return 'len'
        elif isinstance(var, (np.ndarray, torch.Tensor)):
            return 'shape'
        else: return ''

    for var in para:
        name = retrieve_name_ex(var)
        mode = get_mode(var)
        if mode=='len':
            len_var = len(var)
            if isinstance(var, list) and len_var>threshold and threshold>6:
                print(f'{name} : len={len_var}, list([{var[0]}, {var[1]}, {var[2]}, ..., {var[-3]}, {var[-2]}, {var[-1]}])')
            elif isinstance(var, set) and len_var>threshold and threshold>6:
                var = list(var)
                print(f'{name} : len={len_var}, set([{var[0]}, {var[1]}, {var[2]}, ..., {var[-3]}, {var[-2]}, {var[-1]}])')
            elif isinstance(var, dict) and len_var>threshold and threshold>6:
                tmp = []
                for kk, vv in var.items():
                    tmp.append(f'{kk}: {vv}')
                    if len(tmp) > threshold: break
                print(f'{name} : len={len_var}, dict([{tmp[0]}, {tmp[1]}, {tmp[2]}, {tmp[3]}, {tmp[4]}, {tmp[5]}, ...])')
            else:
                print(f'{name} : len={len_var}, {var}')
        elif mode=='shape':
            sp = var.shape
            if len(sp)<2:
                print(f'{name} : shape={sp}, {var}')
            else:
                print(f'{name} : shape={sp}')
                print(var)
        else:
            print(f"{name} = {var}")
            

if __name__ == '__main__':
    from torchvision.models import resnet18
    a = resnet18()
    initModel(a)