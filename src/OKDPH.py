import argparse
import os
import torch
import numpy as np
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from utils import utils
from data import data_loader
from models import model_dict
import sys
import copy


def test(args, net, testloader):
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    return round(100 * correct.item() / total, 2)


def update_hwm(nets, hwm):
    N = len(nets[0:-1]) # num of students
    if hwm == 'dir':
        rr = np.random.dirichlet([[1 for _ in range(N)]])
    elif hwm == 'mean':
        rr = 1./N
        
    for j, net in enumerate(nets[0:-1]):
        if j == 0:
            hybrid_weight = {k : v * (rr) for k, v in net.state_dict().items()}
        else:
            hybrid_weight = {k : v * (rr) + hybrid_weight[k] for k, v in net.state_dict().items()}
            
    nets[-1].load_state_dict(hybrid_weight)
    return nets[-1]


def fusion(net, hwm, gamma):
    if gamma == 1:
        net.load_state_dict(hwm.state_dict())
    elif gamma == 0:
        return net
    else:
        fusion_weight = {k : v*gamma + net.state_dict()[k]*(1.0-gamma) for k, v in hwm.state_dict().items()}
        net.load_state_dict(fusion_weight)
    return net


def train(args, nets, train_loader, opts):
    for net in nets:
        net.train()

    train_loss, total = np.zeros(len(nets)), 0
    for ii, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        nets[-1] = update_hwm(nets, args.hwm)
        
        out_list = []
        for net_idx, net in enumerate(nets):
            out = net(imgs[:, net_idx, ...]) # [b, num_classes]
            out_list.append(out)

        stable_out = torch.stack(out_list).mean(dim=0).detach() # [n, b, num_classes] - > [b, num_classes]
        loss_hwm = F.cross_entropy(out_list[-1], labels)
        
        for net_idx, net in enumerate(nets):
            if net_idx == len(nets) - 1:
                break
            loss_ce = F.cross_entropy(out_list[net_idx], labels)
            loss_kd = F.kl_div(
                F.log_softmax(out_list[net_idx] / args.T, dim=1),
                F.softmax(stable_out / args.T, dim=1),
                reduction='batchmean'
            ) * args.T * args.T

            loss = args.omega * loss_ce + (1-args.omega) *loss_hwm + args.beta * loss_kd

            opts[net_idx].zero_grad()
            if net_idx < len(nets) - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            opts[net_idx].step()

            train_loss[net_idx] += loss.item()
            if net_idx == 0:
                total += labels.size(0)

            if args.level and args.level == 'batch':
                if ii % args.gap == 0:
                    net = fusion(net, nets[-1], args.gamma)
                        
    return list(np.round(train_loss/total, 4))



def main():
    train_loader, test_loader = data_loader.load_dataset(args, args.transes, args.trans_seeds)
    
    nets, opts, schs = [], [], []
    # args.model_names.append(args.model_names[0]) # the HWM model 
    for name in args.model_names:
        model = model_dict[name](num_classes=args.num_classes).to(args.device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
        scheduler = MultiStepLR(optimizer, args.ms)
        if not args.diff_init: # the same init of models
            if len(nets)>0:
                model.load_state_dict(nets[0].state_dict())
        nets.append(model)
        opts.append(optimizer)
        schs.append(scheduler)
    nets.append(copy.deepcopy(nets[0]).to(args.device))  # the HWM model
    
    metrics = {'best_acc': 0}
    for epoch in range(0, args.epochs):
        metrics['train_losses'] = train(args, nets, train_loader, opts)
        
        metrics['test_acces'] = []
        for nid, net in enumerate(nets):
            metrics['test_acces'].append(test(args, net, test_loader))
            if nid < len(nets) - 1:
                schs[nid].step()

        if args.level and args.level == 'epoch':
            if epoch % args.gap == 0:
                logger.info(f'Conduct fusion with ratio {args.gamma} every {args.gap} {args.level}.')
                for net in nets[0:-1]:
                    net = fusion(net, nets[-1], args.gamma)

        best_acc = max(metrics['test_acces'])
        if best_acc  > metrics['best_acc']:
            metrics['best_acc'] = best_acc
        
        to_print = f'epoch: {epoch}, metrics: {metrics}, lr: {opts[0].param_groups[0]["lr"]}'
        logger.info(to_print)

        if args.save:
            for nid, net in enumerate(nets):
                torch.save(net.state_dict(), os.path.join(args.save, f'net{nid}_{epoch}ep.pkl'))
    return metrics['best_acc']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 1. Dataset
    parser.add_argument("--dataset", default="cifar10", type=str, help='dataset name')
    parser.add_argument("--data_path", default="../dataset", type=str, help="dataset path")
    parser.add_argument('--transes', nargs='+', default=[], help="multi transforms")
    # 2. Model
    parser.add_argument("--model_names", type=str, nargs='+', default=['resnet32', 'resnet32'])
    parser.add_argument("--pretrain", type=int, default=0)
    parser.add_argument("--diff_init", action='store_true')
    # 3. Train Setting
    parser.add_argument("--gid", type=int, default=-1)
    parser.add_argument("--runs", default=1, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--bs", default=128, type=int, help='batch_size')
    parser.add_argument("--lr", default= 0.1, type=float, help="Learning rate")
    parser.add_argument("--mom", type=float, default=0.9, help="momentum")
    parser.add_argument("--wd", type=float, default=5e-4, help="weight_decay")
    parser.add_argument("--ms", nargs='+', default=[150, 225], help="Milestones")
    # 4. Log and Save
    parser.add_argument("--log", type=str, default='')
    parser.add_argument("--log_dir", type=str, default='../log')
    parser.add_argument("--save", default="", type=str)
    # 5. Hyper-Parameters
    parser.add_argument("--omega", default=0.8, type=float)
    parser.add_argument("--beta", default=0.8, type=float, help='KD loss weight')
    parser.add_argument("--gamma", default=0.5, type=float, help='Fusion ratio')
    parser.add_argument("--hwm", default='mean', type=str, choices=['mean', 'dir'], help='The construction type of the HWM: Mean or Dirichlet')
    parser.add_argument("--interval", default='1_epoch', type=str, help='Fusion Interval, format: "" or "1_epoch" or "5_batch"')
    parser.add_argument('--T', type=float, default=4.0, help='temperature')
    # 6. Experiment: Noisy and Limited data
    parser.add_argument("--noise_mean", default=0.0, type=float, help='Mean of noise')
    parser.add_argument("--noise_std", default=1.0, type=float)
    parser.add_argument("--sample_path", default="../experiment/sample", type=str, help="sample 10, 1 percent id path")
    
    args = parser.parse_args()
    utils.set_random_seed(seed=42)
    args.device = utils.get_free_device(args)
    
    if len(args.transes) == 0:
        if 'cifar100' in args.dataset:
            args.transes = ['hflip+rot', 'cutout', 'augment']
        elif 'cifar10' in args.dataset:
            args.transes = ['hflip', 'cutout', 'augment']
        elif 'imagenet' in args.dataset:
            args.transes = ['hflip', 'augment', 'auto_aug']
    args.trans_seeds = [ii for ii in range(len(args.transes))]
    assert len(args.transes) == len(args.model_names) + 1 # one more transform for the HWM
    
    if args.interval:
        try:
            args.gap, args.level = args.interval.split('_')
            args.gap = int(args.gap)
            assert args.level == 'epoch' or args.level == 'batch'
            print(f'Conduct fusion with ratio {args.gamma} every {args.gap} {args.level}.')
        except:
            raise ValueError('args.interval must be like "" or "1_epoch" or "5_batch".')
    else:
        args.level = ''
        print('Do not conduct the fusion of students and the HWM.')
        
    if args.save:
        if not os.path.exists(args.save):
            os.mkdir(args.save)
        args.log_dir = args.save
    
    if args.sample_path:
        args.sample_id_path = os.path.join(args.sample_path, args.dataset+'.pkl')
        
    logger = utils.set_logger(args)   
    
    acces = []
    for rid in range(args.runs):
        logger.info(f'Run: {rid} -------------------------------------')
        acc = main()
        acces.append(acc)
        
    mean, std = np.mean(acces), np.std(acces)
    mean, std = np.round(mean, 2), np.round(std, 2)
    logger.info(f'best_acces: {acces}, mean: {mean}, std: {std}')
    
    cmd_input = 'python ' + ' '.join(sys.argv)
    logger.info(cmd_input)
    