import argparse
import os
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import sys

from utils import utils
from data import data_loader
from models import model_dict
# from torchmetrics.classification import MulticlassCalibrationError


def test(args, net, testloader, criterion_MCE=None):
    net.eval()
    with torch.no_grad():
        correct, mce_loss = 0, 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            # mce_loss += criterion_MCE(outputs, labels) * labels.size(0)
    return round(100 * correct.item() / total, 4)#, round(mce_loss.item() / total, 4)


## loss
criterion_CE = nn.CrossEntropyLoss()
criterion_KLD = nn.KLDivLoss(reduction='batchmean')

def train(args, nets, train_loader, opts):
    for net in nets:
        net.train()

    train_loss, total = np.zeros(len(nets)), 0
    for i, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs, labels = imgs.to(args.device), labels.to(args.device)
    
        out_list = []
        for net_idx, net in enumerate(nets):
            out = net(imgs[:, net_idx, ...]) # [b, num_classes]
            out_list.append(out)
        
        for net_idx, net in enumerate(nets):
            ce_loss = criterion_CE(out_list[net_idx], labels)
            KLD_loss = 0
            for l in range(len(nets)):
                if l != net_idx:
                    KLD_loss += criterion_KLD(\
                        F.log_softmax(out_list[net_idx],dim=1), F.softmax(out_list[l],dim=1).detach())
        
            loss = ce_loss + KLD_loss / (len(nets)-1)

            opts[net_idx].zero_grad()
            loss.backward()
            opts[net_idx].step()

            train_loss[net_idx] += loss.item()
            if net_idx == 0:
                total += labels.size(0)

    return list(np.round(train_loss/total, 4))



def main():
    train_loader, test_loader = data_loader.load_dataset(args, args.transes, args.trans_seeds)
    
    # criterion_MCE = MulticlassCalibrationError(num_classes=args.num_classes, n_bins=args.num_classes, norm='l1')
    
    nets, opts, schs = [], [], []
    for name in args.model_names:
        model = model_dict[name](num_classes=args.num_classes).to(args.device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
        scheduler = MultiStepLR(optimizer, args.ms)
        nets.append(model)
        opts.append(optimizer)
        schs.append(scheduler)

    logger.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    metrics = {'best_acc': 0}
    for epoch in range(0, args.epochs):
        # if epoch > 0:
        metrics['train_losses'] = train(args, nets, train_loader, opts)
        
        metrics['test_acces'], metrics['test_mce'] = [], []
        for nid, net in enumerate(nets):
            metrics['test_acces'].append(test(args, net, test_loader))
            # acc, mce = test(args, net, test_loader, criterion_MCE)
            # metrics['test_acces'].append(acc)
            # metrics['test_mce'].append(mce)
            schs[nid].step()
        
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
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--data_path", default="../dataset", type=str, help="dataset path")
    parser.add_argument('--transes', nargs='+', default=[])
    # 2. Model & Loss
    parser.add_argument("--model_names", type=str, nargs='+', default=['resnet32', 'resnet32'])
    parser.add_argument("--pretrain", type=int, default=0)
    # 3. Train Setting
    parser.add_argument("--gid", type=int, default=-1)
    parser.add_argument("--runs", default=1, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--bs", default=128, type=int)
    parser.add_argument("--lr", default= 0.1, type=float, help="Learning rate")
    parser.add_argument("--mom", type=float, default=0.9, help="momentum")
    parser.add_argument("--wd", type=float, default=5e-4, help="weight_decay")
    parser.add_argument("--ms", nargs='+', default=[150, 225], help="Milestones")
    # 4. Log and Save
    parser.add_argument("--log", type=str, default='')
    parser.add_argument("--log_dir", type=str, default='../log')
    parser.add_argument("--save", default="", type=str)
    # 5. Experiment: Noisy and Limited data
    parser.add_argument("--noise_mean", default=0.0, type=float)
    parser.add_argument("--noise_std", default=1.0, type=float)
    parser.add_argument("--sample_path", default="../experiment/sample", type=str, help="sample 10, 1 percent id path")
    
    args = parser.parse_args()
    utils.set_random_seed(seed=42)
    args.device = utils.get_free_device(args)
    
    if len(args.transes) == 0:
        if 'cifar100' in args.dataset:
            args.transes = ['hflip+rot']*len(args.model_names)
        elif 'cifar10' in args.dataset:
            args.transes = ['hflip']*len(args.model_names)
        elif 'imagenet' in args.dataset:
            args.transes = ['hflip']*len(args.model_names)
    args.trans_seeds = [ii for ii in range(len(args.transes))]

    if args.save:
        if not os.path.exists(args.save):
            os.mkdir(args.save)
        args.log_dir = args.save
    
    logger = utils.set_logger(args)
    
    if args.sample_path:
        args.sample_id_path = os.path.join(args.sample_path, args.dataset+'.pkl')
        
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
    