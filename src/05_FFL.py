import argparse
import os
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch.autograd import Variable
import sys

from utils import utils
from data import data_loader
from models import model_dict


class KLLoss(nn.Module):
    def __init__(self, device):
        self.device = device
        super(KLLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.to(self.device),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss
    

def get_current_consistency_weight(current, rampup_length):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    
def test(args, model, module, testloader):
    model.eval()
    module.eval()
    with torch.no_grad():
        correct = torch.zeros(args.num_branches+1).to(args.device)
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(args.device), labels.to(args.device)
            
            out_list, fmap = model(images)
            fused_logit = module(fmap)
            
            for bid, out in enumerate(out_list):
                _, predicted_sub = torch.max(out.data, 1)
                correct[bid] += (predicted_sub == labels).sum()
            
            _, predicted_fused = torch.max(fused_logit.data, 1)
            correct[args.num_branches] += (predicted_fused == labels).sum()

            total += labels.size(0)
    return list(np.round(100 * correct.cpu().numpy() / total, 4))



def train(args, nets, train_loader, opts, consistency_weight, criterion_CE, criterion_kl):
    for net in nets:
        net.train()
    
    train_loss, total = np.zeros(len(nets)), 0
    for i, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs, labels = imgs.to(args.device), labels.to(args.device)
            
        out_list, fmap = nets[0](imgs)
        fused_logit = nets[1](fmap)
        
        for bid, out in enumerate(out_list):
            ensemble_logit = out if bid==0 else ensemble_logit + out
            loss_ce = criterion_CE(out, labels) if bid==0 else loss_ce + criterion_CE(out, labels)
            loss_kl = criterion_kl(out, fused_logit) if bid==0 else loss_kl + criterion_kl(out, fused_logit)
            
        ensemble_logit /= len(out_list)
        loss_ce += criterion_CE(fused_logit, labels)
        
        loss_kl += criterion_kl(fused_logit, ensemble_logit)
        loss_kl *= consistency_weight
        
        loss = loss_ce + loss_kl
        opts[0].zero_grad()
        opts[1].zero_grad()
        loss.backward()
        opts[0].step()
        opts[1].step()
        
        train_loss[0] += loss.item()
        total += labels.size(0)
    return list(np.round(train_loss/total, 4))



def main():
    train_loader, test_loader = data_loader.load_dataset(args, args.transes, args.trans_seeds)
    
    nets, opts, schs = [], [], []
    param_size = 0
    channel_dict = {'resnet32_ffl': 64, 'resnet110_ffl': 256, 'wrn_20_8_ffl': 256, 'vgg16_ffl': 512, 'densenet-40-12_ffl': 132}
    for name in args.model_names:
        if 'fm' in name:
            model = model_dict[name](num_classes=args.num_classes,  channel = channel_dict[args.model_names[0]], 
                                    num_branches = args.num_branches
                                    ).to(args.device)
        else:
            model = model_dict[name](num_classes=args.num_classes,
                                num_branches = args.num_branches
                                ).to(args.device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
        scheduler = MultiStepLR(optimizer, args.ms)
        nets.append(model)
        opts.append(optimizer)
        schs.append(scheduler)
        param_size += sum(p.numel() for p in model.parameters()) / 1000000.0
    logger.info('Total params: %.2fM' % param_size)
    
    criterion_CE = nn.CrossEntropyLoss()
    criterion_kl = KLLoss(args.device)

    metrics = {'best_acc': 0, 'test_acces': []}
    metrics['test_acces'] = test(args, nets[0], nets[1], test_loader)
    print(metrics)
    for epoch in range(0, args.epochs):
        consistency_weight = get_current_consistency_weight(epoch, args.consistency_rampup)
            
        metrics['train_losses'] = train(args, nets, train_loader, opts, consistency_weight, criterion_CE, criterion_kl)
        metrics['test_acces'] = test(args, nets[0], nets[1], test_loader)
        schs[0].step()
        
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
    parser.add_argument("--model_names", type=str, nargs='+', default=['resnet32_ffl', 'resnet_fm']) # ['vgg16_ffl' 'vgg_fm']
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
    # 5. Hype-Paras
    parser.add_argument("--num_branches", default=2, type=int)
    parser.add_argument('--consistency_rampup', '--consistency_rampup', default=80, type=float,
                    metavar='consistency_rampup', help='consistency_rampup ratio')
    # 6. Experiment: Noisy and Limited data
    parser.add_argument("--sample_path", default="../experiment/sample", type=str, help="sample 10, 1 percent id path")
    parser.add_argument("--noise_mean", default=0.0, type=float)
    parser.add_argument("--noise_std", default=1.0, type=float)
    
    args = parser.parse_args()
    utils.set_random_seed(seed=42)
    args.device = utils.get_free_device(args)
    
    if len(args.transes) == 0:
        if 'cifar100' in args.dataset:
            args.transes = 'hflip+rot'
        elif 'cifar10' in args.dataset:
            args.transes = 'hflip'
        elif 'imagenet' in args.dataset:
            args.transes = 'hflip'
    args.trans_seeds = [0, 1]

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