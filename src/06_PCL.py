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
from typing import List
import math
import sys

from utils import utils
from data import data_loader
from models import model_dict


class CrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, input, label):
        loss = self.loss_func(input, label)
        return loss
    

class KDLoss(nn.Module):
    def __init__(self, temperature=3.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.func = nn.KLDivLoss(reduction='batchmean')


    def forward(self, pred, target):
        """_summary_

        Parameters
        ----------
        pred : _type_  [N, C]
            _description_
        target : _type_  [N, C]
            _description_
        """
        loss = self.func(F.log_softmax(pred / self.temperature, dim=1), F.softmax(target / self.temperature, dim=1)) * self.temperature * self.temperature
        return loss
    

class PMLoss(nn.Module):
    def __init__(self, temperature=3.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.kd = KDLoss(temperature=self.temperature)

    def forward(self, pred, targets):
        loss = 0
        for i in range(len(targets)):
            target = targets[i].detach()
            pmloss = self.kd(pred, target)
            loss += pmloss
        return loss / len(targets)
    

class LossWrapper(nn.Module):
    def __init__(self, loss_func_list: List[nn.Module], loss_weight_list: List[float], num_branches,  *args, **kwargs) -> None:
        super(LossWrapper, self).__init__()
        self.loss_func_list = loss_func_list
        self.loss_weight_list = loss_weight_list
        assert len(self.loss_func_list) == len(self.loss_weight_list), "length of loss function list should match the length of loss weight list"
        self.num_meter = len(self.loss_func_list)
        if len(self.loss_func_list) == 1:
            self.loss_weight_list = [1.0]
        
        self.ce, self.kd, self.pm = self.loss_func_list
        self.num_branches = num_branches
    
    def compute_rampup_weight(self, epoch, lambd=1.0, alpha=80):
        if epoch > alpha:
            return lambd
        else:
            return lambd * math.exp(-5 * (1 - epoch / lambd) ** 2)

    def forward(self, preds, pred_en, mean_preds, label, epoch, *args, **kwargs):
        if not kwargs['ema']:
            
            loss = self.ce(pred_en, label)
            rampup_weight = self.compute_rampup_weight(epoch)

            for bid in range(self.num_branches):
                loss_ce = self.ce(preds[bid], label)
                loss_pe = self.kd(preds[bid], pred_en.detach()) * rampup_weight
                loss_pm = self.pm(preds[bid], [mean_preds[jj] for jj in range(len(mean_preds)) if jj != bid])
                
                loss += loss_ce + loss_pe + loss_pm * rampup_weight
            return loss

        else:
            loss = 0
            for bid in range(self.num_branches):
                loss += self.ce(preds[bid], label).detach()
            return loss, preds
        


def update_ema_variables(model, ema_model, alpha=0.999, global_step=-1):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        


def test(args, net, testloader, epoch, loss_wrapper):
    net.eval()
    with torch.no_grad():
        correct = torch.zeros(args.num_branches).to(args.device)
        total = 0
        for data in testloader:
            imgs, labels = data
            imgs, labels = imgs.to(args.device), labels.to(args.device)

            mean_out = net(torch.stack([imgs for _ in range(args.num_branches)], dim=1))
            for bid in range(args.num_branches):
                _, predicted = torch.max(mean_out[bid].data, 1)
                correct[bid] += (predicted == labels).sum()
            # img1, img2, img3 = imgs[:, 0, ...].contiguous(), imgs[:, 1, ...].contiguous(), imgs[:, 2, ...].contiguous()
            # output1, output2, output3 = net(imgs, imgs, imgs)
            # loss, loss_tuple, outputs_no_grad = loss_wrapper([output1, output2, output3], [labels], ema=True, epoch=epoch)
            
            # _, predicted1 = torch.max(outputs_no_grad[0].data, 1)
            # _, predicted2 = torch.max(outputs_no_grad[1].data, 1)
            # _, predicted3 = torch.max(outputs_no_grad[2].data, 1)

            # correct[0] += (predicted1 == labels).sum()
            # correct[1] += (predicted2 == labels).sum()
            # correct[2] += (predicted3 == labels).sum()
            
            total += labels.size(0)
    return list(np.round(100 * correct.cpu().numpy() / total, 4))


def train(args, nets, train_loader, opts, epoch, loss_wrapper):
    for net in nets:
        net.train()
    
    train_loss, total = np.zeros(len(nets)), 0
    for i, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        
        out_list, en_out = nets[0](imgs)
        mean_outs = nets[1](imgs)
        
        loss = loss_wrapper(out_list, en_out, mean_outs, labels, epoch=epoch, ema=False)
        
        opts[0].zero_grad()
        loss.backward()
        opts[0].step()
        
        train_loss[0] += loss.item()
        total += labels.size(0)
        
        update_ema_variables(nets[0], nets[1], alpha=0.999,  global_step=0)
    return list(np.round(train_loss/total, 4))



def main():
    train_loader, test_loader = data_loader.load_dataset(args, args.transes, args.trans_seeds)
    
    nets, opts, schs = [], [], []
    emas = [False, True] # [model, mean_model]
    for ii, name in enumerate(args.model_names):
        model = model_dict[name](num_classes=args.num_classes, \
            ema=emas[ii], num_branches=args.num_branches).to(args.device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
        scheduler = MultiStepLR(optimizer, args.ms)
        nets.append(model)
        opts.append(optimizer)
        schs.append(scheduler)

    logger.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    loss_funcs = [CrossEntropyLoss(), KDLoss(), PMLoss()]
    loss_weights = [1.0, 1.0, 1.0]     
    loss_wrapper = LossWrapper(loss_funcs, loss_weights, num_branches=args.num_branches)
    
    metrics = {'best_acc': 0, 'test_acces': []}
    for epoch in range(0, args.epochs):
        metrics['train_losses'] = train(args, nets, train_loader, opts, epoch, loss_wrapper)
        
        metrics['test_acces'] = test(args, nets[1], test_loader, epoch, loss_wrapper)
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
    parser.add_argument("--model_names", type=str, nargs='+', default=['resnet32_pcl', 'resnet32_pcl']) # model and mean model
    parser.add_argument("--pretrain", type=int, default=0) # 'densenet-40-12_pcl'
    # 3. Train Setting
    parser.add_argument("--gid", type=int, default=-1)
    parser.add_argument("--runs", default=1, type=int)
    parser.add_argument("--epochs", default=300, type=int, help='Num of Epochs')
    parser.add_argument("--bs", default=128, type=int, help='Batch size')
    parser.add_argument("--lr", default= 0.1, type=float, help="Learning rate")
    parser.add_argument("--mom", type=float, default=0.9, help="momentum")
    parser.add_argument("--wd", type=float, default=5e-4, help="weight_decay")
    parser.add_argument("--ms", nargs='+', default=[150, 225], help="Milestones")
    # 4. Log and Save
    parser.add_argument("--log", type=str, default='')
    parser.add_argument("--log_dir", type=str, default='../log')
    parser.add_argument("--save", default="", type=str)
    # 5. Hype-Paras
    parser.add_argument("--num_branches", default=3, type=int)
    # 6. Experiment: Noisy and Limited data
    parser.add_argument("--noise_mean", default=0.0, type=float)
    parser.add_argument("--noise_std", default=1.0, type=float)
    parser.add_argument("--sample_path", default="../experiment/sample", type=str, help="sample 10, 1 percent id path")
   
    args = parser.parse_args()
    utils.set_random_seed(seed=42)
    args.device = utils.get_free_device(args)
    
    if len(args.transes) == 0:
        if 'cifar100' in args.dataset:
            args.transes = ['hflip+rot'] * args.num_branches
        elif 'cifar10' in args.dataset:
            args.transes = ['hflip'] * args.num_branches
        elif 'imagenet' in args.dataset:
            args.transes = ['hflip'] * args.num_branches

    args.trans_seeds = [ii for ii in range(len(args.transes))]

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