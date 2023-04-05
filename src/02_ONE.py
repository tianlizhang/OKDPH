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


class KL_Loss(nn.Module):
    def __init__(self, temperature = 1):
        super(KL_Loss, self).__init__()
        self.T = temperature
    def forward(self, output_batch, teacher_outputs):
        output_batch = F.log_softmax(output_batch/self.T, dim = 1)    
        teacher_outputs = F.softmax(teacher_outputs/self.T, dim = 1) + 10**(-7)
    
        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs) 
        return loss
    

def get_current_consistency_weight(current, rampup_length):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    

def lookup(model_name):
    if model_name == "resnet8" or model_name == "resnet14" or model_name == "resnet20" or model_name == "resnet32":
        input_channel = 64
    elif model_name == "densenetd40k12":
        input_channel = 132
    elif model_name == "densenetd100k12":
        input_channel = 342
    elif model_name == "densenetd100k40":
        input_channel = 1126
    elif model_name == "resnet110":
        input_channel = 256
    elif model_name == "vgg16" or model_name == "resnet34":
        input_channel = 512
    elif model_name == "wide_resnet20_8" or model_name == "wide_resnet28_10":
        input_channel = 256
    # imagenet
    elif model_name == "shufflenet_v2_x1_0": 
        input_channel = 1024
    return input_channel

    
def test(args, net, testloader):
    net.eval()
    with torch.no_grad():
        correct = torch.zeros(args.num_branches+1).to(args.device)
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(args.device), labels.to(args.device)
            
            output_batch, x_m = net(images)
            for i in range(args.num_branches):
                _, predicted = torch.max(output_batch[:, :, i].data, 1)
                correct[i] += (predicted == labels).sum()
            
            _, predicted = torch.max(torch.mean(output_batch, dim=2).data, 1)
            correct[args.num_branches] += (predicted == labels).sum()
            
            total += labels.size(0)
    return list(np.round(100 * correct.cpu().numpy() / total, 4))



def train(args, nets, train_loader, opts, consistency_weight, criterion, criterion_T):
    for net in nets:
        net.train()
    
    train_loss, total = np.zeros(len(nets)), 0
    for i, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        
        output_batch, x_m  = nets[0](imgs)
        loss_true = 0
        loss_group = 0    
        for i in range(args.num_branches):
            loss_true += criterion(output_batch[:,:,i], labels)
            loss_group += criterion_T(output_batch[:,:,i], x_m)
        loss_true += criterion(x_m, labels)
        
        loss = loss_true + args.alpha * consistency_weight * loss_group

        opts[0].zero_grad()
        loss.backward()
        opts[0].step()
        
        train_loss[0] += loss.item()
        total += labels.size(0)
    return list(np.round(train_loss/total, 4))



def main():
    train_loader, test_loader = data_loader.load_dataset(args, args.transes, args.trans_seeds)
    
    nets, opts, schs = [], [], []
    for name in args.model_names:
        model = model_dict[name](num_classes=args.num_classes, \
            num_branches = args.num_branches).to(args.device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
        scheduler = MultiStepLR(optimizer, args.ms)
        nets.append(model)
        opts.append(optimizer)
        schs.append(scheduler)
        
    logger.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    criterion = nn.CrossEntropyLoss()
    criterion_T = KL_Loss(args.temperature).to(args.device)

    metrics = {'best_acc': 0, 'test_acces': []}
    for epoch in range(0, args.epochs):
        # Set consistency_weight or originial temperature scale 
        consistency_epoch = args.start_consistency * args.epochs 
        if epoch < consistency_epoch:
            consistency_weight = 1
        else:
            consistency_weight = get_current_consistency_weight(epoch - consistency_epoch, args.length)
            
        metrics['train_losses'] = train(args, nets, train_loader, opts, consistency_weight, criterion, criterion_T)
        metrics['test_acces'] = test(args, nets[0], test_loader)
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
    parser.add_argument("--model_names", type=str, nargs='+', default=['resnet32_one']) # densenet-40-12_one
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
    parser.add_argument('--num_branches', default=2, type=int, help = 'Input the number of branches: default(4)')
    parser.add_argument('--temperature', default=3.0, type=float, help = 'Input the temperature: default(3.0)')
    parser.add_argument('--alpha', default=1.0, type=float, help = 'Input the relative rate: default(1.0)')
    parser.add_argument('--start_consistency', default=0., type=float, help = 'Input the start consistency rate: default(0.5)')
    parser.add_argument('--length', default=80, type=float, help='length ratio: default(80)')
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
    args.trans_seeds = 0
        
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