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


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes


def test(args, net, testloader, criterion_MCE=None):
    net.eval()
    with torch.no_grad():
        correct, mce_loss = 0, 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = net(images)
            
            if len(outputs) == 2:
                outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            # mce_loss += criterion_MCE(outputs, labels) * labels.size(0)
    return round(100 * correct.item() / total, 4)#, round(mce_loss.item() / total, 4)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
        
global_step = 0

def train(args, nets, train_loader, opts, epoch):
    global global_step
    
    class_criterion = nn.CrossEntropyLoss()
    if args.consistency_type == 'mse':
        consistency_criterion = softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = symmetric_mse_loss
    
    for net in nets:
        net.train()

    train_loss, total = np.zeros(len(nets)), 0
    for i, (imgs, labels) in enumerate(tqdm(train_loader)):
        # adjust_learning_rate(opts[0], epoch, i, len(train_loader))
        minibatch_size = len(labels)
        
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        model_out = nets[0](imgs[:, 0, ...])
        ema_model_out = nets[1](imgs[:, 1, ...])
        
        if len(model_out) == 2:
            class_logit, cons_logit = model_out
            ema_logit, _ = ema_model_out
            res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
        else:
            class_logit, cons_logit = model_out, model_out
            ema_logit = ema_model_out
            res_loss = 0
        
        class_loss = class_criterion(class_logit, labels)# / minibatch_size
        # ema_class_loss = class_criterion(ema_logit, labels) / minibatch_size
        
        if args.consistency: # 100
            consistency_weight = get_current_consistency_weight(epoch)
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
        else:
            consistency_loss = 0
        
        loss = class_loss + consistency_loss + res_loss
        opts[0].zero_grad()
        loss.backward()
        opts[0].step()
        global_step += 1
        update_ema_variables(nets[0], nets[1], args.ema_decay, global_step) # ema_decay=0.99
        
        train_loss[0] += loss.item()
        total += labels.size(0)

    return list(np.round(train_loss/total, 4))



def main():
    train_loader, test_loader = data_loader.load_dataset(args, args.transes, args.trans_seeds)
    
    nets, opts, schs = [], [], []
    ema_list = [False, True] # [model, ema_model]
    for ii, name in enumerate(args.model_names):
        model = model_dict[name](num_classes=args.num_classes).to(args.device)

        if ema_list[ii]:
            for param in model.parameters():
                param.detach_()
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
            opts.append(optimizer)
            scheduler = MultiStepLR(optimizer, args.ms)
            schs.append(scheduler)
        nets.append(model)
        
    logger.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    metrics = {'best_acc': 0}
    for epoch in range(0, args.epochs):
        metrics['train_losses'] = train(args, nets, train_loader, opts, epoch)
        
        metrics['test_acces'] = []
        for nid, net in enumerate(nets):
            metrics['test_acces'].append(test(args, net, test_loader))
            if not ema_list[nid]:
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
    parser.add_argument("--model_names", type=str, nargs='+', default=['resnet32_ema', 'resnet32_ema'])
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
    # Method
    parser.add_argument("--consistency_type", default='mse', type=str)
    parser.add_argument("--logit_distance_cost", default=0.01, type=float)
    parser.add_argument("--consistency", default=10, type=float)
    parser.add_argument("--ema_decay", default=0.99, type=float)
    parser.add_argument("--consistency_rampup", default=5, type=float)
    
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
    