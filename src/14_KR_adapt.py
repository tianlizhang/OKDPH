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
import warnings
warnings.filterwarnings("ignore")
from IPython import embed


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
    return round(100 * correct.item() / total, 4)


## loss
criterion_CELoss = nn.CrossEntropyLoss()
criterion_KLLoss = nn.KLDivLoss(reduction='batchmean')


def my_comp_loss(args, scores, tar, labels, gamma):
    xentropy_loss = criterion_CELoss(scores, labels)
    
    cur_batch_size = scores.shape[0]
    class_num = scores.shape[1]
    
    scores_mask = torch.ones(scores.shape, dtype=torch.bool).to(args.device)
    scores_mask = scores_mask.scatter_(1, labels.reshape((cur_batch_size,1)), 0)
    mask_scores = torch.reshape(torch.masked_select(scores, scores_mask), (cur_batch_size, class_num-1))
    
    tar_prob = F.softmax(tar, dim=1).detach()
    label_loss = criterion_KLLoss(F.log_softmax(mask_scores, dim=1), tar_prob)
    
    tau_mask_prob = F.softmax(mask_scores.detach() / args.tau, dim=1).detach()
    update_label_loss = criterion_KLLoss(F.log_softmax(tar, dim=1), tau_mask_prob)
    
    loss = xentropy_loss + label_loss * args.alpha * gamma + update_label_loss * args.beta
    return loss


class EmbeddingLayer(nn.Module):
    def __init__(self, num_classes, w2v_weight):
        super(EmbeddingLayer, self).__init__()
        self.emb = nn.Embedding(num_classes, num_classes-1)
        self.emb.weight = nn.Parameter(w2v_weight)
    
    def forward(self, x):
        return self.emb(x)
    

def train(args, nets, train_loader, opts, emb):
    for net in nets:
        net.train()

    train_loss, total, correct = np.zeros(len(nets)), 0, 0.
    for i, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs, labels = imgs.to(args.device), labels.to(args.device)

        for net_idx, net in enumerate(nets):
            out = net(imgs[:, net_idx, ...]) # [b, num_classes]
            tar = emb(labels)
            
            if args.gamma == True:
                gamma = 1-accuracy
            else:
                gamma = 1
            
            loss = my_comp_loss(args, out, tar, labels, gamma)
            opts[net_idx].zero_grad()
            loss.backward()
            opts[net_idx].step()
            
            train_loss[net_idx] += loss.item()
            if net_idx == 0:
                total += labels.size(0)
            
            pred = torch.max(out.data, 1)[1]
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total

    return list(np.round(train_loss/total, 4))



def main():
    train_loader, test_loader = data_loader.load_dataset(args, args.transes, args.trans_seeds)
    
    class_dist = np.load('glove/{}_class_{}_dist.npy'.format(args.dataset, args.distance))
    class_dist = torch.from_numpy(class_dist).to(args.device)
    if args.dataset == 'cifar10':
        class_mask = (1 - torch.eye(10)).byte().to(args.device)
        class_dist = (torch.masked_select(class_dist, class_mask)).reshape((10,9))
    elif args.dataset == 'cifar100':
        class_mask = (1 - torch.eye(100)).byte().to(args.device)
        class_dist = (torch.masked_select(class_dist, class_mask)).reshape((100,99))
        
    nets, opts, schs = [], [], []
    for name in args.model_names:
        model = model_dict[name](num_classes=args.num_classes).to(args.device)
        emb = EmbeddingLayer(args.num_classes, class_dist).to(args.device)
        
        optimizer = optim.SGD(list(model.parameters()) + list(emb.parameters()), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
        scheduler = MultiStepLR(optimizer, args.ms)
        nets.append(model)
        opts.append(optimizer)
        schs.append(scheduler)

    logger.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    metrics = {'best_acc': 0, 'test_acces': []}
    for epoch in range(0, args.epochs):
        metrics['train_losses'] = train(args, nets, train_loader, opts, emb)
        
        metrics['test_acces'] = []
        for nid, net in enumerate(nets):
            metrics['test_acces'].append(test(args, net, test_loader))
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
    parser.add_argument("--model_names", type=str, nargs='+', default=['resnet32'])
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
    # 6. Method
    parser.add_argument('--tau', default=2, type=float)
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha')
    parser.add_argument('--beta', type=float, default=1.0, help='beta')
    parser.add_argument('--gamma', action='store_true', default=False, help='gamma')
    parser.add_argument('--distance', default='l2')
    
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
    