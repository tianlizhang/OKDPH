import argparse
import numpy as np
import torch
import os, sys
from tqdm import tqdm, trange
from sklearn.decomposition import PCA

import torch.nn.functional as F
import matplotlib.pylab as plt
import pickle as pkl

sys.path.append('../../src')
from models import model_dict
from data import data_loader
from utils.utils import myout, get_free_device


def load_state(args):
    models_files = []
    models_split = [0]
    for model_folder in args.model_folder:
        for prefix in args.prefix:
            for epoch in trange(args.start_epoch, args.max_epoch, args.save_epoch, desc=f'{model_folder}/{prefix}'):
            # for epoch in args.epochs:
                model_file = model_folder + '/' + prefix + str(epoch) + args.suffix
                assert os.path.exists(model_file), 'model %s does not exist' % model_file
                models_files.append(model_file)
            models_split.append(len(models_files))
    myout(models_files, models_split)

    net = model_dict[args.model](num_classes=args.num_classes).to(args.device)
    return args, models_files, models_split, net


def test(args, net, testloader):
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            
            loss += F.cross_entropy(outputs, labels) * labels.size(0)
    return round(100 * correct.item() / total, 2), round(loss.item() / total, 4)


def get_1d_weight(net_or_dict, diff_keys, net):
    if not isinstance(net_or_dict, dict):
        net_or_dict = net.state_dict()
    
    para, diff = [], []
    for kk, vv in net_or_dict.items():
        if kk in diff_keys:
            diff.append(vv.flatten())
        else:
            para.append(vv.flatten())
    return torch.cat(para), torch.cat(diff)


def save_diff(args, models_files, net):
    dict_keys = set([kk for kk in net.state_dict().keys()])
    names = set([ name for name, para in net.named_parameters()])
    diff_keys = list(dict_keys - names)

    para_lst, diff_lst = [], []
    for mpath in tqdm(models_files):
        para, diff = get_1d_weight(torch.load(mpath, map_location=args.device), diff_keys, net)
        para_lst.append(para.cpu().detach())
        diff_lst.append(diff.cpu().detach())
    
    matrix = torch.stack(para_lst).cpu().detach().numpy()
    diff_save =  torch.stack(diff_lst).cpu().detach().numpy()
    return matrix, diff_save, diff_keys



def vec2net(net, para, diff, diff_keys):
    sdict, left1, left2 = {}, 0, 0
    for kk, vv in net.state_dict().items():
        num = vv.numel()
        if kk in diff_keys:
            sdict[kk] = diff[left1:left1+num].view(vv.shape)
            left1 += num
        else:
            sdict[kk] = para[left2:left2+num].view(vv.shape)
            left2 += num
    net.load_state_dict(sdict)
    return net


def draw(coord):
    plt.figure()
    clst = ['indigo', 'deeppink', 'darkgreen', 'darkorange']
    plt.plot(coord[120:180, 0], coord[120:180, 1], marker='o', c=clst[0])
    plt.plot(coord[180:240, 0], coord[180:240, 1], marker=',', c=clst[1])
    plt.plot(coord[0:60, 0], coord[0:60, 1], marker='.', c=clst[2])
    plt.plot(coord[60:120, 0], coord[60:120, 1], marker='.', c=clst[3])

    # plt.xlim([-40, 45])
    # plt.ylim([-30, 40])
    plt.xlim([args.x_tuple[0], args.x_tuple[1]])
    plt.ylim([args.y_tuple[0], args.y_tuple[1]])

    plt.xlabel('Parameter1')
    plt.ylabel('Parameter2')
    l = plt.legend(['Ours-S1', 'Ours-S2', f'{args.method}-S1', f'{args.method}-S2'])
    plt.savefig(args.save+'_traj.png')


def pca_traj(matrix):
    pca = PCA(n_components=2)
    pca.fit(matrix[args.pca_start:args.pca_end, :]) # [num_models, num_weights], ignore ours-S2 because it is similar with ours-S1

    pc1 = np.array(pca.components_[0])
    pc2 = np.array(pca.components_[1])
    angle = np.ndarray.dot(pc1, pc2)/(np.linalg.norm(pc1)*np.linalg.norm(pc2))
    print("angle between pc1 and pc2: %f" % angle)
    print("pca.explained_variance_ratio_: %s" % str(pca.explained_variance_ratio_))

    coord = pca.transform(matrix)
    draw(coord)
    return pca, coord, pc1, pc2


def point_along_traj(net, mid, dx, dy, matrix, pc1, pc2, diff_save, args, train_loader, diff_keys, pca):
    vec = matrix[mid] + dx*pc1 + dy*pc2
    diff = diff_save[mid] #+ dx*pc1 + dy*pc2
    xy = pca.transform(vec[None, :]).squeeze()
    
    vec = torch.from_numpy(vec).to(args.device)
    diff = torch.from_numpy(diff).to(args.device)
    net = vec2net(net, vec, diff, diff_keys)

    acc, loss = test(args, net, train_loader)
    return xy[0], xy[1], acc, loss


def calc_grid(net, coord, matrix, pc1, pc2, diff_save, args, train_loader, diff_keys, pca):
    mids = [item for item in range(0, 180)] # ids of models except ours-2
    xx_lst = np.linspace(args.x_tuple[0], args.x_tuple[1], args.x_tuple[2]).tolist()
    yy_lst = np.linspace(args.y_tuple[0], args.y_tuple[1], args.y_tuple[2]).tolist()
    
    grid_lst = []
    points = coord[mids]
    for pointx in xx_lst:
        tbar = tqdm(yy_lst, desc=str(int(pointx)))
        for pointy in tbar:
            point1 = np.array([pointx, pointy])
            dist = np.array([np.linalg.norm(point1-points[ii]) for ii in range(len(points))])

            min_id = np.argmin(dist)
            min_point = coord[mids[min_id]]
            dx, dy = point1 - min_point

            xx, yy, acc, loss = point_along_traj(net, min_id, dx, dy, matrix, pc1, pc2, diff_save, args, \
                train_loader, diff_keys, pca)
            grid_lst.append((xx, yy, acc, loss))
            
            tbar.set_postfix(xx=xx, yy=yy, acc=acc)

    # pkl.dump(grid_lst, open("save/08_grid_lst_cifar10_base.pkl", 'wb'))
    pkl.dump(grid_lst, open(args.save + '_grid_lst.pkl', 'wb'))


def along_path(net, matrix, pc1, pc2, diff_save, args, train_loader, diff_keys, pca):
    mids = []
    tmp_lst = [0, 1, 2, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59]
    mids.extend(tmp_lst)
    mids.extend([item+60 for item in tmp_lst])
    mids.extend([item+120 for item in tmp_lst])
    
    coord_lst = []
    # dx_lst = [-1, 0, 1] 
    # dy_lst = [-1, 0, 1]
    dx_lst = args.dx
    dy_lst = args.dy
    for mid in tqdm(mids):
        for dx in dx_lst:
            for dy in dy_lst:
                xx, yy, acc, loss = point_along_traj(net, mid, dx, dy, matrix, pc1, pc2, diff_save, args, \
                    train_loader, diff_keys, pca)
                coord_lst.append((xx, yy, acc, loss))
    
    # pkl.dump(coord_lst, open("save/08_coord_lst_cifar10_base.pkl", 'wb'))
    pkl.dump(coord_lst, open(args.save + '_coord_lst.pkl', 'wb'))
    

def main(args):
    args, models_files, models_split, net = load_state(args)
    matrix, diff_save, diff_keys = save_diff(args, models_files, net)
    pca, coord, pc1, pc2 = pca_traj(matrix)
    pkl.dump(coord, open(args.save + '_coord.pkl', 'wb'))
    
    train_loader, test_loader = data_loader.load_dataset(args, 'base', [0])
    
    calc_grid(net, coord, matrix, pc1, pc2, diff_save, args, train_loader, diff_keys, pca)
    along_path(net, matrix, pc1, pc2, diff_save, args, train_loader, diff_keys, pca)


"""
python grid_loss.py --dataset cifar100 --model densenet-40-12 --method Base --x_tuple -40 45 5 --y_tuple -30 40 5 --save densenet/cifar100_Base --dx 0 --dy 0
python grid_loss.py --dataset cifar100 --model densenet-40-12 --method KDCL --x_tuple -25 35 --y_tuple -25 30 --save densenet/cifar100_KDCL

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 1. Dataset
    parser.add_argument("--dataset", default="cifar10", type=str, help='dataset name')
    parser.add_argument("--data_path", default="../../dataset", type=str, help='dataset path')
    parser.add_argument("--gid", default=-1, type=int)
    parser.add_argument("--bs", default=1024, type=int)
    # 2. Model
    parser.add_argument("--model", default="resnet32", type=str, \
        choices=['resnet32', 'vgg16', 'resnet110', 'densenet-40-12', 'wrn_20_8'])
    parser.add_argument("--method", default="Base", type=str, choices=['Base', 'dml', 'KDCL'])
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--max_epoch", default=300, type=int)
    parser.add_argument("--save_epoch", default=5, type=int)
    # 3. Ckpt name format
    parser.add_argument("--prefix", nargs='+', default=['net0_', 'net1_'])
    parser.add_argument("--suffix", default='ep.pkl', type=str)
    parser.add_argument("--save", default='', type=str)
    # 4. Grid
    parser.add_argument("--x_tuple", nargs='+', default=[])
    parser.add_argument("--y_tuple", nargs='+', default=[])
    parser.add_argument("--dx", nargs='+', default=[-1, 0, 1])
    parser.add_argument("--dy", nargs='+', default=[-1, 0, 1])
    parser.add_argument("--pca_start", default=0, type=int)
    parser.add_argument("--pca_end", default=180, type=int)
       
    args = parser.parse_args()
    args.device = get_free_device(args)
    args.model_folder =  [f'../../ckpt/{args.dataset}/{args.model}_{args.method}/', \
                            f'../../ckpt/{args.dataset}/{args.model}_our/']

    args.x_tuple = [int(item) for item in args.x_tuple]
    args.y_tuple = [int(item) for item in args.y_tuple]
    args.dx = [int(item) for item in args.dx]
    args.dy = [int(item) for item in args.dy]
    if len(args.x_tuple) == 2:
        args.x_tuple.append(args.x_tuple[1]-args.x_tuple[0]+1)
    if len(args.y_tuple) == 2:
        args.y_tuple.append(args.y_tuple[1]-args.y_tuple[0]+1)
    
    if args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'cifar10':
        args.num_classes = 10
    
    if not args.save:
        os.mkdir(args.save)
    
    print(args)
    main(args)