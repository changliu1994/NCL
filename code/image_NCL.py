import argparse
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList_idx, ImageList_twice
import random
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import sys
import os
sys.path.append(os.getcwd())
from randaugment import RandAugmentMC
import faiss
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_strong_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        RandAugmentMC(n=2, m=10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["source_tr"] = ImageList_idx(txt_src, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=False)
    # dsets["target"] = ImageList_idx(txt_tar, transform=image_strong_train())
    dsets["target"] = ImageList_twice(txt_tar, transform= [image_train(), image_strong_train()]  )
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dsets["target_te"] = ImageList_idx(txt_tar, transform=image_test())
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs*3, shuffle=False, 
        num_workers=args.worker, drop_last=False)

    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, 
        num_workers=args.worker, drop_last=False)

    return dset_loaders, dsets

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int),:]
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def correlation_loss(z1, z2, weight=torch.ones(1)):
    # Calculate the covariance
    if weight.numel() ==1:
        covar = z1.T @ z2
    else:
        covar = z1.T @ torch.diag(weight) @ z2

    covar = covar / (torch.sum(covar, dim=1) )
    loss = (torch.sum(covar) - torch.trace(covar)) / z1.shape[1]
    loss += abs(torch.diagonal(covar).add(-1).sum())/z1.shape[1] + 1e-8
    return loss

def train_target(args):
    dset_loaders, dsets = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = max(len(dset_loaders["target"]), len(dset_loaders["source_tr"]) )
    max_iter = args.max_epoch * max_iter
    interval_iter = max_iter //  args.interval
    pbar = tqdm(range(1,max_iter))

    mem_target_feat, mem_target_output = torch.zeros(len(dsets["target"]), args.bottleneck), torch.zeros(len(dsets["target"]), args.class_num)
    mem_src_feat= torch.zeros(len(dsets["source_tr"]), args.bottleneck)

    for iter_num in pbar:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)

        try:
            inputs_src, labels_source, src_idx = next(iter_src)
            inputs_src, labels_source= inputs_src.cuda(), labels_source.cuda()
        except:
            iter_src = iter(dset_loaders["source_tr"])
            inputs_src, labels_source, src_idx = next(iter_src)
            inputs_src, labels_source = inputs_src.cuda(), labels_source.cuda()

        two_output = 0
        if isinstance(inputs_test, list):
            two_output = 1
            inputs_test, inputs_strong= inputs_test[0].cuda(), inputs_test[1].cuda()
        else:
            inputs_test = inputs_test.cuda()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            neighbor_idx, weights = obtain_label(dset_loaders['target_te'], netF, netB, netC, args)
            initc = F.softmax(mem_target_output, -1).T @ (mem_target_feat)
            netF.train()
            netB.train()

        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_src = netB(netF(inputs_src))
        outputs_src = netC(features_src)
        mem_src_feat[src_idx] = features_src.detach().cpu()

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        mem_target_feat[tar_idx], mem_target_output[
            tar_idx] = features_test.detach().cpu(), outputs_test.detach().cpu()

        if two_output:
            features_strong = netB(netF(inputs_strong))
            outputs_strong = netC(features_strong)

        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_src,
                                                                                                   labels_source)

        if args.cls_par > 0 and iter_num > interval_iter:

            (weighted_feat, weighted_output) = get_neighbor_info(mem_target_feat, mem_target_output, neighbor_idx, tar_idx)
            ins_out_strong=  F.normalize(features_strong, p=2, dim=1) @ F.normalize(initc.T.detach().cuda(), p=2, dim=1)
            ins_out_weak = F.normalize(features_test, p=2, dim=1)  @ F.normalize(initc.T.detach().cuda(), p=2, dim=1)
            ins_out_weighted = F.normalize(weighted_feat, p=2, dim=1)  @ F.normalize(initc.T.detach().cuda(), p=2, dim=1)

            classifier_loss += correlation_loss(F.softmax(outputs_test, -1), F.softmax(outputs_strong, -1))
            classifier_loss += correlation_loss(F.softmax(ins_out_strong, -1), F.softmax(ins_out_weak, -1))

            classifier_loss += args.cls_par* correlation_loss(F.softmax(outputs_test, -1),
                                                               F.softmax(weighted_output, -1), weights[tar_idx].cuda())
            classifier_loss += args.cls_par* correlation_loss(F.softmax(ins_out_strong, -1),
                                                               F.softmax(ins_out_weighted, -1),weights[tar_idx].cuda())

            pseudo_label = torch.softmax(outputs_test.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            classifier_loss += (F.cross_entropy(outputs_strong, targets_u,
                                  reduction='none') * mask).mean()
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0

        if args.ent and iter_num > interval_iter:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        classifier_loss.backward()

        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()

    if args.issave:   
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        
    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    # Normalize features for cosine similarity
    if args.distance == 'cosine':
        # all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)  # Add bias term
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()  # Normalize to unit vectors

    # Convert all_fea to numpy for FAISS
    all_fea_np = all_fea.numpy()

    # Initialize FAISS for cosine similarity (normalized vectors)
    faiss.normalize_L2(all_fea_np)
    dim = all_fea_np.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance index for normalized vectors (equivalent to cosine similarity)
    index.add(all_fea_np)  # Add the features to the FAISS index

    distances, indices = index.search(all_fea_np, args.k + 1)  # Get K+1 nearest neighbors for each sample

    return indices, unknown_weight

def get_neighbor_info(all_fea, all_output, indices, tar_idx):
    knn_feats = all_fea[indices[tar_idx][:, 1:]]
    knn_outputs = all_output[indices[tar_idx][:, 1:]]

    weighted_feat = torch.mean(knn_feats , dim=1)
    weighted_output = torch.mean(knn_outputs , dim=1)

    return (weighted_feat.cuda(), weighted_output.cuda())
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT++')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet18, resnet50")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0.9)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='ckps/t2021')
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--ssl', type=float, default=0.0)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    elif args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    elif args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
        args.lr = 1e-3

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = '../data/'
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.k > 0:
             args.savename += ('_k_' + str(args.k))

        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()

        train_target(args)
        args.out_file.close()
