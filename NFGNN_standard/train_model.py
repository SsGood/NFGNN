#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Distributed under terms of the MIT license.

"""

"""

import argparse
import sys
from dataset_utils import DataLoader
from utils import random_planetoid_splits_2 as random_planetoid_splits
from GNN_models import *

import torch
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
import random
import gc
import seaborn as sns
import random
import copy

def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

def RunExp(args, dataset, data, Net, percls_trn, val_lb, RP):

    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data)[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    appnp_net = Net(dataset, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    model = appnp_net.to(device)
    data  = data.to(device)

    if args.net in ['APPNP', 'NFGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': args.prop_wd, 'lr': args.prop_lr
        }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            state = copy.deepcopy(model.state_dict())

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
                    
    model.load_state_dict(model.state_dict())
    model.eval()
    output = model(data)
    eta = model.eta.cpu().detach().numpy()
    gamma = model.prop1.gamma.cpu().detach().numpy()
#     np.save(f'./eta_result/{args.dataset}_{RP}_eta.npy', eta)
#     np.save(f'./gamma_result/{args.dataset}_{RP}_gamma.npy', gamma)
    return test_acc, best_val_acc, gamma


def main(args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--prop_lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--prop_wd', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_rate', type=float, default=0.025)
    parser.add_argument('--val_rate', type=float, default=0.025)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str,
                        default='Mine')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='NF_prop',
                        choices=['PPNP', 'NF_prop'])
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)

    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--rank', type=int, default=3)
    parser.add_argument('--RPMAX', type=int, default=10)
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'NFGNN'],
                        default='GPRGNN')

    args = parser.parse_args()
    print('-----------------------------------------------------')
    print(args)
    print('-----------------------------------------------------')
    Results0 = []
    
    for RP in tqdm(range(args.RPMAX)):
        set_rng_seed(RP)
        gnn_name = args.net
        if gnn_name == 'GCN':
            Net = GCN_Net
        elif gnn_name == 'GAT':
            Net = GAT_Net
        elif gnn_name == 'APPNP':
            Net = APPNP_Net
        elif gnn_name == 'ChebNet':
            Net = ChebNet
        elif gnn_name == 'JKNet':
            Net = GCN_JKNet
        elif gnn_name == 'NFGNN':
            Net = NFGNN
        dname = args.dataset
        dataset, data = DataLoader(dname)

        Init = args.Init

        Gamma_0 = None
        alpha = args.alpha
        train_rate = args.train_rate
        val_rate = args.val_rate
        percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
        val_lb = int(round(val_rate*len(data.y)))
        TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
        print('True Label rate: ', TrueLBrate)

        args.C = len(data.y.unique())
        args.Gamma = Gamma_0


        test_acc, best_val_acc, Gamma_0 = RunExp(
            args, dataset, data, Net, percls_trn, val_lb, RP)
        Results0.append([test_acc, best_val_acc, Gamma_0])
        print(f'best val acc: {best_val_acc}, test acc: {test_acc} on dataset {args.dataset}, in {RP} repeated experiment')

    test_acc_mean, val_acc_mean, _ = np.mean(Results0, axis=0) * 100
    test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
    values=np.asarray(Results0)[:,0]
    uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))
    
    print(f'{gnn_name} on dataset {args.dataset}, in {args.RPMAX} repeated experiment:')
    print(
        f'test acc mean = {test_acc_mean:.4f} \t test acc uncertainty = {uncertainty:.5f} \t val acc mean = {val_acc_mean:.4f}')
    
    return val_acc_mean, test_acc_mean, test_acc_std
    
if __name__ == '__main__':
    with RedirectStdStreams(stdout=sys.stderr):
        val_acc_mean, test_acc_mean, test_acc_std = main()
    print('(%.4f, %.4f, %.4f, %.4f)' % (val_acc_mean, test_acc_mean, 0, 0))
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
