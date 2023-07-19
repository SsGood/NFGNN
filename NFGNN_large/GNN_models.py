
#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
from typing import Optional
from torch_geometric.typing import OptTensor

import torch
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.utils import add_self_loops, get_laplacian, remove_self_loops


# class NF_prop(MessagePassing):
#     '''
#     propagation class for NFGNN
#     '''

#     def __init__(self, K, alpha, Init, num_classes, Gamma=None, bias=True, **kwargs):
#         super(NF_prop, self).__init__(aggr='add', **kwargs)
#         self.K = K
#         self.Init = Init
#         self.alpha = alpha

#         assert Init in ['PPR', 'Random', 'Fix']
#         if Init == 'PPR':
#             # PPR-like
#             TEMP = alpha*(1-alpha)**np.arange(K+1)
#             TEMP[-1] = (1-alpha)**K
#         elif Init == 'Random':
#             # Random
#             bound = np.sqrt(3/(K+1))
#             TEMP = np.random.uniform(-bound, bound, K+1)
#             TEMP = TEMP/np.sum(np.abs(TEMP))
#         elif Init == 'Fix':
#             TEMP = np.ones(K+1)

#         self.gamma = Parameter(torch.tensor(TEMP))
#         self.proj = Linear(num_classes, 1)

#     def reset_parameters(self):
#         torch.nn.init.zeros_(self.gamma)
#         for k in range(self.K+1):
#             self.gamma.data[k] = self.alpha*(1-self.alpha)**k
#         self.gamma.data[-1] = (1-self.alpha)**self.K
    
#     def forward(self, x, edge_index, edge_weight=None):
#         edge_index, norm = self.__norm__(edge_index, x.size(0), edge_weight, dtype=x.dtype)
#         x_list = []
#         temp = []
#         Tx_0 = x
#         Tx_1 = self.propagate(edge_index, x=Tx_0, norm=norm)
        
#         x_list.append(Tx_0)
#         x_list.append(Tx_1*(self.gamma[1]))
        
#         hidden = Tx_0*(self.gamma[0]) + Tx_1*(self.gamma[1])
#         for k in range(1, self.K):
#             Tx_2 = 2. * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
#             Tx_0, Tx_1 = Tx_1, Tx_2
#             hidden = hidden + Tx_1* (self.gamma[k+1])
#             x_list.append(Tx_1 * (self.gamma[k+1]))
            
#         x_k = torch.stack(x_list, dim=1)
#         H = torch.sigmoid(self.proj(x_k)).squeeze().unsqueeze(dim=1)
#         output = torch.matmul(H, x_k).squeeze()
#         return output
    
#     def __norm__(self, edge_index, num_nodes: Optional[int],
#                  edge_weight: OptTensor, normalization: Optional[str] = "sym",
#                  lambda_max: OptTensor = None, dtype: Optional[int] = None,
#                  batch: OptTensor = None):

#         edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

#         edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
#                                                 normalization, dtype,
#                                                 num_nodes)

#         if lambda_max is None:
#             lambda_max = 2.0 * edge_weight.max()
#         elif not isinstance(lambda_max, torch.Tensor):
#             lambda_max = torch.tensor(lambda_max, dtype=dtype,
#                                       device=edge_index.device)
#         assert lambda_max is not None

#         if batch is not None and lambda_max.numel() > 1:
#             lambda_max = lambda_max[batch[edge_index[0]]]

#         edge_weight = (2.0 * edge_weight) / lambda_max
#         edge_weight.masked_fill_(edge_weight == float('inf'), 0)

#         edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
#                                                  fill_value=-1.,
#                                                  num_nodes=num_nodes)
#         assert edge_weight is not None

#         return edge_index, edge_weight

#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * x_j

#     def __repr__(self):
#         return '{}(K={}, gamma={})'.format(self.__class__.__name__, self.K,
#                                           self.gamma)
    

class NF_prop(MessagePassing):
    '''
    propagation class for NFGNN
    '''

    def __init__(self, K, alpha, Init, num_classes, rank=1, Gamma=None, bias=True, **kwargs):
        super(NF_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['PPR', 'Random', 'Fix']
        if Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Fix':
            TEMP = np.ones(K+1)
        TEMP = np.array([TEMP for i in range(rank)])
        self.gamma = Parameter(torch.tensor(TEMP).float())
        self.proj = Linear(num_classes, rank)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.gamma)
        for k in range(self.K+1):
            self.gamma.data[k] = self.alpha*(1-self.alpha)**k
        self.gamma.data[-1] = (1-self.alpha)**self.K
        
    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str] = "sym",
                 lambda_max: OptTensor = None, dtype: Optional[int] = None,
                 batch: OptTensor = None):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype,
                                      device=edge_index.device)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1.,
                                                 num_nodes=num_nodes)
        assert edge_weight is not None

        return edge_index, edge_weight

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, gamma={})'.format(self.__class__.__name__, self.K,
                                          self.gamma)
    
#     def forward(self, x, edge_index, edge_weight=None):
#         edge_index, norm = self.__norm__(edge_index, x.size(0), edge_weight, dtype=x.dtype)
#         x_list = []
#         Tx_0 = x
#         Tx_1 = self.propagate(edge_index, x=Tx_0, norm=norm)
        
#         h = torch.sigmoid(self.proj(Tx_0))
#         eta = torch.matmul(h, self.gamma[:, 0].unsqueeze(dim=1))
#         x_list.append(Tx_0 * eta)
#         h = torch.sigmoid(self.proj(Tx_1))
#         eta = torch.matmul(h, self.gamma[:, 1].unsqueeze(dim=1))
#         x_list.append(Tx_1 * eta)
        
#         for k in range(1, self.K):
#             Tx_2 = 2. * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
#             Tx_0, Tx_1 = Tx_1, Tx_2
            
#             h = torch.sigmoid(self.proj(Tx_1))
#             eta = torch.matmul(h, self.gamma[:, k].unsqueeze(dim=1))
#             temp = Tx_1 * eta
#             x_list.append(temp)
            
#         x_k = torch.stack(x_list, dim=1)
#         output = x_k.sum(dim=1)
#         return output
    
    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = self.__norm__(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        x_list = []
        Tx_0 = x
        Tx_1 = self.propagate(edge_index, x=Tx_0, norm=norm)
        x_list.append(Tx_0)
        x_list.append(Tx_1)
        
        for k in range(1, self.K):
            Tx_2 = 2. * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            Tx_0, Tx_1 = Tx_1, Tx_2
            x_list.append(Tx_1)
            
        x_k = torch.stack(x_list, dim=0)
        H = torch.sigmoid(self.proj(x_k))
        eta = torch.matmul(H, self.gamma.T.unsqueeze(dim=-1)).squeeze().T.unsqueeze(dim=1)
        x_final = torch.stack(x_list, dim=1)
        output = torch.matmul(eta, x_final).squeeze()
        return output
    
    


class NFGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(NFGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        if args.ppnp == 'PPNP':
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == 'NF_prop':
            self.prop1 = NF_prop(args.K, args.alpha, args.Init, dataset.num_classes, args.rank, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.lin2(x)
            return F.log_softmax(x, dim=1)


class GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(dataset.num_features, 32, K=2)
        self.conv2 = ChebConv(32, dataset.num_classes, K=2)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN_JKNet(torch.nn.Module):
    def __init__(self, dataset, args):
        in_channels = dataset.num_features
        out_channels = dataset.num_classes

        super(GCN_JKNet, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin1 = torch.nn.Linear(16, out_channels)
        self.one_step = APPNP(K=1, alpha=0)
        self.JK = JumpingKnowledge(mode='lstm',
                                   channels=16,
                                   num_layers=4
                                   )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index)
        x = self.lin1(x)
        return F.log_softmax(x, dim=1)
