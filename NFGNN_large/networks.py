
# ! /usr/bin/env python
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



class NF_prop(MessagePassing):
    '''
    propagation class for NFGNN
    '''

    def __init__(self, K, alpha, Init, num_classes, rank=1, bias=True, **kwargs):
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
        

    def forward(self, x_k):   
#         x_k = torch.stack(x_list, dim=0)
        H = torch.sigmoid(self.proj(x_k))
        eta = torch.matmul(H, self.gamma.T.unsqueeze(dim=-1)).squeeze().T.unsqueeze(dim=1)
        x_final = x_k.transpose(0,1)
        output = torch.matmul(eta, x_final).squeeze()
        return output

    

class NFGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(NFGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.hidden)
        self.lin3 = Linear(args.hidden, dataset.num_classes)

        if args.ppnp == 'PPNP':
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == 'NF_prop':
            self.prop1 = NF_prop(args.K, args.alpha, args.Init, dataset.num_features, args.rank)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x):
        if self.dprate == 0.0:
            x = self.prop1(x)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        
        return F.log_softmax(x, dim = 1)

