import numpy as np
import torch
import pickle
import argparse
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
import scipy.sparse as sp
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor

def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    row_sum=(row_sum==0)*1+row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def main(args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='products')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--data_path', type=str, default='../data/pre/')    
    
    args = parser.parse_args()
    
    print('-------------loading data--------------')
    dataset = PygNodePropPredDataset(name=f'ogbn-{args.dataset}', root = '../data/OGB/')
    print('-------------data loading complete--------------')
    
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    del dataset
    trn_idx = split_idx['train'] 
    val_idx = split_idx['valid']
    tst_idx = split_idx['test']
#     edge_index = data.edge_index
    num_nodes = data.num_nodes
    with open(args.data_path+"all_feat_"+args.dataset+".pickle","wb") as fopen:
        pickle.dump(data.x.numpy(),fopen)
    print('-------------Feature saved--------------')
    
    with open(args.data_path+"all_edge_"+args.dataset+".pickle","wb") as fopen:
        pickle.dump(data.edge_index.numpy(),fopen)
    print('-------------edge saved--------------')
        
    with open(args.data_path+"all_ind_"+args.dataset+".pickle","wb") as fopen:
        pickle.dump([trn_idx, val_idx, tst_idx],fopen)
    print('-------------index saved--------------')


    edge_index = to_undirected(edge_index)
    row, col = edge_index
    row      = row.numpy()
    col      = col.numpy()
    adj_mat = sp.csr_matrix((np.ones(row.shape[0]),(row,col)),shape=(num_nodes,num_nodes))    
    adj_mat = sys_normalized_adjacency(adj_mat)
    L       = -1.0*adj_mat
    adj_mat = sparse_mx_to_torch_sparse_tensor(L)

    list_mat_trn = []
    list_mat_val = []
    list_mat_tst = []
    
    T_0_feat = data.x.numpy()
    T_0_feat = torch.from_numpy(T_0_feat).float()
    
    del dataset, data, edge_index, row, col, labels
    
    list_mat_trn.append(T_0_feat[trn_idx,:])
    list_mat_val.append(T_0_feat[val_idx,:])
    list_mat_tst.append(T_0_feat[tst_idx,:])
    print('-------------T_0 have been saved!-------------')
    
    T_1_feat = torch.spmm(adj_mat,T_0_feat)
    list_mat_trn.append(T_1_feat[trn_idx,:])
    list_mat_val.append(T_1_feat[val_idx,:])
    list_mat_tst.append(T_1_feat[tst_idx,:])
    print('-------------T_1 have been saved!-------------')
    
    for i in range(1,args.K):
    #T_k(\hat{L})X
        T_2_feat = torch.spmm(adj_mat,T_1_feat)
        T_2_feat = 2*T_2_feat-T_0_feat
        T_0_feat, T_1_feat =T_1_feat, T_2_feat

        list_mat_trn.append(T_2_feat[trn_idx,:])
        list_mat_val.append(T_2_feat[val_idx,:])
        list_mat_tst.append(T_2_feat[tst_idx,:])
        print(f'-------------T_{i+1} have been saved!-------------')
        
    with open(args.data_path+"training_"+args.dataset+".pickle","wb") as fopen:
        pickle.dump(list_mat_trn,fopen)

    with open(args.data_path+"validation_"+args.dataset+".pickle","wb") as fopen:
        pickle.dump(list_mat_val,fopen)

    with open(args.data_path+"test_"+args.dataset+".pickle","wb") as fopen:
        pickle.dump(list_mat_tst,fopen)
        
    print(f'-------------{args.dataset} have been successfully processed!-------------')


if __name__ == '__main__':
    main()

    

    

    

























