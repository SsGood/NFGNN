#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import shutil

import os
import os.path as osp
import pickle
import torch_geometric.transforms as T

from cSBM_dataset import dataset_ContextualSBM
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Actor
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected

import networkx as nx
from torch_geometric.utils import *


class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, pre_transform=None):

        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class WebKB(InMemoryDataset):
    r"""The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)
    
    
class Airports(InMemoryDataset):
    def __init__(self, root, dataset_name, transform=None, pre_transform=None):
        self.dataset_name  = dataset_name
        self.dump_location = "../data/airports/airports_dataset_dump"
        super(Airports, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.dataset_name+'-airports.edgelist', 'labels-'+self.dataset_name+'-airports.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            source = self.dump_location + '/' + name
            shutil.copy(source, self.raw_dir)

    def process(self):

        fin_labels = open(self.raw_paths[1])
        labels = []
        node_id_mapping = dict()
        node_id_labels_dict = dict()
        for new_id, line in enumerate(fin_labels.readlines()[1:]): # first line is header so ignore
            old_id, label = line.strip().split()
            labels.append(int(label))
            node_id_mapping[old_id] = new_id
            node_id_labels_dict[new_id] = int(label)
        fin_labels.close()

        edges = []
        fin_edges = open(self.raw_paths[0])
        for line in fin_edges.readlines():
            node1, node2 = line.strip().split()[:2]
            edges.append([node_id_mapping[node1], node_id_mapping[node2]])
        fin_edges.close()

        networkx_graph = nx.Graph(edges)

        print("No. of Nodes: ",networkx_graph.number_of_nodes())
        print("No. of edges: ",networkx_graph.number_of_edges())


        attr = {}
        for node in networkx_graph.nodes():
            deg = networkx_graph.degree(node)
            attr[node] = {"y": node_id_labels_dict[node], "x": [float(deg)]}
        nx.set_node_attributes(networkx_graph, attr)
        data = from_networkx(networkx_graph)


        if self.pre_filter is not None:
            data = self.pre_filter(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
    
    
    
def convert_ndarray(x):
    y = list(range(len(x)))
    for k, v in x.items():
        y[int(k)] = v
    return np.array(y)
def rm_useless(G, feats, class_map, unlabeled_nodes, num_layers):
    # find useless nodes
    print('start to check and remove {} unlabeled nodes'.format(len(unlabeled_nodes)))

    rm_nodes = unlabeled_nodes
    if len(rm_nodes):
        for node in rm_nodes:
            G.remove_node(node)
        G_new = nx.relabel.convert_node_labels_to_integers(G, ordering='sorted')
        feats = np.delete(feats, rm_nodes, 0)
        class_map = np.delete(class_map, rm_nodes, 0)
        print('remove {} '.format(len(rm_nodes)), 'useless unlabeled nodes')
    return G_new, feats, class_map



class BGP(InMemoryDataset):


    def __init__(self, root, transform=None, pre_transform=None):

        self.dump_location = "../data/bgp/bgp_data_dump"
        super(BGP, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['as-G.json', 'as-class_map.json', 'as-feats.npy','as-feats_t.npy', 'as-edge_list']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            # download_url('{}/{}'.format(self.url, name), self.raw_dir)
            source = self.dump_location + '/' + name
            shutil.copy(source, self.raw_dir)

    def process(self):
        G = nx.json_graph.node_link_graph(json.load(open(self.raw_paths[0])), False)
        class_map = json.load(open(self.raw_paths[1]))
        feats = np.load(self.raw_paths[2])
        feats_t = np.load(self.raw_paths[3])


        train_nodes = [n for n in G.nodes() if not G.nodes[n]['test'] and not G.nodes[n]['val']]
        val_nodes = [n for n in G.nodes() if not G.nodes[n]['test'] and G.nodes[n]['val']]
        test_nodes = [n for n in G.nodes() if G.nodes[n]['test'] and not G.nodes[n]['val']]
        unlabeled_nodes = [n for n in G.nodes() if G.nodes[n]['test'] and G.nodes[n]['val']]
        class_map = convert_ndarray(class_map)

        G, feats, class_map = rm_useless(G, feats, class_map, unlabeled_nodes, 1)
        train_nodes = [n for n in G.nodes() if not G.nodes[n]['test'] and not G.nodes[n]['val']]
        val_nodes = [n for n in G.nodes() if not G.nodes[n]['test'] and G.nodes[n]['val']]
        test_nodes = [n for n in G.nodes() if G.nodes[n]['test'] and not G.nodes[n]['val']]
        unlabeled_nodes = [n for n in G.nodes() if G.nodes[n]['test'] and G.nodes[n]['val']]


        data = from_networkx(G)
        data.train_mask = ~(data.test | data.val)
        data.val_mask = data.val
        data.test_mask = data.test
        data.test = None
        data.val = None
        data.x = torch.FloatTensor(feats)
        data.y = torch.LongTensor(np.argmax(class_map, axis=1))

        if self.pre_filter is not None:
            data = self.pre_filter(data)

        if self.pre_transform is not None:
            data.edge_index = to_undirected(data.edge_index)
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def DataLoader(name):
    if 'cSBM_data' in name:
        path = '../data/'
        dataset = dataset_ContextualSBM(path, name=name)
    else:
        name = name.lower()

    if name in ['cora', 'citeseer', 'pubmed']:
        root_path = '../'
        path = osp.join(root_path, 'data', name)
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        root_path = '../'
        path = osp.join(root_path, 'data', name)
        dataset = Amazon(path, name, T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        # use everything from "geom_gcn_preprocess=False" and
        # only the node label y from "geom_gcn_preprocess=True"
        preProcDs = WikipediaNetwork(
            root='../data/', name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root='../data/', name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
        return dataset, data

    elif name in ['film']:
        dataset = Actor(
            root='../data/film', transform=T.NormalizeFeatures())
    elif name in ['texas', 'cornell']:
        dataset = WebKB(root='../data/',
                        name=name, transform=T.NormalizeFeatures())
    elif name in ['brazil', 'europe', 'usa']:
        dataset = Airports(root='../data/',
                        dataset_name=name, transform=T.NormalizeFeatures())
    elif name in ['bgp']:
        dataset = BGP(root="../data/", transform=T.NormalizeFeatures())
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset, dataset[0]
