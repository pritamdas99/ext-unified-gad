import os
import argparse
import random
from sklearn.model_selection import train_test_split
import pickle
import json
import pprint
from tqdm import tqdm
import dgl
from dgl.data.utils import load_graphs
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.dataloading import GraphDataLoader
from dgl import KHopGraph, save_graphs
import dgl.function as fn

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from functools import partial

from line_profiler import profile


NAME_MAP = {
    'n': "Node",
    'e': "Edge",
    'g': "Graph",
}

DATASETS = ['reddit', 'weibo', 'amazon', 'yelp', 'tfinance', 'elliptic', 'tolokers', 'questions', 'dgraphfin', 'tsocial', 'hetero/amazon', 'hetero/yelp', 
            'uni-tsocial', 
            'mnist/dgl/mnist0', 'mnist/dgl/mnist1', 
            'mutag/dgl/mutag0', 
            'bm/dgl/bm_mn_dgl', 'bm/dgl/bm_ms_dgl', 'bm/dgl/bm_mt_dgl',
            'tfinace'
            ]

EPS = 1e-12 # for nan
ROOT_SEED = 3407

# ======================================================================
#   Model activation/normalization creation function
# ======================================================================

def obtain_act(name=None):
    """
    Return activation function module
    """
    if name == 'relu':
        act = nn.ReLU(inplace=True)
    elif name == "gelu":
        act = nn.GELU()
    elif name == "prelu":
        act = nn.PReLU()
    elif name == "elu":
        act = nn.ELU()
    elif name == "leakyrelu":
        act = nn.LeakyReLU()
    elif name == "tanh":
        act = nn.Tanh()
    elif name == "sigmoid":
        act = nn.Sigmoid()
    elif name is None:
        act = nn.Identity()
    else:
        raise NotImplementedError("{} is not implemented.".format(name))

    return act


def obtain_norm(name):
    """
    Return normalization function module
    """
    if name == "layernorm":
        norm = nn.LayerNorm
    elif name == "batchnorm":
        norm = nn.BatchNorm1d
    elif name == "instancenorm":
        norm = partial(nn.InstanceNorm1d, affine=True, track_running_stats=True)
    else:
        return nn.Identity

    return norm


def obtain_pooler(pooling):
    """
    Return pooling function module
    """
    if pooling == "mean":
        pooler = AvgPooling()
    elif pooling == "max":
        pooler = MaxPooling()
    elif pooling == "sum":
        pooler = SumPooling()
    else:
        raise NotImplementedError

    return pooler

# -----

def set_seed(seed=ROOT_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        torch.use_deterministic_algorithms(mode=True)


def select_all_khop(star_khop_graph_big, central_node_id, khop, select_topk):
    pres = star_khop_graph_big.predecessors(central_node_id) # in edges
    sucs = star_khop_graph_big.successors(central_node_id) # out edges
    node_ids = torch.unique(torch.cat([pres, sucs], dim=0))
    nbs = torch.unique(node_ids)
    weights = torch.ones(nbs.shape[0], 1)
    weights[:-1, 0] = weights[:-1, 0]/(nbs.shape[0] + EPS)
    return nbs, weights

def select_rand_khop(star_khop_graph_big, central_node_id, khop, select_topk):
    pres = star_khop_graph_big.predecessors(central_node_id) # in edges
    sucs = star_khop_graph_big.successors(central_node_id) # out edges
    node_ids = torch.unique(torch.cat([pres, sucs], dim=0))
    idx = torch.randperm(node_ids.shape[0])
    nbs = node_ids[idx[:100]]
    weights = torch.ones(nbs.shape[0], 1)
    weights[:-1, 0] = weights[:-1, 0]/(nbs.shape[0] + EPS)
    return nbs, weights

def select_topk_star_normft(star_khop_graph_big, node_ids, central_node_id):
    h_xs, id_xs, h_x0, id_x0 = star_khop_graph_big.ndata['feature_normed'][node_ids], node_ids, star_khop_graph_big.ndata['feature_normed'][central_node_id], central_node_id

    xs = list(zip(h_xs, id_xs))
    x0 = (h_x0, id_x0)

    up = 0
    down = torch.pow( x0[0], 2 ) + EPS # element wise
    best = up / down
    greedy = lambda xi: - (x0[0] - xi[0])**2 / (xi[0]**2)
    xs_sorted = sorted(xs, key=greedy)
    nbs = []
    for i,xi in enumerate(xs_sorted):
        tmp_up = (x0[0] - xi[0])**2
        tmp_down = xi[0]**2 + EPS
        if best < tmp_up/tmp_down:
            up += tmp_up
            down += tmp_down
            best = up / down
            nbs.append(xi[1]) # sotre the id
        else:
            break
    return nbs

def select_topk_star_unionft(star_khop_graph_big, node_ids, central_node_id):
    h_xs, id_xs, h_x0, id_x0 = star_khop_graph_big.ndata['feature'][node_ids], node_ids, star_khop_graph_big.ndata['feature'][central_node_id], central_node_id


    nbs = set()
    for feature_id in range(h_xs.shape[1]):
        xs = list(zip(h_xs[:, feature_id], id_xs))
        x0 = (h_x0[feature_id], id_x0)

        up = 0
        down = torch.pow( x0[0], 2 ) # element wise
        best = up / down
        greedy = lambda xi: - (x0[0] - xi[0])**2 / (xi[0]**2)
        xs_sorted = sorted(xs, key=greedy)

        for i,xi in enumerate(xs_sorted):
            tmp_up = (x0[0] - xi[0])**2
            tmp_down = xi[0]**2
            if best < tmp_up/tmp_down:
                up += tmp_up
                down += tmp_down
                best = up / down
                nbs.add(xi[1]) # sotre the id
    return list(nbs)

def get_star_topk_nbs(star_khop_graph_big, central_node_id, khop, select_topk):
    star_khop_graph_in = star_khop_graph_big.sample_neighbors([central_node_id],fanout=-1, edge_dir='in')
    star_khop_graph_out = star_khop_graph_big.sample_neighbors([central_node_id],fanout=-1, edge_dir='out')
    # print("### util topknbs: star_khop_graph_in", star_khop_graph_in.edges())
    # print("### util topknbs: star_khop_graph_out", star_khop_graph_out.edges())
    node_ids_in = star_khop_graph_in.edges()[0]
    node_ids_out = star_khop_graph_out.edges()[1]
    node_ids = torch.cat([node_ids_in, node_ids_out], dim=0)
    # print("### util topknbs: node_ids_in", node_ids_in, "out", node_ids_out)
    # print("### util topknbs: node_ids", node_ids)
    node_ids = torch.unique(node_ids)


    nbs = select_topk(star_khop_graph_big, node_ids, central_node_id)
    nbs.append(torch.tensor(central_node_id).long()) # make sure self is added
    weights = torch.ones(len(nbs), 1)*0.5
    weights[:-1, 0] = weights[:-1, 0]/(len(nbs) + EPS)

    return nbs, weights

@profile
def get_convtree_topk_nbs_norm(graph_whole, xi, khop, select_topk):
    '''
        return topk neighbors weight matrix in Conv Tree graph setting
    '''
    # find all 1st-order neighbours
    pres = graph_whole.predecessors(xi) # in edges
    sucs = graph_whole.successors(xi) # out edges
    nbs_xi = torch.unique(torch.cat([pres, sucs], dim=0)) # FIXME: if all bidirected, delete this for performance
    if nbs_xi.shape[0] == 0:
        # no neighbours
        return tuple([xi]), tuple([1.0])
    # some refrences for help
    xf = graph_whole.ndata['feature_normed']
    Pij = {} 
    Pik = {}
    Pij_tmp = {}
    Smaxj_list = []
    quant = lambda x: - x[1] / x[2]
    for xj in nbs_xi:
        # clear tmp ik for j
        Pik_tmp = {}
        # add parent edge
        aj = ( xf[xj] - xf[xi] )**2
        bj = ( xf[xj] )**2
        Smaxj = aj / bj
        # get xj's neighbours 
        pres = graph_whole.predecessors(xj) # in edges
        sucs = graph_whole.successors(xj) # out edges
        nbs_xj = torch.unique(torch.cat([pres, sucs], dim=0))
        if nbs_xj.shape[0] == 0:
            # xj no neighbours
            Pij_tmp[xj] = 0.5 # 1/2
        else:
            Pij_tmp[xj] = 0.25 # 1/4, because it has to avg with sons
            num_hop2 = 0 # how many sons has been selected, could be 0?
            ss = [ (xk.item(), (xf[xk]-xf[xj])**2, xf[xk]**2) for xk in nbs_xj] # store in (k, ak,bk) form
            ss.sort(key = lambda x: -x[1]/x[2]) # from big to small ak/bk
            # loop to find the optimal value
            for xk, ak, bk in ss:
                if ak / bk > Smaxj:
                    num_hop2 += 1
                    # update the best sons
                    aj += ak
                    bj += bk
                    Smaxj = aj / bj
                    Pik_tmp[xk] = 0.25 # Pik_tmp[xk] = 1/4
                else:
                    # the rest is impossible to make the ans bigger
                    break
            if num_hop2 != 0:
                # update all Pik_tmp
                for xk in Pik_tmp:
                    Pik_tmp[xk] /= num_hop2
                    # add to global Pik
                    if xk in Pik:
                        Pik[xk] += Pik_tmp[xk]
                    else:
                        Pik[xk] = Pik_tmp[xk]
            else:
                Pij_tmp[xj] = 0.5
        Smaxj_list.append((xj, aj, bj)) # j, aj, bj
    Smaxj_list = sorted(Smaxj_list, key=lambda x: -x[1]/x[2]) # from big to small
    ai = Smaxj_list[0][1]
    bi = Smaxj_list[0][2]
    RQ_max = ai / bi # at least the largest one should be selected
    num_hop1 = 1
    for xj, aj, bj in Smaxj_list[1:]: # check the rest
        if aj / bj > RQ_max:
            num_hop1 += 1
            ai += aj
            bi += bj
            RQ_max = ai / bi
            Pij[xj] = Pij_tmp[xj] # select xj
        else:
            break
    
    for xj in Pij:
        # update all Pij
        Pij[xj] /= num_hop1
    Pij[xi] = 0.5 # self loop

    Pfinal = {k:v for k,v in Pij.items()}
    for k,v in Pik.items():
        if k in Pfinal:
            Pfinal[k] += v
        else:
            Pfinal[k] = v

    adj_list, weight_list  = tuple(Pfinal.keys()), tuple(Pfinal.values())

    return adj_list, weight_list


def collate_with_sp(batch):
    graphs = [item[0] for item in batch]
    labels_dicts = [item[1] for item in batch]
    nodes_list = [item[2] for item in batch]
    edges_list = [item[3] for item in batch]
    sp_matrices = [item[4] for item in batch]

    return {
        'graphs': graphs,
        'labels_dicts': labels_dicts,
        'nodes_list': nodes_list,
        'edges_list': edges_list,
        'sp_matrices': sp_matrices
    }


class Dataset:
    def __init__(self, name='tfinance', prefix='../datasets/', labels_have="ng", sp_type='star+norm', debugnum = -1):
        self.full_name = prefix + name
        ### avoid repeat calcs
        self.prepare_dataset_done = False
        self.make_sp_matrix_graph_list_done = False

        if "unified" not in prefix and "edge_labels" not in prefix:
            graph = load_graphs(prefix + name)[0][0]
            self.name = name
            self.graph = graph
            self.in_dim = graph.ndata['feature'].shape[1]
        else: # mainly handle single graph type datasets
            print("Unified dataset ", prefix + name, labels_have)
            self.labels_have = labels_have
            # graph list as well as node labels
            graph, label = load_graphs(prefix + name)
            print("utils.py:===========> Total graphs loaded: ", len(graph))
            self.name = name
            if debugnum == -1:
                self.graph_list = graph
            else:
                self.graph_list = graph[:debugnum]

            print("utils.py:===========> Total graphs used: ", len(self.graph_list))

            self.in_dim = self.graph_list[0].ndata['feature'].shape[1]
            self.sp_type = sp_type
            self.sp_method, self.agg_ft = sp_type.split('+')

            if self.sp_method == 'star':
                self.get_sp_adj_list = get_star_topk_nbs
                if self.agg_ft == 'norm':
                    self.select_topk_fn = select_topk_star_normft
                elif self.agg_ft == "union":
                    self.select_topk_fn = select_topk_star_unionft
                else:
                    raise NotImplementedError
            elif self.sp_method == 'convtree':
                if self.agg_ft == 'norm':
                    self.get_sp_adj_list = get_convtree_topk_nbs_norm
                    self.select_topk_fn = None
                elif self.agg_ft == "union":
                    raise NotImplementedError
                else:
                    raise NotImplementedError
            elif self.sp_method == 'khop':
                self.get_sp_adj_list = select_all_khop
                self.select_topk_fn = None
            elif self.sp_method == 'rand':
                self.get_sp_adj_list = select_rand_khop
                self.select_topk_fn = None
            else:
                raise NotImplementedError

    def split(self):
        self.training_graph_labels_dict = []
        self.validation_graph_labels_dict = []
        self.testing_graph_labels_dict = []

        if len(self.graph_list) == 1:
            train_nodes = self.training_graph_nodes
            train_edges = self.training_graph_edges
            i=0
            for x, n, e in zip(self.training_graph_sampled, train_nodes, train_edges):
                labels_dict = {}
                if 'n' in self.labels_have:
                    labels_dict['node_label'] = x.ndata['node_label']
                if 'e' in self.labels_have:
                    labels_dict['edge_label'] = x.edata['edge_label']
                if i <= 2:
                    print(labels_dict['node_label'][:10], labels_dict['edge_label'][:10])
                self.training_graph_labels_dict.append(labels_dict)

            val_nodes = self.validation_graph_nodes
            val_edges = self.validation_graph_edges
            for x, n, e in zip(self.validation_graph_sampled, val_nodes, val_edges):
                labels_dict = {}
                if 'n' in self.labels_have:
                    labels_dict['node_label'] = x.ndata['node_label']
                if 'e' in self.labels_have:
                    labels_dict['edge_label'] = x.edata['edge_label']
                self.validation_graph_labels_dict.append(labels_dict)

            test_nodes = self.testing_graph_nodes
            test_edges = self.testing_graph_edges
            for x, n, e in zip(self.testing_graph_sampled, test_nodes, test_edges):
                labels_dict = {}
                if 'n' in self.labels_have:
                    labels_dict['node_label'] = x.ndata['node_label']
                if 'e' in self.labels_have:
                    labels_dict['edge_label'] = x.edata['edge_label']
                self.testing_graph_labels_dict.append(labels_dict)

    def get_graph_and_sp_dataloaders(self, batch_size=32, shuffle=True, num_workers=0):
        self.split()
        train_graphs_with_all_labels = list(zip(self.training_graph_sampled, 
                                                self.training_graph_labels_dict,
                                                self.training_graph_nodes,
                                                self.training_graph_edges,
                                                self.sp_matrix_graph_train_list))
        self.train_loader = GraphDataLoader(
            train_graphs_with_all_labels,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_with_sp,
        )

        val_graphs_with_all_labels = list(zip(self.validation_graph_sampled,
                                              self.validation_graph_labels_dict,
                                              self.validation_graph_nodes,
                                              self.validation_graph_edges,
                                              self.sp_matrix_graph_val_list))
        self.val_loader = GraphDataLoader(
            val_graphs_with_all_labels,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_with_sp,
        )

        test_graphs_with_all_labels = list(zip(self.testing_graph_sampled,
                                               self.testing_graph_labels_dict,
                                               self.testing_graph_nodes,
                                               self.testing_graph_edges,
                                               self.sp_matrix_graph_test_list))
        self.test_loader = GraphDataLoader(
            test_graphs_with_all_labels,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_with_sp,
        )
        return self.train_loader, self.val_loader, self.test_loader





    def make_sp_matrix_graph_list(self, khop=1, sp_type='star+union', load_kg = False):
        self.sp_matrix_graph_train_list = []
        self.sp_matrix_graph_val_list = []
        self.sp_matrix_graph_test_list = []
        self.sp_matrix_graphs_train_filename = f"{self.full_name}.khop_{khop}.sp_type_{self.sp_type}.sp_matrix.train"
        self.sp_matrix_graphs_val_filename = f"{self.full_name}.khop_{khop}.sp_type_{self.sp_type}.sp_matrix.val"
        self.sp_matrix_graphs_test_filename = f"{self.full_name}.khop_{khop}.sp_type_{self.sp_type}.sp_matrix.test"

        if load_kg and os.path.exists(self.sp_matrix_graphs_train_filename):
            self.sp_matrix_graph_train_list, _ = load_graphs(self.sp_matrix_graphs_train_filename)
            self.sp_matrix_graph_val_list, _ = load_graphs(self.sp_matrix_graphs_val_filename)
            self.sp_matrix_graph_test_list, _ = load_graphs(self.sp_matrix_graphs_test_filename)
        else:
            print("### util: graph list len: ", len(self.training_graph_sampled))
            j=1
            for idx,graph in enumerate(tqdm(self.training_graph_sampled)):
                with graph.local_scope():
                    if self.agg_ft == 'norm':
                        if self.full_name.endswith("mutag0"):
                            graph.ndata['feature_normed'] =  graph.ndata['feature'].argmax(dim=1)
                        else:
                            graph.ndata['feature_normed'] =  graph.ndata['feature']
                            # norm it
                            graph.ndata['feature_normed'] -= graph.ndata['feature_normed'].min(0, keepdim=True)[0] # take min value per column
                            graph.ndata['feature_normed'] /= graph.ndata['feature_normed'].max(0, keepdim=True)[0] + EPS # N by F
                            graph.ndata['feature_normed'] = torch.norm(graph.ndata['feature_normed'], dim=1) # L2 Norm per node dim: N by 1
                    if khop !=0 :
                        sp_matrix_graph = dgl.graph(([], []))
                        sp_matrix_graph.add_nodes(graph.num_nodes()) # keep the node num same
                        if self.sp_method == 'star':
                            assert khop == 1
                            transform = KHopGraph(khop)
                            if j <= 2:
                                print("### sp func: what is transform: ",transform)
                            tmp_graph = transform(graph)
                            tmp_graph = tmp_graph.to_simple()
                            tmp_graph = tmp_graph.remove_self_loop()
                            if j <= 2:
                                print("### spfunc: graph after k-hop transform edges:", tmp_graph, "edges", tmp_graph.edges())
                            j+=1
                        elif self.sp_method == 'convtree':
                            assert khop == 2
                            # we directly use the big graph
                            tmp_graph= graph
                        elif self.sp_method == 'khop' or self.sp_method == 'rand':
                            transform = KHopGraph(khop)
                            tmp_graph = transform(graph)
                            tmp_graph = tmp_graph.to_simple()
                            tmp_graph = tmp_graph.remove_self_loop()
                        i=0
                        for central_node_id in graph.nodes():
                            if i <= 2:
                                print("### sp func: central_node_id ", central_node_id, central_node_id.item())
                            adj_list, weight_list = self.get_sp_adj_list(tmp_graph, central_node_id.item(), khop, self.select_topk_fn)
                            if i <= 2:
                                print("### sp func: adj_list ", adj_list)
                            i+=1
                            sp_matrix_graph.add_edges(adj_list, central_node_id.long(), {'pw': torch.tensor(weight_list) }) # adj_list->node_id, edata['pw'] = weights
                        
                        self.sp_matrix_graph_train_list.append(sp_matrix_graph)
                    else:
                        self.sp_matrix_graph_train_list.append(dgl.graph(([], []))) # make a empty graph
                    if self.agg_ft == 'norm':
                        graph.ndata.pop('feature_normed') # remove normed feature
    
            save_graphs(self.sp_matrix_graphs_train_filename, self.sp_matrix_graph_train_list)
            print("### util: finished training sp graphs generation ")

        if load_kg and os.path.exists(self.sp_matrix_graphs_val_filename):
            self.sp_matrix_graph_list, _ = load_graphs(self.sp_matrix_graphs_val_filename)
        else:
            # print("### util: graph list len: ", len(self.validation_graph_sampled))
            for idx,graph in enumerate(tqdm(self.validation_graph_sampled)):
                with graph.local_scope():
                    if self.agg_ft == 'norm':
                        if self.full_name.endswith("mutag0"):
                            graph.ndata['feature_normed'] =  graph.ndata['feature'].argmax(dim=1)
                        else:
                            graph.ndata['feature_normed'] =  graph.ndata['feature']
                            # norm it
                            graph.ndata['feature_normed'] -= graph.ndata['feature_normed'].min(0, keepdim=True)[0] # take min value per column
                            graph.ndata['feature_normed'] /= graph.ndata['feature_normed'].max(0, keepdim=True)[0] + EPS # N by F
                            graph.ndata['feature_normed'] = torch.norm(graph.ndata['feature_normed'], dim=1) # L2 Norm per node dim: N by 1
                    if khop !=0 :
                        sp_matrix_graph = dgl.graph(([], []))
                        sp_matrix_graph.add_nodes(graph.num_nodes()) # keep the node num same
                        if self.sp_method == 'star':
                            assert khop == 1
                            transform = KHopGraph(khop)
                            # print("### sp func: what is transform: ",transform)
                            tmp_graph = transform(graph)
                            tmp_graph = tmp_graph.to_simple()
                            tmp_graph = tmp_graph.remove_self_loop()
                            # print("### spfunc: graph after k-hop transform edges:", tmp_graph, "edges", tmp_graph.edges())
                        elif self.sp_method == 'convtree':
                            assert khop == 2
                            # we directly use the big graph
                            tmp_graph= graph
                        elif self.sp_method == 'khop' or self.sp_method == 'rand':
                            transform = KHopGraph(khop)
                            tmp_graph = transform(graph)
                            tmp_graph = tmp_graph.to_simple()
                            tmp_graph = tmp_graph.remove_self_loop()
                        for central_node_id in graph.nodes():
                            # print("### sp func: central_node_id ", central_node_id, central_node_id.item())
                            adj_list, weight_list = self.get_sp_adj_list(tmp_graph, central_node_id.item(), khop, self.select_topk_fn)
                            # print("### sp func: adj_list ", adj_list)
                            sp_matrix_graph.add_edges(adj_list, central_node_id.long(), {'pw': torch.tensor(weight_list) }) # adj_list->node_id, edata['pw'] = weights
                        
                        self.sp_matrix_graph_val_list.append(sp_matrix_graph)
                    else:
                        self.sp_matrix_graph_val_list.append(dgl.graph(([], []))) # make a empty graph
                    if self.agg_ft == 'norm':
                        graph.ndata.pop('feature_normed') # remove normed feature
    
            save_graphs(self.sp_matrix_graphs_val_filename, self.sp_matrix_graph_val_list)
            print("### util: finished validation sp graphs generation ")

        if load_kg and os.path.exists(self.sp_matrix_graphs_test_filename):
            self.sp_matrix_graph_list, _ = load_graphs(self.sp_matrix_graphs_test_filename)
        else:
            print("### util: graph list len: ", len(self.testing_graph_sampled))
            for idx,graph in enumerate(tqdm(self.testing_graph_sampled)):
                with graph.local_scope():
                    if self.agg_ft == 'norm':
                        if self.full_name.endswith("mutag0"):
                            graph.ndata['feature_normed'] =  graph.ndata['feature'].argmax(dim=1)
                        else:
                            graph.ndata['feature_normed'] =  graph.ndata['feature']
                            # norm it
                            graph.ndata['feature_normed'] -= graph.ndata['feature_normed'].min(0, keepdim=True)[0] # take min value per column
                            graph.ndata['feature_normed'] /= graph.ndata['feature_normed'].max(0, keepdim=True)[0] + EPS # N by F
                            graph.ndata['feature_normed'] = torch.norm(graph.ndata['feature_normed'], dim=1) # L2 Norm per node dim: N by 1
                    if khop !=0 :
                        sp_matrix_graph = dgl.graph(([], []))
                        sp_matrix_graph.add_nodes(graph.num_nodes()) # keep the node num same
                        if self.sp_method == 'star':
                            assert khop == 1
                            transform = KHopGraph(khop)
                            print("### sp func: what is transform: ",transform)
                            tmp_graph = transform(graph)
                            tmp_graph = tmp_graph.to_simple()
                            tmp_graph = tmp_graph.remove_self_loop()
                            print("### spfunc: graph after k-hop transform edges:", tmp_graph, "edges", tmp_graph.edges())
                        elif self.sp_method == 'convtree':
                            assert khop == 2
                            # we directly use the big graph
                            tmp_graph= graph
                        elif self.sp_method == 'khop' or self.sp_method == 'rand':
                            transform = KHopGraph(khop)
                            tmp_graph = transform(graph)
                            tmp_graph = tmp_graph.to_simple()
                            tmp_graph = tmp_graph.remove_self_loop()
                        for central_node_id in graph.nodes():
                            print("### sp func: central_node_id ", central_node_id, central_node_id.item(), len(graph.nodes()))
                            adj_list, weight_list = self.get_sp_adj_list(tmp_graph, central_node_id.item(), khop, self.select_topk_fn)
                            print("### sp func: adj_list ", adj_list)
                            sp_matrix_graph.add_edges(adj_list, central_node_id.long(), {'pw': torch.tensor(weight_list) }) # adj_list->node_id, edata['pw'] = weights
                        
                        self.sp_matrix_graph_test_list.append(sp_matrix_graph)
                    else:
                        self.sp_matrix_graph_test_list.append(dgl.graph(([], []))) # make a empty graph
                    if self.agg_ft == 'norm':
                        graph.ndata.pop('feature_normed') # remove normed feature
    
            save_graphs(self.sp_matrix_graphs_test_filename, self.sp_matrix_graph_test_list)
            print("### util: finished testing sp graphs generation ")

        if khop != 0:
            # fix nan
            for kg in self.sp_matrix_graph_test_list:
                kg.edata['pw'] = torch.nan_to_num(kg.edata['pw'])
            for kg in self.sp_matrix_graph_train_list:
                kg.edata['pw'] = torch.nan_to_num(kg.edata['pw'])
            for kg in self.sp_matrix_graph_val_list:
                kg.edata['pw'] = torch.nan_to_num(kg.edata['pw'])

        self.make_sp_matrix_graph_list_done = True
        return
        

    def prepare_dataset(self):
        if self.prepare_dataset_done:
            return
        
        self.node_label = []
        self.edge_label = []

        self.training_graph_nodes = []
        self.training_graph_sampled = []
        self.training_graph_edges = []

        self.validation_graph_nodes = []
        self.validation_graph_sampled = []
        self.validation_graph_edges = []

        self.testing_graph_nodes = []
        self.testing_graph_sampled = []
        self.testing_graph_edges = []

        self.node_test_masks = []
        
        # some preprocess
        for idx,graph in enumerate(tqdm(self.graph_list)):
            graph.ndata['feature'] = graph.ndata['feature'].float()
            if 'n' in self.labels_have:
                self.node_label.append(graph.ndata['node_label'])
            if 'e' in self.labels_have:
                self.edge_label.append(graph.edata['edge_label'])

        if len(self.graph_list) == 1:
            self.original_graph = self.graph_list[0]
            self.is_single_graph = True
            node_labels = self.node_label[0]
            num_nodes = self.graph_list[0].num_nodes()
            all_node_ids = list(range(num_nodes))
            zero_labeled = [n for n, l in zip(all_node_ids, node_labels) if l == 0]
            one_labeled = [n for n, l in zip(all_node_ids, node_labels) if l == 1]
            print("zero labeled ", zero_labeled[:50])
            print("one labeled ", one_labeled[:50])
            for i in range(100):
                print("sampling training graph ", i)
                seed = ROOT_SEED+100*i
                set_seed(seed)
                sample_zeros = random.sample(zero_labeled, min(10, len(zero_labeled)))
                sample_ones  = random.sample(one_labeled, min(10, len(one_labeled)))
                if i <=2:
                    print("sampled zeros ", sample_zeros[:10])
                    print("sampled ones ", sample_ones[:10])
                k=2
                for _ in range(k):
                    one_labeled_nodes = torch.tensor(sample_ones).long()
                    for n in one_labeled_nodes:
                        pres = self.original_graph.predecessors(n)
                        sucs = self.original_graph.successors(n)
                        neighbors = torch.unique(torch.cat([pres, sucs], dim=0))
                        for nb in neighbors:
                            if nb.item() in zero_labeled and nb.item() not in sample_zeros:
                                sample_zeros.append(nb.item())
                            if nb.item() in one_labeled and nb.item() not in sample_ones:
                                sample_ones.append(nb.item())  

                if i <= 2:
                    print("after expand sampled zeros ", sample_zeros[:10], len(sample_zeros))
                    print("after expand sampled ones ", sample_ones[:10], len(sample_ones))
                selected_node_ids = sample_zeros + sample_ones
                selected_node_ids = torch.tensor(selected_node_ids).long()
                self.training_graph_nodes.append(selected_node_ids)
                sampled_graph = dgl.node_subgraph(self.original_graph, selected_node_ids, store_ids=True)
                if i <=2 :
                    print("few nodes from sampled graph: ", sampled_graph.ndata[dgl.NID][:10])
                self.training_graph_sampled.append(sampled_graph)
                self.training_graph_edges.append(sampled_graph.edata[dgl.EID])

            print("traing graph sampled num: ", len(self.training_graph_sampled))

            for i in range(50):
                seed = ROOT_SEED+50*i
                set_seed(seed)
                sample_zeros = random.sample(zero_labeled, min(10, len(zero_labeled)))
                sample_ones  = random.sample(one_labeled, min(10, len(one_labeled)))
                k=2
                for _ in range(k):
                    one_labeled_nodes = torch.tensor(sample_ones).long()
                    for n in one_labeled_nodes:
                        pres = self.original_graph.predecessors(n)
                        sucs = self.original_graph.successors(n)
                        neighbors = torch.unique(torch.cat([pres, sucs], dim=0))
                        for nb in neighbors:
                            if nb.item() in zero_labeled and nb.item() not in sample_zeros:
                                sample_zeros.append(nb.item())
                            if nb.item() in one_labeled and nb.item() not in sample_ones:
                                sample_ones.append(nb.item())  
                selected_node_ids = sample_zeros + sample_ones
                selected_node_ids = torch.tensor(selected_node_ids).long()
                self.validation_graph_nodes.append(selected_node_ids)
                sampled_graph = dgl.node_subgraph(self.original_graph, selected_node_ids, store_ids=True)
                self.validation_graph_sampled.append(sampled_graph)
                self.validation_graph_edges.append(sampled_graph.edata[dgl.EID])

            print("validation graph sampled num: ", len(self.validation_graph_sampled))

            for i in range(50):
                seed = ROOT_SEED+50*i
                set_seed(seed)
                sample_zeros = random.sample(zero_labeled, min(10, len(zero_labeled)))
                sample_ones  = random.sample(one_labeled, min(10, len(one_labeled)))
                k=2
                for _ in range(k):
                    one_labeled_nodes = torch.tensor(sample_ones).long()
                    for n in one_labeled_nodes:
                        pres = self.original_graph.predecessors(n)
                        sucs = self.original_graph.successors(n)
                        neighbors = torch.unique(torch.cat([pres, sucs], dim=0))
                        for nb in neighbors:
                            if nb.item() in zero_labeled and nb.item() not in sample_zeros:
                                sample_zeros.append(nb.item())
                            if nb.item() in one_labeled and nb.item() not in sample_ones:
                                sample_ones.append(nb.item())  
                selected_node_ids = sample_zeros + sample_ones
                selected_node_ids = torch.tensor(selected_node_ids).long()
                self.testing_graph_nodes.append(selected_node_ids)
                sampled_graph = dgl.node_subgraph(self.original_graph, selected_node_ids, store_ids=True)
                self.testing_graph_sampled.append(sampled_graph)
                self.testing_graph_edges.append(sampled_graph.edata[dgl.EID])

            print("testing graph sampled num: ", len(self.testing_graph_sampled))
       
        self.prepare_dataset_done = True
        print("### util: dataset prepared.")

        return

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default="", help='addtional tags for distinguish result')
    parser.add_argument('--khop', type=int, default=0)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr_ft', type=float, default=0.003)
    parser.add_argument("--l2_ft", type=float, default=0)
    parser.add_argument('--epoch_ft', type=int, default=200)
    parser.add_argument("--stitch_mlp_layers", type=int, default=1, help="Number of hidden layer in stitch MLP")
    parser.add_argument("--final_mlp_layers", type=int, default=2, help="Number of hidden layer in final MLP")
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--metric', type=str, default='AUROC')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--task_level', type=str, default='unify')
    parser.add_argument('--node_loss_weight', type=float, default=1)
    parser.add_argument('--edge_loss_weight', type=float, default=1)
    parser.add_argument('--graph_loss_weight', type=float, default=1)
    parser.add_argument('--cross_modes', type=str, default="ng2ng")
    parser.add_argument('--sp_type', type=str, default='star+union', help="neighbor sampling strategy")
    parser.add_argument('--force_remake_sp',  action="store_true", help="force remaking neighbor sampling matrix")
    # pretrain model parameters
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument('--epoch_pretrain', type=int, default=100)
    parser.add_argument('--pretrain_model', type=str, default='graphmae')
    parser.add_argument('--kernels', type=str, default='gcn', help="Encoder/Decode GNN model types")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0, help="Coefficient of L2 penalty")
    parser.add_argument("--decay_rate", type=float, default=1, help="Decay rate of learning rate")
    parser.add_argument("--decay_step", type=int, default=100, help="Decay step of learning rate")
    parser.add_argument('--drop_rate', type=float, default=0)
    parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dimension")
    parser.add_argument("--num_layer_pretrain", type=int, default=2, help="Number of hidden layer in pretrain model")
    parser.add_argument("--act", type=str, default='leakyrelu', help="Activation function type")
    parser.add_argument("--act_ft", type=str, default='ReLU', help="Activation function for mlp")
    parser.add_argument("--norm", type=str, default="", help="Normlaization layer type")
    parser.add_argument("--concat", action="store_true", default=False, help="Indicator of where using raw and generated embeddings")
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    # GraphMAE
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Masking ratio for GraphMAE")
    parser.add_argument("--replace_ratio", type=float, default=0, help="Replace ratio for GraphMAE")
    
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate for node in training")
    
    
    args = parser.parse_args()
    return args