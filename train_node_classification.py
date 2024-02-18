import os
import sys
import numpy as np
import pickle
import scipy.sparse as sp
import logging
import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from model_node_classification import Model
from preprocess_node_classification import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor
from utils import *
import networkx as nx
from llm_component import convert_to_networkx_graph
import pdb

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--dataset', type=str, default='ACM')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50, help='maximum number of training epochs')
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--no_norm', action='store_true', default=False, help='disable layer norm')
parser.add_argument('--in_nl', action='store_true', default=False, help='non-linearity after projection')
parser.add_argument('--attn_dim', type=int, default=128)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--num_structures', type=int, default=1)
parser.add_argument('--population_size', type=int, default=5)
args = parser.parse_args()

prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + "_h" + str(args.n_hid) + \
         "_drop" + str(args.dropout) + "_epoch" + str(args.epochs) + "_cuda" + str(args.gpu)

if args.no_norm is True:
    prefix += "_noLN"
if args.in_nl is True:
    prefix += "_nl"

logdir = os.path.join("log_node_classification/eval", args.dataset)
if not os.path.exists(logdir):
    os.makedirs(logdir)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(logdir, prefix + ".txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():

    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if(args.dataset=='IMDB'): 
        dir = f'./log_node_classification/train/IMDB_lr0.005 (best test 0.6091)'
    elif(args.dataset=='ACM'): dir = f'./log_node_classification/train/ACM_lr0.005 (best test 0.9227)'
    gene_pools_history_dict_path = os.path.join(dir, 'gene_pools_history_dict.pkl')
    gene_pools_performance_dict_path = os.path.join(dir, 'gene_pools_performance_dict.pkl')

    with open(gene_pools_history_dict_path, 'rb') as f:
        gene_pools_history_dict = pickle.load(f)
    with open(gene_pools_performance_dict_path, 'rb') as f:
        gene_pools_performance_dict = pickle.load(f)
    print(gene_pools_history_dict.keys())
    gene_pools_best_val_dict = gene_pools_performance_dict['best_val']
    gene_pools_correspond_test_dict = gene_pools_performance_dict['correspond_test']
    best_val_array = np.zeros((len(gene_pools_history_dict.keys()), args.population_size))
    for generation,perfs in gene_pools_best_val_dict.items():
        best_val_array[generation] = perfs
    # Flatten the array to find indices of the largest elements
    flat_indices = np.argsort(best_val_array.flatten())[-args.num_structures:]
    # Convert flat indices to (row, col) indices
    row_indices, col_indices = np.unravel_index(flat_indices, best_val_array.shape)
    gene_comb = [gene_pools_history_dict[row_indices[i]][col_indices[i]] for i in range(args.num_structures)]
    individual_best_val = [gene_pools_best_val_dict[row_indices[i]][col_indices[i]] for i in range(args.num_structures)]
    individual_correspond_test = [gene_pools_correspond_test_dict[row_indices[i]][col_indices[i]] for i in range(args.num_structures)]
    
    print('Structures: ', gene_comb)
    print('Individual Best val: ', individual_best_val)
    print('Individual Correspond.test: ', individual_correspond_test)

    # Set a meta
    test_structure_list_sym = [gene_comb[i] for i in range(args.num_structures)]
    print('len(test_structure_list_sym): ', len(test_structure_list_sym))
    pdb.set_trace()

    archs = {args.dataset: ([],[])} # Initialization
    edge_type_lookup_dict = temp_edge_type_lookup(args.dataset.lower())

    for test_structure in test_structure_list_sym:
        G = convert_to_networkx_graph({'nodes': test_structure[0], 'edges': test_structure[1]})
        state_list = get_state_list(G, 0)
        arch = construct_arch(G, state_list, edge_type_lookup_dict) 
        archs[args.dataset][0].append(arch[0])
        archs[args.dataset][1].append(arch[1])
    
    steps = [len(meta) for meta in archs[args.dataset][0]]
    print("Steps: {}".format(steps))

    datadir = 'data_node_classification' 
    prefix = os.path.join(datadir, args.dataset)

    #* load data
    with open(os.path.join(prefix, "node_features.pkl"), "rb") as f:
        node_feats = pickle.load(f)
        f.close()
    node_feats = torch.from_numpy(node_feats.astype(np.float32)).cuda()

    node_types = np.load(os.path.join(prefix, "node_types.npy"))
    num_node_types = node_types.max() + 1
    node_types = torch.from_numpy(node_types).cuda()

    with open(os.path.join(prefix, "edges.pkl"), "rb") as f:
        edges = pickle.load(f)
        f.close()
    
    adjs_pt = []
    for mx in edges:
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(mx.astype(np.float32) + sp.eye(mx.shape[0], dtype=np.float32))).cuda())
    adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(edges[0].shape[0], dtype=np.float32).tocoo()).cuda())
    adjs_pt.append(torch.sparse.FloatTensor(size=edges[0].shape).cuda())
    print("Loading {} adjs...".format(len(adjs_pt)))

    #* load labels
    with open(os.path.join(prefix, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)
        f.close()
    
    train_idx = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.long).cuda()
    train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.long).cuda()
    valid_idx = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.long).cuda()
    valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.long).cuda()
    test_idx = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.long).cuda()
    test_target = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.long).cuda()

    n_classes = train_target.max().item() + 1
    print("Number of classes: {}".format(n_classes), "Number of node types: {}".format(num_node_types))

    model = Model(node_feats.size(1), args.n_hid, num_node_types, n_classes, steps, dropout = args.dropout, use_norm = not args.no_norm, in_nl = args.in_nl).cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    best_val = None
    final = None
    anchor = None
    patience = 0
    for epoch in range(args.epochs):
        train_loss = train(archs, node_feats, node_types, adjs_pt, train_idx, train_target, model, optimizer)
        val_loss, f1_val, f1_test = infer(archs, node_feats, node_types, adjs_pt, valid_idx, valid_target, test_idx, test_target, model)
        logging.info("Epoch {}; Train err {}; Val err {}".format(epoch + 1, train_loss, val_loss))
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            final = f1_test
            anchor = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience == 10:
                break
    logging.info("Best val {} at epoch {}; Test score {}".format(best_val, anchor, final))

def train(archs, node_feats, node_types, adjs, train_idx, train_target, model, optimizer):

    model.train()
    optimizer.zero_grad()
    out = model(node_feats, node_types, adjs, archs[args.dataset][0], archs[args.dataset][1])
    loss = F.cross_entropy(out[train_idx], train_target)
    loss.backward()
    optimizer.step()
    return loss.item()

def infer(archs, node_feats, node_types, adjs, valid_idx, valid_target, test_idx, test_target, model):

    model.eval()
    with torch.no_grad():
        out = model(node_feats, node_types, adjs, archs[args.dataset][0], archs[args.dataset][1])
    loss = F.cross_entropy(out[valid_idx], valid_target)
    f1_val = f1_score(valid_target.cpu().numpy(), torch.argmax(out[valid_idx], dim=-1).cpu().numpy(), average='macro')
    f1_test = f1_score(test_target.cpu().numpy(), torch.argmax(out[test_idx], dim=-1).cpu().numpy(), average='macro')
    return loss.item(), f1_val, f1_test

def get_state_list(G, target):
    # Generate BFS tree by searching reversely from the target node
    bfs_tree_result = nx.bfs_tree(G, target)
    bfs_node_order = [target]
    for edge in list(bfs_tree_result.edges()):
        bfs_node_order.append(edge[1])
    #print(bfs_node_order)
    return bfs_node_order[::-1] # target node ranks last

def construct_arch(G, state_list, edge_type_lookup_dict):
    connection_dict = get_connection_dict(G, state_list)
    seq_arch, res_arch = [], []
    for i in range(1, len(state_list)):
        this_node = state_list[i]
        this_neighbors = connection_dict[this_node]     
        if(state_list[i-1] in this_neighbors):
            this_edge_type = G.nodes[this_node]['type'][0] + G.nodes[state_list[i-1]]['type'][0]
            seq_arch.append(edge_type_lookup_dict[this_edge_type])
        else:
            seq_arch.append(edge_type_lookup_dict['O'])
    for i in range(len(state_list)):
        this_node = state_list[i]
        for j in range(i+2, len(G.nodes)):
            if(this_node in connection_dict[state_list[j]]):
                this_edge_type = G.nodes[state_list[j]]['type'][0] + G.nodes[this_node]['type'][0]
                try:
                    res_arch.append(edge_type_lookup_dict[this_edge_type])
                except:pdb.set_trace()
            else:
                res_arch.append(edge_type_lookup_dict['O'])
    meta_arch = (seq_arch, res_arch)
    return meta_arch

def get_connection_dict(G, state_list):
    connection_dict = dict()
    state_order_dict = dict(zip(state_list, np.arange(len(state_list))))
    for state in state_list:
        connection_dict[state] = [neighbor for neighbor in list(G.neighbors(state)) if state_order_dict[neighbor] < state_order_dict[state]]
    return connection_dict

if __name__ == '__main__':
    main()
