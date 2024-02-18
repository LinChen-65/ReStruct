import os
import sys
import numpy as np
import pickle
import scipy.sparse as sp
import logging
import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from model_attention_improve_xfl import Model
from preprocess_recommendation import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor
from utils import *

import networkx as nx
from llm_component import convert_to_networkx_graph
import pdb

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--dataset', type=str, default='Yelp')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--dropout', type=float, default=0.2) 
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--non_symmetric', default=False, action='store_true')
parser.add_argument('--attn_dim', type=int, default=128)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--num_structures', type=int, default=1)
parser.add_argument('--loss_margin', type=float, default=0.3)
parser.add_argument('--dataset_seed', type=int, default=2)
parser.add_argument('--population_size', type=int, default=5)
args = parser.parse_args()

prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + "_h" + str(args.n_hid) + \
         "_drop" + str(args.dropout) + "_epoch" + str(args.epochs) + "_cuda" + str(args.gpu)

logdir = os.path.join("log/eval", args.dataset)
if not os.path.exists(logdir):
    os.makedirs(logdir)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(logdir, prefix + ".txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if(args.num_structures==1):
    single=True
else: 
    single=False

def main():
    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    connection_dict_s = {}
    connection_dict_t = {}

    dir = f'./log_recommendation/train_threshold_2_datasetseed_{args.dataset_seed}/{args.dataset}' 
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
    flat_indices = np.argsort(best_val_array.flatten())[-args.num_structures:][::-1]
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
    test_structure_list_source = [gene_comb[int(2*i)] for i in range(int(args.num_structures/2))] 
    test_structure_list_target = [gene_comb[int(2*i+1)] for i in range(int(args.num_structures/2))] 
    pdb.set_trace()

    archs = {args.dataset: {'source': ([],[]), 'target': ([],[])}} # Initialization
    edge_type_lookup_dict = temp_edge_type_lookup(args.dataset.lower())
    if(not args.non_symmetric):
        for test_structure in test_structure_list_sym:
            G = convert_to_networkx_graph({'nodes': test_structure[0], 'edges': test_structure[1]})
            state_list = get_state_list(G, 1)
            arch_target = construct_arch(G, state_list, edge_type_lookup_dict) 
            state_list = get_state_list(G, 0)
            arch_source = construct_arch(G, state_list, edge_type_lookup_dict) 
            archs[args.dataset]['source'][0].append(arch_source[0])
            archs[args.dataset]['source'][1].append(arch_source[1])
            archs[args.dataset]['target'][0].append(arch_target[0])
            archs[args.dataset]['target'][1].append(arch_target[1])
    else:
        for test_structure in test_structure_list_source:
            G = convert_to_networkx_graph({'nodes': test_structure[0], 'edges': test_structure[1]})
            state_list = get_state_list(G, 1)
            arch_source = construct_arch(G, state_list, edge_type_lookup_dict) 
            archs[args.dataset]['source'][0].append(arch_source[0])
            archs[args.dataset]['source'][1].append(arch_source[1])
        for test_structure in test_structure_list_target:   
            G = convert_to_networkx_graph({'nodes': test_structure[0], 'edges': test_structure[1]})
            state_list = get_state_list(G, 1)
            arch_target = construct_arch(G, state_list, edge_type_lookup_dict) 
            archs[args.dataset]['target'][0].append(arch_target[0])
            archs[args.dataset]['target'][1].append(arch_target[1])

    steps_s = [len(meta) for meta in archs[args.dataset]["source"][0]] 
    steps_t = [len(meta) for meta in archs[args.dataset]["target"][0]] 

    datadir = 'data_recommendation' 
    prefix = os.path.join(datadir, args.dataset)

    #* load data
    node_types = np.load(os.path.join(prefix, "node_types.npy"))
    num_node_types = node_types.max() + 1
    node_types = torch.from_numpy(node_types).cuda()

    adjs_offset = pickle.load(open(os.path.join(prefix, "adjs_offset.pkl"), "rb"))
    adjs_pt = []
    if '0' in adjs_offset:
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_sym(adjs_offset['0'] + sp.eye(adjs_offset['0'].shape[0], dtype=np.float32))).cuda())
    for i in range(1, int(max(adjs_offset.keys())) + 1):
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(i)] + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(i)].T + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
    adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(adjs_offset['1'].shape[0], dtype=np.float32).tocoo()).cuda())
    adjs_pt.append(torch.sparse.FloatTensor(size=adjs_offset['1'].shape).cuda())
    print("Loading {} adjs...".format(len(adjs_pt)))

    #* load labels
    pos = np.load(os.path.join(prefix, f"pos_pairs_offset_larger_than_2_{args.dataset_seed}.npz"))
    pos_train = pos['train']
    pos_val = pos['val']
    pos_test = pos['test']

    neg = np.load(os.path.join(prefix, f"neg_pairs_offset_for_pos_larger_than_2_{args.dataset_seed}.npz"))
    neg_train = neg['train']
    neg_val = neg['val']
    neg_test = neg['test']

    #* one-hot IDs as input features
    in_dims = []
    node_feats = []
    for k in range(num_node_types):
        in_dims.append((node_types == k).sum().item())
        i = torch.stack((torch.arange(in_dims[-1], dtype=torch.long), torch.arange(in_dims[-1], dtype=torch.long)))
        v = torch.ones(in_dims[-1])
        node_feats.append(torch.sparse.FloatTensor(i, v, torch.Size([in_dims[-1], in_dims[-1]])).cuda())
    assert(len(in_dims) == len(node_feats))   

    model_s = Model(in_dims, args.n_hid, steps_s, dropout = args.dropout, single=single, attn_dim=args.attn_dim, num_heads=args.num_heads).cuda()
    model_t = Model(in_dims, args.n_hid, steps_t, dropout = args.dropout, single=single, attn_dim=args.attn_dim, num_heads=args.num_heads).cuda()

    optimizer = torch.optim.Adam(
        list(model_s.parameters()) + list(model_t.parameters()),
        lr=args.lr,
        weight_decay=args.wd
    )

    best_val = None
    final = None
    anchor = None
    for epoch in range(args.epochs):
        train_loss = train(archs, node_feats, node_types, adjs_pt, pos_train, neg_train, model_s, model_t, connection_dict_s, connection_dict_t, optimizer)
        val_loss, auc_val, auc_test, wrong_predictions = infer(archs, node_feats, node_types, adjs_pt, pos_val, neg_val, pos_test, neg_test, model_s, model_t, connection_dict_s, connection_dict_t)
        logging.info("Epoch {}; Train err {}; Val err {}; Val auc {}".format(epoch + 1, train_loss, val_loss, auc_val))
        if best_val is None or auc_val > best_val:
            best_val = auc_val
            final = auc_test
            anchor = epoch + 1
            final_wrong_predictions = wrong_predictions.copy()
    logging.info("Best val auc {} at epoch {}; Test auc {}".format(best_val, anchor, final))


def train(archs, node_feats, node_types, adjs, pos_train, neg_train, model_s, model_t, connection_dict_s, connection_dict_t, optimizer):

    model_s.train()
    model_t.train()
    optimizer.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, archs[args.dataset]["source"][0], archs[args.dataset]["source"][1], connection_dict_s)
    out_t = model_t(node_feats, node_types, adjs, archs[args.dataset]["target"][0], archs[args.dataset]["target"][1], connection_dict_t)
    loss = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_train[:, 0]], out_t[pos_train[:, 1]]).sum(dim=-1)) + \
                        F.logsigmoid(- torch.mul(out_s[neg_train[:, 0]], out_t[neg_train[:, 1]]).sum(dim=-1)))
    loss.backward()
    optimizer.step()
    return loss.item()

def infer(archs, node_feats, node_types, adjs, pos_val, neg_val, pos_test, neg_test, model_s, model_t, connection_dict_s, connection_dict_t):

    model_s.eval()
    model_t.eval()
    with torch.no_grad():
        out_s = model_s(node_feats, node_types, adjs, archs[args.dataset]["source"][0], archs[args.dataset]["source"][1], connection_dict_s)
        out_t = model_t(node_feats, node_types, adjs, archs[args.dataset]["target"][0], archs[args.dataset]["target"][1], connection_dict_t)
    
    #* validation performance
    pos_val_prod = torch.mul(out_s[pos_val[:, 0]], out_t[pos_val[:, 1]]).sum(dim=-1)
    neg_val_prod = torch.mul(out_s[neg_val[:, 0]], out_t[neg_val[:, 1]]).sum(dim=-1)
    loss = - torch.mean(F.logsigmoid(pos_val_prod) + F.logsigmoid(- neg_val_prod))
    y_true_val = np.zeros((pos_val.shape[0] + neg_val.shape[0]), dtype=np.long)
    y_true_val[:pos_val.shape[0]] = 1
    y_pred_val = np.concatenate((torch.sigmoid(pos_val_prod).cpu().numpy(), torch.sigmoid(neg_val_prod).cpu().numpy()))
    auc_val = roc_auc_score(y_true_val, y_pred_val)
    y_pred_val_binary = np.where(y_pred_val < 0.5, 0, 1)
    wrong_predictions = np.where(y_pred_val_binary != y_true_val)[0]

    #* test performance
    pos_test_prod = torch.mul(out_s[pos_test[:, 0]], out_t[pos_test[:, 1]]).sum(dim=-1)
    neg_test_prod = torch.mul(out_s[neg_test[:, 0]], out_t[neg_test[:, 1]]).sum(dim=-1)

    y_true_test = np.zeros((pos_test.shape[0] + neg_test.shape[0]), dtype=np.long)
    y_true_test[:pos_test.shape[0]] = 1
    y_pred_test = np.concatenate((torch.sigmoid(pos_test_prod).cpu().numpy(), torch.sigmoid(neg_test_prod).cpu().numpy()))
    auc_test = roc_auc_score(y_true_test, y_pred_test)
    
    return loss.item(), auc_val, auc_test, wrong_predictions


def get_state_list(G, target):
    # Generate BFS tree by searching reversely from the target node
    bfs_tree_result = nx.bfs_tree(G, target)
    bfs_node_order = [target]
    for edge in list(bfs_tree_result.edges()):
        bfs_node_order.append(edge[1])
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