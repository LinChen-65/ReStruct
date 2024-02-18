from llm_component_no_prob import LLM4Meta
from utils import *
import os
import argparse
import logging
import sys
from model import Model
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import pdb
import numpy as np
import time
import networkx as nx
import pickle

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--dataset', type=str, default='Yelp')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs') 
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--non_symmetric', default=False, action='store_true')
parser.add_argument('--num_generations', type=int, default=20)
parser.add_argument('--dataset_seed', type=int, default=2)
parser.add_argument('--num_neighbors_for_explanation', type=int, default=3)

args = parser.parse_args()

GPT_MODEL = "gpt-4-1106-preview"

api_key_list=[
    # Put your API keys here
]
api_key = api_key_list[-1]

client = OpenAI(api_key=api_key, http_client=httpx.Client(
        #proxies=proxies['https'],
    ),)


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

    #* test performance
    pos_test_prod = torch.mul(out_s[pos_test[:, 0]], out_t[pos_test[:, 1]]).sum(dim=-1)
    neg_test_prod = torch.mul(out_s[neg_test[:, 0]], out_t[neg_test[:, 1]]).sum(dim=-1)

    y_true_test = np.zeros((pos_test.shape[0] + neg_test.shape[0]), dtype=np.long)
    y_true_test[:pos_test.shape[0]] = 1
    y_pred_test = np.concatenate((torch.sigmoid(pos_test_prod).cpu().numpy(), torch.sigmoid(neg_test_prod).cpu().numpy()))
    #pdb.set_trace()
    auc_test = roc_auc_score(y_true_test, y_pred_test)
    
    return loss.item(), auc_val, auc_test

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


def load_data(datadir='DiffMG_preprocessed'):
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
    print(pos_train.shape, pos_val.shape, pos_test.shape)

    neg = np.load(os.path.join(prefix, f"neg_pairs_offset_for_pos_larger_than_2_{args.dataset_seed}.npz"))
    neg_train = neg['train']
    neg_val = neg['val']
    neg_test = neg['test']
    print(neg_train.shape, neg_val.shape, neg_test.shape)

    #* one-hot IDs as input features
    in_dims = []
    node_feats = []
    for k in range(num_node_types):
        in_dims.append((node_types == k).sum().item())
        i = torch.stack((torch.arange(in_dims[-1], dtype=torch.long), torch.arange(in_dims[-1], dtype=torch.long)))
        v = torch.ones(in_dims[-1])
        node_feats.append(torch.sparse.FloatTensor(i, v, torch.Size([in_dims[-1], in_dims[-1]])).cuda())
    assert(len(in_dims) == len(node_feats))   
    return node_types, num_node_types, adjs_pt, pos_train, pos_val, pos_test, neg_train, neg_val, neg_test, in_dims, node_feats


def structure2arch(test_structure_list_sym=[], test_structure_list_source=[], test_structure_list_target=[]):
    if(args.non_symmetric):
        assert (len(test_structure_list_source)>0 & len(test_structure_list_target)>0)
    else:
        assert len(test_structure_list_sym)>0
    
    archs = {args.dataset: {'source': ([],[]), 'target': ([],[])}} # Initialization
    edge_type_lookup_dict = temp_edge_type_lookup(dataset_string)
    if(not args.non_symmetric):
        for test_structure in test_structure_list_sym:
            G = convert_to_networkx_graph({'nodes': test_structure[0], 'edges': test_structure[1]})
            state_list = get_state_list(G, 1)
            arch_target = construct_arch(G, state_list, edge_type_lookup_dict) # target node is B
            state_list = get_state_list(G, 0)
            arch_source = construct_arch(G, state_list, edge_type_lookup_dict) # target node is U
            archs[args.dataset]['source'][0].append(arch_source[0])
            archs[args.dataset]['source'][1].append(arch_source[1])
            archs[args.dataset]['target'][0].append(arch_target[0])
            archs[args.dataset]['target'][1].append(arch_target[1])
    else:
        for test_structure in test_structure_list_source:
            G = convert_to_networkx_graph({'nodes': test_structure[0], 'edges': test_structure[1]})
            state_list = get_state_list(G, 1)
            arch_source = construct_arch(G, state_list, edge_type_lookup_dict) # target node is B
            archs[args.dataset]['source'][0].append(arch_source[0])
            archs[args.dataset]['source'][1].append(arch_source[1])
        for test_structure in test_structure_list_target:   
            G = convert_to_networkx_graph({'nodes': test_structure[0], 'edges': test_structure[1]})
            state_list = get_state_list(G, 1)
            arch_target = construct_arch(G, state_list, edge_type_lookup_dict) # target node is U
            archs[args.dataset]['target'][0].append(arch_target[0])
            archs[args.dataset]['target'][1].append(arch_target[1])
    return archs


def evaluate(gene_pools): 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    archs = structure2arch(gene_pools)

    steps_s = [len(meta) for meta in archs[args.dataset]["source"][0]] # steps_s: [4]
    steps_t = [len(meta) for meta in archs[args.dataset]["target"][0]] # steps_t: [6]

    model_s = Model(in_dims, args.n_hid, steps_s, dropout = args.dropout).cuda()
    model_t = Model(in_dims, args.n_hid, steps_t, dropout = args.dropout).cuda()

    connection_dict_s = {}
    connection_dict_t = {}

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
        val_loss, auc_val, auc_test = infer(archs, node_feats, node_types, adjs_pt, pos_val, neg_val, pos_test, neg_test, model_s, model_t, connection_dict_s, connection_dict_t)
        if(epoch%50==0):
            logging.info("Epoch {}; Train err {}; Val err {}; Val auc {}".format(epoch + 1, train_loss, val_loss, auc_val))
        if best_val is None or auc_val > best_val:
            best_val = auc_val
            final = auc_test
            anchor = epoch + 1
    logging.info("Best val auc {} at epoch {}; Test auc {}".format(best_val, anchor, final))
    
    return best_val, final


def matrix2logic(matrix, dataset_string):
    nx_meta = convert_to_networkx_graph({'nodes': matrix[0], 'edges': matrix[1]})
    logic = graph2logic(nx_meta, dataset_string, initialization=True)        
    return logic


if __name__ == "__main__":

    prefix = 'llm_explanation'
    logdir = os.path.join(f"log_recommendation/explanation", args.dataset)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    else:
        print('Already logdir??')
        pdb.set_trace()
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    fh = logging.FileHandler(os.path.join(logdir, prefix + ".txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    dataset_string = args.dataset.lower() #'douban_movie' #'yelp'
    print('dataset_string: ', dataset_string)
    task_string = 'recommendation'
    dataset_task_string = dataset_string + '_' + task_string #dataset_task_string = 'yelp_recommendation'

    # Load data
    node_types, num_node_types, adjs_pt, pos_train, pos_val, pos_test, neg_train, neg_val, neg_test, in_dims, node_feats = load_data(datadir='new_preprocessed_recommendation')

    # cuda settings
    torch.cuda.set_device(args.gpu)
    
    # LLM intialization
    trial='_component_v3_diffmgdata_diffmgeval' 
    dialogs_save_path = logdir #f'./data/gpt4{trial}'
    llm4meta = LLM4Meta(client=client, dataset=dataset_string, downstream_task=task_string, dialogs_save_path=dialogs_save_path)
    
    # Yelp
    best_structures_from_dataset =  [
        [['U', 'B', 'A', 'B', 'I', 'B', 'U', 'B'], [[0, 1, 0, 1, 0, 1, 1, 1], [1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0, 1, 0]]], 
        [['U', 'B', 'A', 'B', 'U', 'I'], [[0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 1, 1], [1, 0, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0]]],
        [['U', 'B', 'I', 'B', 'A', 'B'], [[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0]]]
    ]

    start = time.time()
    rank = 7#0
    for structure in best_structures_from_dataset:
        structure_perf, final_test = evaluate([structure])

        logic_neighbors, structure_neighbors_dict = llm4meta.modify_metas(0, [structure], [], '', '', {}, get_all_candidates=True)
        
        # Sample neighbors
        indices = np.arange(len(logic_neighbors))
        sampled_indices = np.random.choice(indices, args.num_neighbors_for_explanation)
        logic_neighbors = list(np.array(logic_neighbors)[sampled_indices])
        structure_neighbors_dict = list(np.array(structure_neighbors_dict)[sampled_indices])
        # dict to list
        structure_neighbors = []
        for i in range(len(structure_neighbors_dict)):
            structure_neighbors.append([structure_neighbors_dict[i]['nodes'], structure_neighbors_dict[i]['edges']])
        
        performance_list = []
        neighbor_perf_matrix_str = ''
        neighbor_perf_grammar_str = ''
        for i in range(len(structure_neighbors)):
            neigh_matrix = structure_neighbors[i]
            neigh_grammar = logic_neighbors[i]
            performance, final_test = evaluate([neigh_matrix])
            performance_list.append(performance)
            neighbor_perf_matrix_str += f'{neigh_matrix}, performance: {np.round(performance,4)}'
            neighbor_perf_grammar_str += f'{neigh_grammar}, performance: {np.round(performance,4)}'

        explanation = llm4meta.generate_explanation(rank, structure, structure_perf, structure_neighbors, performance_list, grammar = True, diff=True)
        
        print('Used time: ', time.time()-start) # 600s for one explanation with diff=True
        rank += 1
