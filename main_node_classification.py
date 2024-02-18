import os
import sys
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import time
from datetime import datetime
import argparse
import logging
from model_node_classification import Model
import copy
from llm_component_no_prob import LLM4Meta 
from utils import *
import pdb
import networkx as nx

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
parser.add_argument('--test_known_metas', default=False, action='store_true')
parser.add_argument('--num_generations', type=int, default=20)
parser.add_argument('--no_norm', action='store_true', default=False, help='disable layer norm')
parser.add_argument('--in_nl', action='store_true', default=False, help='non-linearity after projection')
args = parser.parse_args()

api_key_list=[
    # Put your API keys here
] 
GPT_MODEL = "gpt-4-1106-preview"

api_key = api_key_list[0]

client = OpenAI(api_key=api_key, http_client=httpx.Client(
        #proxies=proxies['https'],
    ),)

#########################################################################

if(args.test_known_metas):
    POPULATION_SIZE = 1
else:
    POPULATION_SIZE = 5  # Number of individuals in the population
GENE_NUM = 1  # Number of genes carried by each individual
GENE_POOL_SIZE = POPULATION_SIZE * GENE_NUM
ELIMINATE_RATE = 0.4 

current_date = datetime.now().date().strftime("%Y-%m-%d")

prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + "_h" + str(args.n_hid) + \
         "_drop" + str(args.dropout) + "_epoch" + str(args.epochs) + "_cuda" + str(args.gpu)
if args.no_norm is True:
    prefix += "_noLN"
if args.in_nl is True:
    prefix += "_nl"

logdir = os.path.join("log_node_classification/train", args.dataset + f'_lr{args.lr}') #os.path.join("log/eval", args.dataset)
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

##############################################################################################################

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
    #* validation performance
    f1_val = f1_score(valid_target.cpu().numpy(), torch.argmax(out[valid_idx], dim=-1).cpu().numpy(), average='macro')
    f1_test = f1_score(test_target.cpu().numpy(), torch.argmax(out[test_idx], dim=-1).cpu().numpy(), average='macro')
    return loss.item(), f1_val, f1_test

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


def load_data(datadir):
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

    return node_feats, node_types, adjs_pt, train_idx, train_target, valid_idx, valid_target, test_idx, test_target, n_classes, num_node_types


def structure2arch(test_structure_list_sym=[]):
    assert len(test_structure_list_sym)>0

    archs = {args.dataset: ([],[])} # Initialization
    edge_type_lookup_dict = temp_edge_type_lookup(dataset_string)

    for test_structure in test_structure_list_sym:
        G = convert_to_networkx_graph({'nodes': test_structure[0], 'edges': test_structure[1]})
        state_list = get_state_list(G, 0)
        arch = construct_arch(G, state_list, edge_type_lookup_dict) 
        archs[args.dataset][0].append(arch[0])
        archs[args.dataset][1].append(arch[1])

    return archs

def evaluate(gene_pools): 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    archs = structure2arch(gene_pools)
    steps = [len(meta) for meta in archs[args.dataset][0]] # steps: [4]

    model = Model(node_feats.size(1), args.n_hid, num_node_types, n_classes, steps, dropout = args.dropout, use_norm = not args.no_norm, in_nl = args.in_nl).cuda()

    optimizer = torch.optim.Adam(
        list(model.parameters()),
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
        if(epoch%50==0):
            logging.info("Epoch {}; Train err {}; Val err {}; Val f1 {}".format(epoch + 1, train_loss, val_loss, f1_val))
        if best_val is None or f1_val > best_val:
            best_val = f1_val
            final = f1_test
            anchor = epoch + 1
            patience = 0
        else:
            patience += 1
            if(patience == 10):
                logging.info("Early stopping at Epoch {}".format(epoch + 1))
                break
    logging.info("Best val F1 {} at epoch {}; Test F1 {}".format(best_val, anchor, final))
    
    return best_val, final


def eliminate_and_reproduce(old_gene_pools, population_performance):
    old_gene_pools_cp = copy.deepcopy(old_gene_pools)

    # Eliminate
    ranking = np.argsort(population_performance) # hit_rate_50 (HR20), refer to metrics()
    preserved_index = ranking[int(ELIMINATE_RATE*POPULATION_SIZE):]
    new_gene_pools = []
    for i in preserved_index:
        new_gene_pools = new_gene_pools + old_gene_pools_cp[GENE_NUM*i:(GENE_NUM*i+GENE_NUM)]

    # Reproduce
    while len(new_gene_pools) < len(old_gene_pools_cp):
        try:
            pre_p = np.exp(population_performance[preserved_index])
        except: pdb.set_trace()
        pre_p = pre_p / pre_p.sum()
        index = np.random.choice(preserved_index,size = 1,p=pre_p)[0]
        new_gene_pools = new_gene_pools + old_gene_pools_cp[GENE_NUM*index:(GENE_NUM*index+GENE_NUM)]
    
    return new_gene_pools

def structure2logic(meta, dataset_string):
    nx_meta = convert_to_networkx_graph({'nodes': meta[0], 'edges': meta[1]})
    logic = graph2logic(nx_meta, dataset_string, initialization=True)
    return logic

if __name__ == "__main__":

    dataset_string = args.dataset.lower() 
    print('dataset_string: ', dataset_string)
    assert dataset_string in ['acm', 'imdb']
    task_string = 'node_classification' #'recommendation'
    dataset_task_string = dataset_string + '_' + task_string #dataset_task_string = 'yelp_recommendation'

    # Load data
    node_feats, node_types, adjs_pt, train_idx, train_target, valid_idx, valid_target, test_idx, test_target, n_classes, num_node_types = load_data(datadir='data_node_classification')

    # cuda settings
    torch.cuda.set_device(args.gpu)
    
    # Initialization: Hyperparams
    gene_pools_history_dict = dict() 

    ##################################################################################################################
    # Initialization

    # LLM intialization
    trial='_component_v3' 
    dialogs_save_path = logdir 
    llm4meta = LLM4Meta(client=client, dataset=dataset_string, downstream_task=task_string, dialogs_save_path=dialogs_save_path)
    
    # Gene pool initialization: GENE_POOL_SIZE = GENE_NUM * POPULATION_SIZE
    # acm init
    PAP = [['P', 'P', 'A'], [[0,0,1],[0,0,1],[1,0,1]]]
    PAPSP_1 = [['P', 'P', 'A', 'P', 'S'], [[0,0,1,0,0],[0,0,0,0,1],[1,0,0,1,0],[0,0,1,0,1],[0,1,0,1,0]]]
    PAPSP_2 = [['P', 'P', 'A', 'P', 'S'], [[0,0,1,0,1],[0,0,0,0,1],[1,0,0,1,0],[0,0,1,0,1],[1,1,0,1,0]]]
    PAPAP_1 = [['P', 'P', 'A', 'P', 'A'], [[0,0,1,0,0],[0,0,0,0,1],[1,0,0,1,0],[0,0,1,0,1],[0,1,0,1,0]]]
    PAPAP_2 = [['P', 'P', 'A', 'P', 'A'], [[0,0,1,0,1],[0,0,0,0,1],[1,0,0,1,0],[0,0,1,0,1],[1,1,0,1,0]]]
    
    # imdb init
    MDM = [['M', 'M', 'D'], [[0,0,1],[0,0,1],[1,0,1]]]
    MDMAM_1 = [['M', 'M', 'D', 'M', 'A'], [[0,0,1,0,0],[0,0,0,0,1],[1,0,0,1,0],[0,0,1,0,1],[0,1,0,1,0]]]
    MDMDM_1 = [['M', 'M', 'D', 'M', 'D'], [[0,0,1,0,0],[0,0,0,0,1],[1,0,0,1,0],[0,0,1,0,1],[0,1,0,1,0]]]
    DMDM_1 = [['D','M','D','M'], [[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]]]
    AMDM_1 = [['A','M','D','M'], [[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]]]

    initialization_dict = {
        'acm': [PAP, PAPSP_1, PAPSP_2, PAPAP_1, PAPAP_2], 
        'imdb': [MDM, MDMAM_1, MDMDM_1, DMDM_1, AMDM_1]  
    }

    gene_pools = initialization_dict[dataset_string]

    ##################################################################################################################
    # Optimization

    if(args.test_known_metas):
        print('***********Testing pre-identified meta-structures.*****************')

    performance_dict = {'best_val': {}, 'correspond_test': {}}
    gene_pools_performance_dict = {'best_val': {}, 'correspond_test': {}}
    full_logic_perf_dict = dict()
    best_performance = 0
    best_test = None
    best_gene_pools = gene_pools.copy()
    best_generation = None
    best_individual = None
    seen_structures = set()
    num_seen_structures = len(seen_structures)
    start0 = time.time()
    for gen in range(args.num_generations):
        start = time.time()
        gene_pools_history_dict[gen] = gene_pools
        population_performance = []
        population_final_test = []
        new_gene_pools = []
        if(len(seen_structures)>num_seen_structures):
            num_seen_structures = len(seen_structures)
            print(f'Explored new meta-structure (s). Total exploration: {num_seen_structures}')
        
        for i in range(len(gene_pools)):
            gene = gene_pools[i]
            logic = structure2logic(gene, dataset_string)

            # Evaluation
            if(logic in performance_dict['best_val']):
                print('Evaluated.')
                performance = performance_dict['best_val'][logic]
                final_test = performance_dict['correspond_test'][logic]
            else:
                performance, final_test = evaluate([gene])
                if(performance>best_performance):
                    best_performance = performance
                    best_test = final_test
                    best_gene = gene.copy()
                    best_logic = structure2logic(best_gene, dataset_string)
                    best_generation = gen
                    best_individual = i
                performance_dict['best_val'][logic] = performance
                performance_dict['correspond_test'][logic] = final_test
            print(f'Generation {gen}, Individual {i}, current performance: {performance}, best performance: {best_performance}, correspond. test: {best_test}, at Gen {best_generation} Individual {best_individual} \n')
            population_performance.append(performance)
            population_final_test.append(final_test)
            gene_pools_performance_dict['best_val'][gen] = population_performance.copy()
            gene_pools_performance_dict['correspond_test'][gen] = population_final_test.copy()

            full_logic_perf_dict[logic] = np.round(performance,6)

            most_recent_performances_prompt = ''
            best_performance_prompt = ''

        # Save performance
        if(not args.test_known_metas):
            with open(os.path.join(logdir, 'performance_dict.pkl'), 'wb') as f:
                pickle.dump(performance_dict, f)
            with open(os.path.join(logdir, 'gene_pools_history_dict.pkl'), 'wb') as f:
                pickle.dump(gene_pools_history_dict, f)    
            with open(os.path.join(logdir, 'gene_pools_performance_dict.pkl'), 'wb') as f:
                pickle.dump(gene_pools_performance_dict, f)    

        # Eliminate and reproduce
        population_performance = np.array(population_performance)
        gene_pools = eliminate_and_reproduce(gene_pools, population_performance)

        # Improve meta-structures
        for gene in gene_pools:
            new_gene, seen_structures, _  = llm4meta.modify_metas(gen, [gene], seen_structures, most_recent_performances_prompt, best_performance_prompt, full_logic_perf_dict)
            new_gene_pools.append(new_gene)
        gene_pools = new_gene_pools.copy()
         
        print('LLM improvement completed. Used time: ', time.time()-start)
    
    print(f'All {args.num_generations} generated. Used time: ', time.time() - start0)
    print(f'best_gene_pools: {best_gene_pools}, best val: {best_performance}, corrspond.test: {best_test}, at Generation {best_generation} Individual {best_individual}')
    client.close()
    pdb.set_trace()
    
