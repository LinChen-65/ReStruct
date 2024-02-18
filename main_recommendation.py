import os
import sys
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import time
from datetime import datetime
import argparse
import logging
from model_recommendation import Model
import copy
from utils import *
import networkx as nx
import pdb
from llm_component_no_prob import LLM4Meta 

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
parser.add_argument('--neg_train_size', type=int, default=4)
parser.add_argument('--neg_val_test_size', type=int, default=100)
parser.add_argument('--loss_margin', type=float, default=0.3)
parser.add_argument('--dataset_seed', type=int, default=2)
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

##############################################################################################################

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


def load_data(datadir):
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
    pdb.set_trace()
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

def evaluate(gene_pools, dataset_string): 
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

def get_recent_performances(perf_dict, num_to_get):
    num_gens = len(perf_dict)
    if(num_gens>num_to_get):
        return {i : perf_dict[i] for i in range(num_gens-num_to_get, num_gens)}
    else:
        return perf_dict

def structure2logic(meta, dataset_string):
    nx_meta = convert_to_networkx_graph({'nodes': meta[0], 'edges': meta[1]})
    logic = graph2logic(nx_meta, dataset_string, initialization=True)
    return logic

if __name__ == "__main__":

    prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + "_h" + str(args.n_hid) + \
         "_drop" + str(args.dropout) + "_epoch" + str(args.epochs) + "_cuda" + str(args.gpu)

    logdir = os.path.join(f"log_recommendation/train_threshold_2_datasetseed_{args.dataset_seed}_changeinit", args.dataset)
        
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
    node_types, num_node_types, adjs_pt, pos_train, pos_val, pos_test, neg_train, neg_val, neg_test, in_dims, node_feats = load_data(datadir='data_recommendation')

    # cuda settings
    torch.cuda.set_device(args.gpu)
    
    # Initialization: Hyperparams
    gene_pools_history_dict = dict() 

    ##################################################################################################################
    # Initialization

    # LLM intialization
    trial='_component_v3l' 
    dialogs_save_path = logdir 
    llm4meta = LLM4Meta(client=client, dataset=dataset_string, downstream_task=task_string, dialogs_save_path=dialogs_save_path)
    
    # Gene pool initialization: GENE_POOL_SIZE = GENE_NUM * POPULATION_SIZE
    # yelp init
    UUB1 = [['U', 'B', 'U'], [[0,0,1],[0,0,1],[1,1,0]]]
    UBAB2 = [['U', 'B', 'B', 'A'], [[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]]]
    UBIB1 = [['U', 'B', 'B', 'I'], [[0,0,1,0],[0,0,0,1],[1,0,0,1],[0,1,1,0]]]
    UOUB1 = [['U', 'B', 'O', 'U'], [[0,0,1,0],[0,0,0,1],[1,0,0,1],[0,1,1,0]]]
    UUB_complex = [['U', 'B', 'U'], [[0,1,1],[1,0,1],[1,1,0]]]
    UUBUB_complex = [['U', 'B', 'U', 'B', 'U'], [[0,0,1,0,0],[0,0,0,0,1],[1,0,0,1,1],[0,0,1,0,1],[0,1,1,1,0]]]
    
    # douban_movie init
    UMAM1 = [['U', 'M', 'M', 'A'], [[0,0,1,0],[0,0,0,1],[1,0,0,1],[0,1,1,0]]]
    UMTM1 = [['U', 'M', 'M', 'T'], [[0,0,1,0],[0,0,0,1],[1,0,0,1],[0,1,1,0]]]
    UUMDM1 = [['U','M','U','M','D'], [[0,0,1,0,0],[0,0,0,0,1],[1,0,0,1,0],[0,0,1,0,1],[0,1,0,1,0]]]
    UGUM1 = [['U', 'M', 'G', 'U'], [[0,0,1,0],[0,0,0,1],[1,0,0,1],[0,1,1,0]]]
    UUM_complex = [['U', 'M', 'U'], [[0,0,1],[0,0,1],[1,1,0]]]
    
    # amazon init
    UIVI1 = [['U', 'I', 'I', 'V'], [[0,0,1,0],[0,0,0,1],[1,0,0,1],[0,1,1,0]]]
    UICI1 = [['U', 'I', 'I', 'C'], [[0,0,1,0],[0,0,0,1],[1,0,0,1],[0,1,1,0]]]
    UIBI2 = [['U', 'I', 'I', 'B'], [[0,1,1,0],[1,0,0,1],[1,0,0,1], [0,1,1,0]]]
    UIUI1 = [['U', 'I', 'I', 'U'], [[0,0,1,0],[0,0,0,1],[1,0,0,1],[0,1,1,0]]]
    UIBIV = [['U', 'I', 'I', 'B', 'V'], [[0,0,1,0,0],[0,0,0,1,1],[1,0,0,1,1],[0,1,1,0,0],[0,1,1,0,0]]]
    
    initialization_dict = {
        'yelp': [UUB_complex, UUB_complex, UUB_complex, UUB_complex, UUB_complex],
        'amazon': [UIVI1, UICI1, UIBI2, UIUI1, UIBIV], #[UIVI1, UICI1, UIBI1, UIUI1, UIBIV],
        'douban_movie': [UMAM1, UMTM1, UUMDM1, UGUM1, UUM_complex] #[UMAM1, UMTM1, UMDM1, UGUM1, UUMUM_complex]#UMUM1 #UUUM1
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

    
