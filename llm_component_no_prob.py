from utils import *
import numpy as np
import fire
import os
from itertools import combinations_with_replacement #
from collections import defaultdict
import networkx as nx


class LLM4Meta:

    def __init__(self, client, dataset, downstream_task, num_gene=5, dialogs_save_path='./data/'):
        self.client = client
        self.dialogs_save_path = dialogs_save_path
        self.max_num_candidates = 20 
        self.num_similar_retrieval = 3 
        self.num_warmup = 0
        self.max_num_nodes_per_candidate = 8
        self.num_gene = num_gene
        self.dataset = dataset
        self.downstream_task = downstream_task
        self.num_sampled_evaluated = 30 
        if(self.dataset=='yelp'):
            self.node_types = {'User': 'U', 'Local Business': 'B', 'Business Category': 'A', 'Business City': 'I', 'User Compliment': 'O'}
            self.edge_types = ['U-U', 'U-B', 'U-O', 'B-A', 'B-I']
            self.edge_types_with_reverse = ['U-U', 'U-B', 'B-U', 'U-O', 'O-U', 'B-A', 'A-B', 'B-I', 'I-B']
            self.et_str = ['U-U: Is Friend Of', 'U-B: Visited', 'B-U: Is Visited By', 'U-O: Provided', 'U-O: Is Provided By', 'B-A: Belongs To', 'A-B: Contains', 'B-I: Is Located In', 'I-B: Has']
            self.source_nt = 'U'
            self.target_nt = 'B'
            self.example_logic = '''"B (2, node index) (THAT Is Visited By U (1) THAT Is Friend Of U (0)) AND (THAT Is Visited By U (0))", which indicates that both the target user and his friend likes B.'''
            self.relationship = {
                'U-U': 'Is Friend Of',
                'U-B': 'Visits',
                'B-U': 'Is Visited By',
                'U-O': 'Provides Compliment To',
                'O-U': 'Receives Compliment From',
                'B-A': 'Belongs To',
                'A-B': 'Contains',
                'B-I': 'Is Located In',
                'I-B': 'Has'
            }
        elif(self.dataset=='amazon'):
            self.node_types = {'User': 'U', 'Item': 'I', 'View': 'V', 'Category': 'C', 'Brand': 'B'}
            self.edge_types = ['U-I', 'I-V', 'I-C', 'I-B']
            self.edge_types_with_reverse = ['U-I', 'I-U', 'I-V', 'V-I', 'I-C', 'C-I', 'I-B', 'B-I']
            self.et_str = ['U-I: Purchased', 'I-U: Is Purchased By', 'I-V: Is Viewed By', 'V-I: Views', 'I-C: Is Categorized Under', 'C-I: Encompasses', 'I-B: Is Branded As', 'B-I: Incoporates']
            self.source_nt = 'U'
            self.target_nt = 'I'
            self.example_logic = '''"I (3, node index) THAT Is Categorized Under C (2) THAT Encompasses I (1) THAT Is Purchased By U (0)", which indicates that the target user may purchase another item under the same category of his previous purchased item.'''
            self.relationship = {
                'U-I': 'Purchased', 
                'I-U': 'Is Purchased By', 
                'I-V': 'Is Viewed By', 
                'V-I': 'Views', 
                'I-C': 'Is Categorized Under', 
                'C-I': 'Encompasses', 
                'I-B': 'Is Branded As', 
                'B-I': 'Incoporates'
            }
        elif(self.dataset=='douban_movie'):
            self.node_types = {'User': 'U', 'Movie': 'M', 'Group': 'G', 'Actor': 'A', 'Director': 'D', 'Type': 'T'}
            self.edge_types = ['U-U', 'U-M', 'U-G', 'M-A', 'M-D', 'M-T']
            self.edge_types_with_reverse = ['U-U', 'U-M', 'M-U', 'U-G', 'G-U', 'M-A', 'A-M', 'M-D', 'D-M', 'M-T', 'T-M']
            self.et_str = ['U-U: Is Friend Of', 'U-M: Watches', 'M-U: Is Watched By', 'U-G: Belongs To', 'G-U: Incoporates', 'M-A: Features Actor', 'A-M: Stars In', 'M-D: Is Directed By', 'D-M: Directs', 'M-T: Belongs to Type', 'T-M: Encompasses']
            self.source_nt = 'U'
            self.target_nt = 'M'
            self.example_logic = '''"M (2, node index) (THAT Is Watched By U (1) THAT Is Friend Of U (0)) AND (THAT Is Visited By U (0))", which indicates that both the target user and his friend likes Movie M.'''
            self.relationship = {
                'U-U': 'Is Friend Of', 
                'U-M': 'Watches', 
                'M-U': 'Is Watched By', 
                'U-G': 'Belongs To', 
                'G-U': 'Incoporates', 
                'M-A': 'Features Actor', 
                'A-M': 'Stars In', 
                'M-D': 'Is Directed By', 
                'D-M': 'Directs', 
                'M-T': 'Belongs to Type', 
                'T-M': 'Encompasses'
            }
        elif(self.dataset=='acm'):
            self.node_types = {'Author': 'A', 'Paper': 'P', 'Subject': 'S'}
            self.edge_types = ['A-P', 'P-S']
            self.edge_types_with_reverse = ['A-P', 'P-A', 'P-S', 'S-P']
            self.et_str = ['A-P: Authors', 'P-A: Is Authored By', 'P-S: Studies', 'S-P: Is Studied In']
            self.source_nt = 'P'
            self.target_nt = 'P'
            self.example_logic = '''A (3, node index) THAT Authors P (2) ((THAT Studies S (1)) AND (THAT Is Authored By A (0)))'''
            self.relationship = {'A-P': 'Authors', 'P-A': 'Is Authored By', 'P-S': 'Studies', 'S-P': 'Is Studied In'}
        elif(self.dataset=='imdb'):
            self.node_types = {'Movie': 'M', 'Actor': 'A', 'Director': 'D', 'Conference': 'C', 'Type': 'T'}
            self.edge_types = ['M-D', 'M-A']
            self.edge_types_with_reverse = ['M-D', 'D-M', 'M-A', 'A-M']
            self.et_str = ['M-D: Is Directed By', 'D-M: Directs', 'M-A: Features Actor', 'A-M: Stars In']
            self.source_nt = 'M'
            self.target_nt = 'M'
            self.example_logic = '''M (2, node type) THAT Is Directed By D (1) THAT Directs M (0)'''
            self.relationship = {'M-D': 'Is Directed By', 'D-M': 'Directs', 'M-A': 'Features Actor', 'A-M': 'Stars In'}
        else:
            print(f'{self.dataset}, not implemented yet.')
            pdb.set_trace()
        self.nt_str = [f'{st}' for t, st in self.node_types.items()]
        self.nt_semantic_str = [f'{st}: {t}' for t, st in self.node_types.items()]
        
        self.components = []
        for n_comp in [1, 2, 3]:
            this_comp = []
            for nodes in list(combinations_with_replacement(self.nt_str, n_comp)):
                this_comp.extend(generate_graphs_with_paths(nodes, self.edge_types))
            self.components.extend(this_comp)
        #test = graph2logic(self.components[11], initialization=True)
        self.logic2component = {graph2logic(comp, self.dataset, initialization=False, source_node=0, target_node=len(comp.nodes())-1): comp for comp in self.components}
        self.logic2meta = {}
        self.logic2meta.update(self.logic2component)
        self.combination_perf = defaultdict(list)
        self.total_cost = 0

        self.rollback_target = self.generate_rollback_target()
    
    def generate_rollback_target(self):
        G = nx.Graph()
        if(self.dataset=='yelp'):
            G.add_node(0, type='U0', label='source')
            G.add_node(1, type='B1', label='target')
            G.add_edge(0, 1)
        elif(self.dataset=='amazon'):
            G.add_node(0, type='U0', label='source')
            G.add_node(1, type='I1', label='target')
            G.add_edge(0, 1)
        elif(self.dataset=='douban_movie'):
            G.add_node(0, type='U0', label='source')
            G.add_node(1, type='M1', label='target')
            G.add_edge(0, 1)
        elif(self.dataset=='acm'):
            G.add_node(0, type='P0', label='source')
            G.add_node(1, type='P1', label='target')
            G.add_node(2, type='A2', label='')
            G.add_edge(0, 2)
            G.add_edge(1, 2)
        elif(self.dataset=='imdb'):
            G.add_node(0, type='M0', label='source')
            G.add_node(1, type='M1', label='target')
            G.add_node(2, type='D2', label='')
            G.add_edge(0, 2)
            G.add_edge(1, 2)
        else:
            print(f'Dataset {self.dataset} not implemented yet.')
            pdb.set_trace()
        return G

    @property
    def tools(self):        
        return None
    
    @property
    def meta_tools(self):
        return [self.meta_insert, self.meta_graft]

    
    def get_one_step_candidates(self, logic):
        candidate_list = []
        # Get one-step candidates from insertion (with 1-node components)
        candidates_from_insert = []
        locations, _ = self.get_insert_location_candidates(logic)
        for loc in locations:
            meta_candidates_without_prefix, _ = self.get_insert_meta_candidates(logic, loc, step=1)
            candidates_from_insert.extend(meta_candidates_without_prefix)
        candidate_list.extend(candidates_from_insert)
        # Get one-step candidates from grafting (with 2- or 3-node components)
        candidates_from_graft = []
        locations, _ = self.get_graft_location_candidates(logic)
        for loc in locations:
            meta_candidates_without_prefix, _ = self.get_graft_meta_candidates(logic, loc)
            candidates_from_graft.extend(meta_candidates_without_prefix)
        candidate_list.extend(candidates_from_graft)
        # Get one-step candidates from deleting 1 node
        candidates_from_delete, _ = self.get_delete_meta_candidates(logic, step=1)
        candidate_list.extend(candidates_from_delete)
        # Get the logic itself (can choose not to change)
        candidate_list = list(set(candidate_list))
        return candidate_list

    def get_two_step_candidates(self, logic):
        candidate_list = []
        # Get two-step candidates from insertion (with 2-node components)
        candidates_from_insert = []
        locations, _ = self.get_insert_location_candidates(logic)
        for loc in locations:
            meta_candidates_without_prefix, _ = self.get_insert_meta_candidates(logic, loc, step=2)
            candidates_from_insert.extend(meta_candidates_without_prefix)
        candidate_list.extend(candidates_from_insert)
        # Don't get grafting components, as grafting with 2/3-node components are both one-step
        # Get two-step candidates from deleting 2 nodes? 
        candidates_from_delete, _ = self.get_delete_meta_candidates(logic, step=2)
        candidate_list.extend(candidates_from_delete)
        candidate_list = list(set(candidate_list))
        return candidate_list

    def predict_candidate_performances(self, candidate_list, full_logic_perf_dict):
        # Sample from full_logic_perf_dict
        if(len(full_logic_perf_dict)>self.num_sampled_evaluated):
            sampled_logic_perf_dict = dict(random.sample(full_logic_perf_dict.items(), self.num_sampled_evaluated))
            sentence = f'''When making predictions, you can refer to the performances of {self.num_sampled_evaluated} randomly-sampled evaluated meta-structures (the larger the better): {sampled_logic_perf_dict}.'''
        else:
            sentence = f'''When making predictions, you can refer to the performances of all evaluated meta-structures (the larger the better): {full_logic_perf_dict}.'''
        dialogs_save_path = 'candidate-perf-pred.txt'
        dialogs = []
        system_prompt = f'''
                        You are an expert in heterogeneous information networks (HIN) and specialize in identifying meta-structures, which are crucial, maybe high-order, logical structures between various types of nodes. 
                        You have an HIN from {self.dataset} with node types {self.nt_str} and edge types {self.edge_types}.
                        The semantic meanings of each node type are {self.nt_semantic_str}.
                        The semantic meanings of each valid edge type are {self.et_str}.
                        The ultimate goal is to improve meta-structures from {self.source_nt} to {self.target_nt} for a {self.downstream_task} model based on this HIN.
                        Each meta-structure is represented by a logic flow, which is nested attributed clauses with "THAT".
                        Specifically, starting from a {self.source_nt} node, during the process of traversing the entire graph, the connection between every two neighboring nodes is converted to a logical relationship contained in the edge types. 
                        Multiple logical relationships are combined using THAT; when a node is connected to multiple other nodes, logic flows are combined using AND.
                        For example, the logic flow of one meta-structure may be {self.example_logic}.
                        Your current goal is to predict the performance of each candidate meta-structure in the given list.
                        {sentence}
                        You can also refer to these intuitions: (1) Meta-structures with more similar logics tend to have more similar performances. (2) It is generally more possible to find a good meta-structure when considering diverse types of nodes.
                        ''' 
                        
        dialogs.append(encap_msg(system_prompt, 'system'))
        candidate_perf_pred_dict = dict()
        first_seen_candidates = []
        first_seen_candidate_indices = []
        for i in range(len(candidate_list)):
            candidate = candidate_list[i]
            if candidate in full_logic_perf_dict:
                candidate_perf_pred_dict[candidate] = [full_logic_perf_dict[candidate], 1.0] # performance, confidence
            else:
                first_seen_candidates.append(f'Candidate {i}: {candidate}')
                first_seen_candidate_indices.append(i)
        prompt = f'''
                    For the current list of candidate meta-structures: {first_seen_candidates}, please predict the performance of each of them based on your understanding of its semantic meaning and the given evaluation results.
                    Please think carefully, do step-by-step reasoning, and return a float number between 0 and 1 as your predicted performance and another float number between 0 and 1 as your confidence in this prediction (both are the larger the better).
                    For each candidate, start your response with 'Candidate i, Performance: ' and append your predicted performance, followed by ', Confidence: ' and your confidence. 
                    PLEASE STRICTLY FOLLOW THE REQUIRED FORMAT!!! DO NOT include anything else, especially long explanations.
                    Example: Candidate 1, Performance: 0.65, Confidence: 0.8
                    ''' 
        dialogs.append(encap_msg(prompt))
        msg, this_cost = get_gpt_completion(dialogs, client=self.client, tools=None) 
        self.total_cost += this_cost
        dialogs.append(encap_msg(msg.content, 'assistant'))
        try:
            batch_perf_conf_dict = extract_batch_predicted_performance_confidence(msg.content)
        except:
            print(f'Cannot extract batch predicted performance and confidence. msg.content: {msg.content}')
            self.save_dialogs(dialogs, dialogs_save_path, mode='a')
            pdb.set_trace()
        for i in range(len(candidate_list)):
            if(i in first_seen_candidate_indices):
                candidate = candidate_list[i]
                try:
                    candidate_perf_pred_dict[candidate] = batch_perf_conf_dict[i]
                except:
                    print('Something wrong in batch_perf_conf_dict[i].')
                    pdb.set_trace()

        self.save_dialogs(dialogs, dialogs_save_path, mode='a')   
        return candidate_perf_pred_dict

    def modify_metas(self, gen, metas, seen_structures, most_recent_performances, best_performance, full_logic_perf_dict, get_all_candidates=False):
        dialogs_save_path = 'one-step-improve.txt'
        dialogs = []
        system_prompt = f'''
                        You are an expert in heterogeneous information networks (HIN) and specialize in identifying meta-structures, which are crucial, maybe high-order, logical structures between various types of nodes. 
                        You have an HIN from {self.dataset} with node types {self.nt_str} and edge types {self.edge_types}.
                        The semantic meanings of each node type are {self.nt_semantic_str}.
                        The semantic meanings of each valid edge type are {self.et_str}.
                        Your goal is to improve meta-structures from {self.source_nt} to {self.target_nt} for a {self.downstream_task} model based on this HIN.
                        Each meta-structure is represented by a logic flow, which is nested attributed clauses with "THAT".
                        Specifically, starting from a {self.source_nt} node, during the process of traversing the entire graph, the connection between every two neighboring nodes is converted to a logical relationship contained in the edge types. 
                        Multiple logical relationships are combined using THAT; when a node is connected to multiple other nodes, logic flows are combined using AND.
                        For example, the logic flow of one meta-structure may be {self.example_logic}.
                        You can also refer to these intuitions: (1) Meta-structures with more similar logics tend to have more similar performances. (2) It is generally more possible to find a good meta-structure when considering diverse types of nodes.
                        '''
        dialogs.append(encap_msg(system_prompt, 'system'))

        logics = []
        for meta in metas:
            nx_meta = convert_to_networkx_graph({'nodes': meta[0], 'edges': meta[1]})
            logic = graph2logic(nx_meta, self.dataset, initialization=True)
            self.logic2meta[logic] = nx_meta
            logics.append(logic)
        
        new_metas, new_nxs = [], []
        for idx, (meta, logic) in enumerate(zip(metas, logics)):
            # Try getting one-step candidates
            candidate_list = self.get_one_step_candidates(logic) # Lists of logics
            if(get_all_candidates): 
                candidate_logic_list = candidate_list.copy()
                candidate_matrix_list = []
                for candidate_logic in candidate_logic_list:
                    new_meta = self.logic2meta[candidate_logic] 
                    meta_dict = nx2dict(new_meta)
                    candidate_matrix_list.append(meta_dict)
                return candidate_logic_list, candidate_matrix_list
            original_candidate_num = len(candidate_list)
            # Remove stuctures with too many nodes
            candidate_list = [candidate for candidate in candidate_list if len(self.logic2meta[candidate])<=self.max_num_nodes_per_candidate]
            print(f'{len(candidate_list)} out of {original_candidate_num} candidates contain no more than {self.max_num_nodes_per_candidate} nodes.')
            original_candidate_num = len(candidate_list)
            # Remove seen structures
            seen_candidate_list = [candidate for candidate in candidate_list if candidate in seen_structures]
            candidate_list = [candidate for candidate in candidate_list if candidate not in seen_structures]
            print(f'{len(candidate_list)} out of {original_candidate_num} candidates are new.')

            if(len(candidate_list)==0): 
                print('No candidate from one-step change. Try getting two-step candidates.')
                # Try getting two-step candidates
                candidate_list = self.get_two_step_candidates(logic)
                original_candidate_num = len(candidate_list)
                # Remove stuctures with too many nodes
                candidate_list = [candidate for candidate in candidate_list if len(self.logic2meta[candidate])<=self.max_num_nodes_per_candidate]
                print(f'{len(candidate_list)} out of {original_candidate_num} candidates contain no more than {self.max_num_nodes_per_candidate} nodes.')
                original_candidate_num = len(candidate_list)
                # Remove seen structures
                seen_candidate_list = [candidate for candidate in candidate_list if candidate in seen_structures]
                candidate_list = [candidate for candidate in candidate_list if candidate not in seen_structures]
                print(f'{len(candidate_list)} out of {original_candidate_num} candidates are new.')
                
            if(len(candidate_list)==0): 
                print('No candidate from one-step or two-step change. Random sample from seen candidates.')    
                if(len(seen_candidate_list)>3):
                    candidate_list = random.sample(seen_candidate_list, 3)
                else:
                    print('No candidate from one-step or two-step change or seen candidates. Random sample from other seen structures.')
                    #candidate_list = seen_candidate_list.copy()
                    try:
                        candidate_list = random.sample(seen_structures, 3)
                    except:
                        candidate_list = seen_structures.copy()
            
            if(len(candidate_list)==0):
                print('Still no candidate at all...')
                pdb.set_trace()

            # Random sampling from candidate list to prevent number explosion
            if(len(candidate_list)>self.max_num_candidates):
                candidate_list = random.sample(candidate_list, self.max_num_candidates)

            if(gen>self.num_warmup):
                # Prompt LLM to predict the performances of each candidate
                candidate_perf_pred_dict = self.predict_candidate_performances(candidate_list, full_logic_perf_dict)
                candidate_dict = dict()
                for i in range(len(candidate_list)):
                    candidate = candidate_list[i]
                    try:
                        if candidate in full_logic_perf_dict:
                            candidate_dict[f'Candidate {i}'] = f'''logic: {candidate}, predicted performance: {full_logic_perf_dict[candidate]}, confidence: 1'''
                        elif candidate in candidate_perf_pred_dict:
                            candidate_dict[f'Candidate {i}'] = f'''logic: {candidate}, predicted performance: {candidate_perf_pred_dict[candidate][0]}, confidence: {candidate_perf_pred_dict[candidate][1]}'''
                        else: pass 
                    except:
                        pdb.set_trace()
                candidate_description = f'''the logic flows of its one-step adjacent meta-structures are as follows, each accompanied by a predicted performance and the confidence of this prediction (format: Candidate i: logic: xxx, [predicted performance: xxx, confidence: xxx]): {candidate_dict}'''
            
            else:
                candidate_dict = dict()
                for i in range(len(candidate_list)):
                    candidate_dict[f'Candidate {i}'] = candidate_list[i]
                candidate_description = f'''the logic flows of its one-step adjacent meta-structures are as follows: {candidate_dict}'''
            
            # Prompt LLM to select one candidate
            prompt = f'''
                    For the current meta-structure with this logic flow: {logic}, {candidate_description}.
                    Please think carefully and choose one candidate from this list, which you think makes most sense based on your understanding of their semantic meanings and the recorded performances of their similar counterparts, if any. 
                    If you feel that the current meta-structure is already too complicated to be meaningful, choose from the ones that are shorter than the current one. 
                    If you believe that the current meta-structure is already optimal, choose the last candidate.
                    Please return only ONE candidate after step-by-step reasoning. 
                    Start your response with 'Candidate ' and append your chosen index. DO NOT include anything else.
                    Example: Candidate 1.
                ''' 
            dialogs.append(encap_msg(prompt))
            msg, this_cost = get_gpt_completion(dialogs, client=self.client, tools=None) 
            self.total_cost += this_cost
            dialogs.append(encap_msg(msg.content, 'assistant'))

            try:
                new_logic = candidate_list[extract_candidate_index(msg.content)]
            except:
                print(f'Cannot extract location index. msg.content: {msg.content}')
                self.save_dialogs(dialogs, dialogs_save_path, mode='a')
                pdb.set_trace()
            
            new_meta = self.logic2meta[new_logic] # Should contain one node with label 'source' and one node with label 'target'

            meta_dict = nx2dict(new_meta)
            new_nxs.append(new_meta)
            new_metas.append(list(meta_dict.values())+meta[2:])
            print(f'Meta {idx} modified. Num of nodes change from {len(nx_meta)} to {len(new_meta)}. Total cost: {self.total_cost}')
            print('After modification, logic: ', new_logic)

        self.save_dialogs(dialogs, dialogs_save_path, mode='a')   
        seen_structures.add(new_logic)
        return new_metas[0], seen_structures, new_nxs # only 1 meta-structure per individual

    def meta_insert(self, logic, meta_idx, comb_perf):
        dialogs = []
        system_prompt = f'''
                        You are an expert in heterogeneous information networks (HIN) and specialize in identifying meta-structures, which are crucial, maybe high-order, logical structures between various types of nodes. 
                        You have an HIN from {self.dataset} with node types {self.nt_str} and edge types {self.edge_types}.
                        The semantic meanings of each node type are {self.nt_semantic_str}.
                        The semantic meanings of each valid edge type are {self.et_str}.
                        Your goal is to improve meta-structures from {self.source_nt} to {self.target_nt} for a {self.downstream_task} for a recommendation model based on this HIN.
                        Each meta-structure is represented by a logic flow, which is nested attributed clauses with "THAT".
                        Specifically, starting from a {self.source_nt} node, during the process of traversing the entire graph, the connection between every two neighboring nodes is converted to a logical relationship contained in the edge types. 
                        Multiple logical relationships are combined using THAT; when a node is connected to multiple other nodes, logic flows are combined using AND.
                        For example, the logic flow of one meta-structure may be {self.example_logic}.
                        '''
        dialogs.append(encap_msg(system_prompt, 'system'))
        
        location_candidates_index, location_candidates_type = self.get_insert_location_candidates(logic)
        perf_prompt = '\n'.join([f'Evolution: {perf}' for perf in comb_perf])
        prompt = f'''
                    Here is the logic flow of the current meta-structure: {logic}. 
                    This meta-structure belongs to a combination (several meta-structures), which evolves iteratively and is applied for a recommendation model.
                    The following is the performances of this combination over multiple evolutions, where each is represented as a tuple: (logic flow of {self.num_gene} meta-structures, recommendation performance),
                    {perf_prompt}
                    Also for the meta-structure above, all locations that a new component can possibly be inserted include: {location_candidates_type}. 
                    Please think carefully and choose one location for component insertion, based on your understanding of the semantic meanings and the meta-structures' historical performance on the {self.downstream_task} task, if any. 
                    Please return only ONE index of the chosen location. Avoid ANY redundancy in your response. 
                    Start your response with 'Location ' and append your chosen index.
                    Example: Location 0.
                '''
        dialogs.append(encap_msg(prompt))
        msg, this_cost = get_gpt_completion(dialogs, client=self.client, tools=None) 
        self.total_cost += this_cost
        
        dialogs.append(encap_msg(msg.content, 'assistant'))
        
        if len(location_candidates_index)==0:
            new_logic = logic
            prompt = f'''
                        Here is the logic flow of the current meta-structure: {logic}. 
                        For this meta-structure, no locations.
                        Thus we skip this round.
                    '''
            dialogs.append(encap_msg(prompt))
            self.save_dialogs(dialogs, 'insert.txt', mode='a')
            return new_logic
        elif(len(location_candidates_index)==1):
            chosen_location = location_candidates_index[0]
        else:
            try:
                chosen_location = location_candidates_index[extract_location_index(msg.content)]
            except:
                print(f'Cannot extract location index. msg.content: {msg.content}')
                self.save_dialogs(dialogs, 'insert.txt', mode='a')
                pdb.set_trace()
        meta_candidates_without_prefix, meta_candidates_with_prefix = self.get_insert_meta_candidates(logic, chosen_location)
        if(len(meta_candidates_without_prefix)==0): # If there is no candidate, skip deletion
            new_logic = logic
            prompt = f'''
                        Here is the logic flow of the current meta-structure: {logic}. 
                        For this meta-structure, no new meta-strutures can possibly be generated.
                        Thus we skip this round.
                    '''
            dialogs.append(encap_msg(prompt))
        elif(len(meta_candidates_without_prefix)==1):
            new_logic = meta_candidates_without_prefix[0]
        else:
            try:
                comb_idx = meta_idx//self.num_gene
                for cand_idx, cand_logic in enumerate(meta_candidates_without_prefix):
                    comb_logic = self.combination_perf[f'Combination {comb_idx+1}'][-1][0]
                    comb_logic[meta_idx%self.num_gene] = cand_logic
                    comb_meta = [self.logic2meta[logic] for logic in comb_logic]
                    sim_perf = []
                    for c in self.combination_perf:
                        for history_comb in self.combination_perf[c]:
                            history_meta = [self.logic2meta[logic] for logic in history_comb[0]]
                            sim = calculate_graph_similarity(comb_meta, history_meta)
                            sim_perf.append((np.around(sim, decimals=3), np.around(history_comb[1], decimals=3)))
                    sim_perf = sorted(sim_perf, key=lambda x: x[0])[-5:]
                    meta_candidates_with_prefix[cand_idx] += f'. If this new meta-structure replaces the old one in the combination, the new combination will be the most similar to the following combinations that have been evaluated, where each tuple denotes (the similarity between this combination and new combination, the performance of this combination): {sim_perf}.'
            except:
                print('Something wrong in meta_insert(). Please check.')
                pdb.set_trace()
            prompt = f'''
                        For the current meta-structure {logic} and the chose location {chosen_location}, all new meta-strutures that can possibly be generated from insertion include: {meta_candidates_with_prefix}.
                        Please analyze the semantic meanings of them and choose the most meaningful one. Think carefully before making your choice.
                        Please return only ONE index of the chosen meta-structure. Avoid ANY redundancy in your response. 
                        Start your response with 'Meta ' and append your chosen index.
                        Example: Meta 0.
                    '''
            dialogs.append(encap_msg(prompt))
            msg, this_cost = get_gpt_completion(dialogs, client=self.client, tools=None) 
            self.total_cost += this_cost
            dialogs.append(encap_msg(msg.content, 'assistant'))

            try: 
                chosen_meta = meta_candidates_without_prefix[extract_meta_index(msg.content)]
            except:
                print(f'Cannot extract meta index. msg.content: {msg.content}')
                self.save_dialogs(dialogs, 'insert.txt', mode='a')
                pdb.set_trace()
            
            new_logic = chosen_meta
        dialogs.append(encap_msg('New logic: '+new_logic, 'user'))
        self.save_dialogs(dialogs, 'insert.txt', mode='a')
        return new_logic
    

    def get_insert_location_candidates(self, logic):
        locations_index = []
        locations_type = []
        meta = self.logic2meta[logic]
        # Return types of every pair of adjacent nodes
        count = 0
        for edge in meta.edges():
            node1, node2 = edge
            type1 = meta.nodes[node1]['type'] #[0] # Remove the numbering: from 'U0' to 'U'
            type2 = meta.nodes[node2]['type'] #[0]
            locations_index.append((node1, node2))
            locations_type.append(f'Location {count}: {type1}, {type2}')
            count += 1
        return locations_index, locations_type


    def get_insert_meta_candidates(self, logic, location, step=1):
        candidates_without_prefix = []
        candidates_with_prefix = []
        count = 0
        for component_idx in range(len(self.components)):
            if(len(self.components[component_idx])<=step): # make step a changeable param    
                try: 
                    new_logic = self.insert_component(location, component_idx, logic)
                    if(('None' not in new_logic) & (new_logic!=logic)):
                        candidates_without_prefix.append(new_logic)
                        candidates_with_prefix.append(f'Meta {count}: ' + new_logic)
                        count += 1
                except:
                    pass 
        return candidates_without_prefix, candidates_with_prefix


    def insert_component(self, location, component_idx, logic): # Insert a component (light meta-path) into the meta-structure in the specified location to create a new meta-structure. This tool creates new edges, thus the new edges must be among valid edge types
        component = self.components[component_idx]
        meta = self.logic2meta[logic]
        merged_graph = meta.copy()
        max_node = len(merged_graph.nodes)-1
        start_node = location[0]
        for node in component.nodes:
            node_type = component.nodes[node]['type'][0]
            merged_graph.add_node(max_node+node+1, type=f'{node_type}{int(max_node+node+1)}')
            merged_graph.add_edge(start_node, max_node+node+1)
            start_node = max_node+node+1
        if len(location) > 1:
            merged_graph.add_edge(location[1], max_node+node+1)
        # Delete the original edge connecting location[0] and location[1]
        merged_graph.remove_edge(location[0], location[1])
        new_logic = graph2logic(merged_graph, self.dataset, initialization=False)
        
        # Clean graph
        record = {}
        for node, attr in merged_graph.nodes(data=True):
            if(attr.get('label') == 'source'):
                record['source'] = node
            elif(attr.get('label') == 'target'):
                record['target'] = node
        if(('source' not in record) or ('target' not in record)):
            print('Something wrong. Cant find source or target.')
            pdb.set_trace()
        merged_graph_cleaned = clean_nx_graph(merged_graph, record['source'], record['target'], visualize=False)
        for node,attr in merged_graph_cleaned.nodes(data=True):
            if(node==record['source']): 
                attr['label'] = 'source'
            if(node==record['target']): 
                attr['label'] = 'target'

        new_logic = graph2logic(merged_graph_cleaned, self.dataset, initialization=False)
        if('None' not in new_logic):
            self.logic2meta.update({new_logic: merged_graph_cleaned})
        return new_logic
    

    def meta_graft(self, logic, meta_idx, comb_perf):
        dialogs = []
        system_prompt = f'''
                        You are an expert in heterogeneous information networks (HIN) and specialize in identifying meta-structures, which are crucial, maybe high-order, logical structures between various types of nodes. 
                        You have an HIN from {self.dataset} with node types {self.nt_str} and edge types {self.edge_types}.
                        The semantic meanings of each node type are {self.nt_semantic_str}.
                        The semantic meanings of each valid edge type are {self.et_str}.
                        Your goal is to improve meta-structures from {self.source_nt} to {self.target_nt} for a {self.downstream_task} for a recommendation model based on this HIN.
                        Each meta-structure is represented by a logic flow, which is nested attributed clauses with "THAT".
                        Specifically, starting from a {self.source_nt} node, during the process of traversing the entire graph, the connection between every two neighboring nodes is converted to a logical relationship contained in the edge types. 
                        Multiple logical relationships are combined using THAT; when a node is connected to multiple other nodes, logic flows are combined using AND.
                        For example, the logic flow of one meta-structure may be {self.example_logic}.
                        '''
        dialogs.append(encap_msg(system_prompt, 'system'))
        
        location_candidates_index, location_candidates_type = self.get_graft_location_candidates(logic)
        perf_prompt = '\n'.join([f'Evolution: {perf}' for perf in comb_perf])
        prompt = f'''
                    Here is the logic flow of the current meta-structure: {logic}. 
                    This meta-structure belongs to a combination (several meta-structures), which evolves iteratively and is applied for a recommendation model.
                    The following is the performances of this combination over multiple evolutions, where each is represented as a tuple: (logic flow of {self.num_gene} meta-structures, recommendation performance),
                    {perf_prompt}
                    Also for this meta-structure, all locations that a new component can possibly be grafted onto include: {location_candidates_type}. 
                    Please think carefully and choose one location for component grafting, based on your understanding of the semantic meanings and the meta-structures' historical performance on the {self.downstream_task} task, if any. 
                    Please return only ONE index of the chosen location. Avoid ANY redundancy in your response. 
                    Example: 0.
                '''
        dialogs.append(encap_msg(prompt))
        msg, this_cost = get_gpt_completion(dialogs, client=self.client, tools=None) 
        self.total_cost += this_cost
        
        dialogs.append(encap_msg(msg.content, 'assistant'))

        if len(location_candidates_index)==0:
            new_logic = logic
            prompt = f'''
                        Here is the logic flow of the current meta-structure: {logic}. 
                        For this meta-structure, no locations.
                        Thus we skip this round.
                    '''
            dialogs.append(encap_msg(prompt))
            self.save_dialogs(dialogs, 'graft.txt', mode='a')
            return new_logic
        elif(len(location_candidates_index)==1):
            chosen_location = location_candidates_index[0]
        else:
            try:
                chosen_location = location_candidates_index[extract_location_index(msg.content)]
            except:
                print(f'Cannot extract location index. msg.content: {msg.content}')
                self.save_dialogs(dialogs, 'graft.txt', mode='a')
                pdb.set_trace()
        meta_candidates_without_prefix, meta_candidates_with_prefix = self.get_graft_meta_candidates(logic, chosen_location)
        if(len(meta_candidates_without_prefix)==0): # If there is no candidate, skip deletion
            new_logic = logic
            prompt = f'''
                        Here is the logic flow of the current meta-structure: {logic}. 
                        For this meta-structure, no new meta-strutures can possibly be generated.
                        Thus we skip this round.
                    '''
            dialogs.append(encap_msg(prompt))
        elif(len(meta_candidates_without_prefix)==1):
            new_logic = meta_candidates_without_prefix[0]
        else:
            try:
                comb_idx = meta_idx//self.num_gene
                for cand_idx, cand_logic in enumerate(meta_candidates_without_prefix):
                    comb_logic = self.combination_perf[f'Combination {comb_idx+1}'][-1][0]
                    comb_logic[meta_idx%self.num_gene] = cand_logic
                    comb_meta = [self.logic2meta[logic] for logic in comb_logic]
                    sim_perf = []
                    for c in self.combination_perf:
                        for history_comb in self.combination_perf[c]:
                            history_meta = [self.logic2meta[logic] for logic in history_comb[0]]
                            sim = calculate_graph_similarity(comb_meta, history_meta)
                            sim_perf.append((np.around(sim, decimals=3), np.around(history_comb[1], decimals=3)))
                    sim_perf = sorted(sim_perf, key=lambda x: x[0])[-5:]
                    meta_candidates_with_prefix[cand_idx] += f'. If this new meta-structure replaces the old one in the combination, the new combination will be the most similar to the following combinations that have been evaluated, where each tuple denotes (the similarity between this combination and new combination, the performance of this combination): {sim_perf}.'
            except:
                print('Something wrong in meta_graft(). Please check.')
                pdb.set_trace()
            prompt = f'''
                        For the current meta-structure {logic} and the chose location {chosen_location}, all new meta-strutures that can possibly be generated from grafting include: {meta_candidates_with_prefix}.
                        Please analyze the semantic meanings of them and choose the most meaningful one. Think carefully before making your choice.
                        Please return only ONE index of the chosen meta-structure. Avoid ANY redundancy in your response. 
                        Start your response with 'Meta ' and append your chosen index.
                        Example: Meta 0.
                    '''
            dialogs.append(encap_msg(prompt))
            msg, this_cost = get_gpt_completion(dialogs, client=self.client, tools=None) 
            self.total_cost += this_cost
            dialogs.append(encap_msg(msg.content, 'assistant'))

            try: 
                chosen_meta = meta_candidates_without_prefix[extract_meta_index(msg.content)]
            except:
                print(f'Cannot extract meta index. msg.content: {msg.content}')
                self.save_dialogs(dialogs, 'graft.txt', mode='a')
                pdb.set_trace()
            
            new_logic = chosen_meta
        #draw_multiple_graphs(list(self.logic2meta.values()), save_filename=os.path.join(self.dialogs_save_path, 'graph.jpg'))
        dialogs.append(encap_msg('New logic: '+new_logic, 'user'))
        self.save_dialogs(dialogs, 'graft.txt', mode='a')
        return new_logic


    def get_graft_location_candidates(self, logic):
        locations_index = []
        locations_type = []
        meta = self.logic2meta[logic]
        # Return types of every pair of nodes (whether adjacent or not)
        count = 0
        for node1 in meta.nodes():
            for node2 in meta.nodes():
                if(node1 < node2):
                    type1 = meta.nodes[node1]['type'] 
                    type2 = meta.nodes[node2]['type'] 
                    # Check whether there exists any meta-structure candidate
                    if(self.check_graft_meta_candidate_existance(logic, (node1, node2))):
                        locations_index.append((node1, node2))
                        locations_type.append(f'Location {count}: {type1}, {type2}')
                        count += 1
        return locations_index, locations_type

    
    def check_graft_meta_candidate_existance(self, logic, location): # Adapted from get_graft_meta_candidates()
        for component_idx in range(len(self.components)):
            component = self.components[component_idx]
            if(len(component.nodes)==1): # Don't need to consider components of length 1, because this will be degraded to insertion rather than grafting.
                continue
            try: 
                new_logic = self.graft_component(location, component_idx, logic)
                if(('None' not in new_logic) & (new_logic!=logic)):
                    return True
            except:
                pass 
        return False


    def get_graft_meta_candidates(self, logic, location):
        candidates_without_prefix = []
        candidates_with_prefix = []
        count = 0
        for component_idx in range(len(self.components)):
            component = self.components[component_idx]
            if(len(component.nodes)==1): # Don't need to consider components of length 1, because this will be degraded to insertion rather than grafting.
                continue
            try: 
                new_logic = self.graft_component(location, component_idx, logic)
                if(('None' not in new_logic) & (new_logic!=logic)):
                    candidates_without_prefix.append(new_logic)
                    candidates_with_prefix.append(f'Meta {count}: ' + new_logic)
                    count += 1
            except:
                pass 
        return candidates_without_prefix, candidates_with_prefix


    def graft_component(self, location, component_idx, logic): # Graft a component onto the meta-structure in the specified location to create a new meta-structure. This tool merge the starting (and ending) nodes of the component with meta-structure nodes, thus the node types must be the same correspondingly.
        component = self.components[component_idx]
        meta = self.logic2meta[logic]
        merged_graph = meta.copy()
        
        if(len(component.nodes)) == 1:
            new_logic = 'None'
        elif((component.nodes[0]['type'][0]!=meta.nodes[location[0]]['type'][0]) | (component.nodes[len(component.nodes)-1]['type'][0]!=meta.nodes[location[1]]['type'][0])):
            new_logic = 'None'
        
        else:
            if len(component.nodes) == 2:
                merged_graph.add_edge(location[0], location[1])
            else:
                max_node = len(merged_graph.nodes)-1
                start_node = location[0]
                for node in list(component.nodes)[1:len(component.nodes)-(len(location)-1)]:
                    node_type = component.nodes[node]['type'][0]
                    merged_graph.add_node(max_node+node, type=f'{node_type}{int(max_node+node)}')
                    merged_graph.add_edge(start_node, max_node+node)
                    start_node = max_node+node
                if len(location) > 1:
                    merged_graph.add_edge(location[1], max_node+node)
            
            # Clean graph
            record = {}
            for node, attr in merged_graph.nodes(data=True):
                if(attr.get('label') == 'source'):
                    record['source'] = node
                elif(attr.get('label') == 'target'):
                    record['target'] = node
            if(('source' not in record) or ('target' not in record)):
                print('Something wrong. Cant find source or target.')
                pdb.set_trace()
            merged_graph_cleaned = clean_nx_graph(merged_graph, record['source'], record['target'], visualize=False)
            for node,attr in merged_graph_cleaned.nodes(data=True):
                if(node==record['source']): 
                    attr['label'] = 'source'
                if(node==record['target']): 
                    attr['label'] = 'target'

            new_logic = graph2logic(merged_graph_cleaned, self.dataset, initialization=False)
            if('None' not in new_logic):
                self.logic2meta.update({new_logic: merged_graph_cleaned})
        
        return new_logic    


    def meta_delete(self, logic, meta_idx, comb_perf):
        dialogs = []
        system_prompt = f'''
                        You are an expert in heterogeneous information networks (HIN) and specialize in identifying meta-structures, which are crucial, maybe high-order, logical structures between various types of nodes. 
                        You have an HIN from {self.dataset} with node types {self.nt_str} and edge types {self.edge_types}.
                        The semantic meanings of each node type are {self.nt_semantic_str}.
                        The semantic meanings of each valid edge type are {self.et_str}.
                        Your goal is to improve meta-structures from {self.source_nt} to {self.target_nt} for a {self.downstream_task} for a recommendation model based on this HIN.
                        Each meta-structure is represented by a logic flow, which is nested attributed clauses with "THAT".
                        Specifically, starting from a {self.source_nt} node, during the process of traversing the entire graph, the connection between every two neighboring nodes is converted to a logical relationship contained in the edge types. 
                        Multiple logical relationships are combined using THAT; when a node is connected to multiple other nodes, logic flows are combined using AND.
                        For example, the logic flow of one meta-structure may be {self.example_logic}.
                        '''
        dialogs.append(encap_msg(system_prompt, 'system'))
        
        meta_candidates_without_prefix, meta_candidates_with_prefix = self.get_delete_meta_candidates(logic)     
        
        if(len(meta_candidates_without_prefix)==0): # If there is no candidate, skip deletion
            new_logic = logic
            prompt = f'''
                        Here is the logic flow of the current meta-structure: {logic}. 
                        For this meta-structure, no new meta-strutures can possibly be generated from deleting one node.
                        Thus we skip this round of deletion.
                    '''
            dialogs.append(encap_msg(prompt))
        elif(len(meta_candidates_without_prefix)==1):
            new_logic = meta_candidates_without_prefix[0]
        else:
            try:
                comb_idx = meta_idx//self.num_gene
                for cand_idx, cand_logic in enumerate(meta_candidates_without_prefix):
                    comb_logic = self.combination_perf[f'Combination {comb_idx+1}'][-1][0]
                    comb_logic[meta_idx%self.num_gene] = cand_logic
                    comb_meta = [self.logic2meta[logic] for logic in comb_logic]
                    sim_perf = []
                    for c in self.combination_perf:
                        for history_comb in self.combination_perf[c]:
                            history_meta = [self.logic2meta[logic] for logic in history_comb[0]]
                            sim = calculate_graph_similarity(comb_meta, history_meta)
                            sim_perf.append((np.around(sim, decimals=3), np.around(history_comb[1], decimals=3)))
                    sim_perf = sorted(sim_perf, key=lambda x: x[0])[-5:]
                    meta_candidates_with_prefix[cand_idx] += f'. If this new meta-structure replaces the old one in the combination, the new combination will be the most similar to the following combinations that have been evaluated, where each tuple denotes (the similarity between this combination and new combination, the performance of this combination): {sim_perf}.'
            except:
                print('Something wrong in meta_delete(). Please check.')
                pdb.set_trace()
            perf_prompt = '\n'.join([f'Evolution: {perf}' for perf in comb_perf])
            prompt = f'''
                        Here is the logic flow of the current meta-structure: {logic}. 
                        This meta-structure belongs to a combination (several meta-structures), which evolves iteratively and is applied for a recommendation model.
                        The following is the performances of this combination over multiple evolutions, where each is represented as a tuple: (logic flow of {self.num_gene} meta-structures, recommendation performance),
                        {perf_prompt}
                        Also for this meta-structure, all new meta-strutures that can possibly be generated from deleting one node include: {meta_candidates_with_prefix}.
                        Please analyze the semantic meanings of them and choose the most meaningful one. Think carefully before making your choice.
                        Please return only ONE index of the chosen meta-structure. Avoid ANY redundancy in your response. 
                        Start your response with 'Meta ' and append your chosen index.
                        Example: Meta 0.
                    '''
            dialogs.append(encap_msg(prompt))
            msg, this_cost = get_gpt_completion(dialogs, client=self.client, tools=None) 
            self.total_cost += this_cost
            dialogs.append(encap_msg(msg.content, 'assistant'))

            try: 
                chosen_meta = meta_candidates_without_prefix[extract_meta_index(msg.content)]
            except:
                print(f'Cannot extract meta index. msg.content: {msg.content}')
                self.save_dialogs(dialogs, 'delete.txt', mode='a')
                pdb.set_trace()
            
            new_logic = chosen_meta
        
        dialogs.append(encap_msg('New logic: '+new_logic, 'user'))
        self.save_dialogs(dialogs, 'delete.txt', mode='a')
        return new_logic


    def get_delete_meta_candidates(self, logic, step=1):
        meta = self.logic2meta[logic]
        candidates_without_prefix = []
        candidates_with_prefix = []
        count = 0

        G = meta.copy()
        G_before_deletion = G.copy()
        node_dict_before_deletion = {node:attr for node,attr in G.nodes(data=True)}

        record = {}
        for node, attr in G.nodes(data=True):
            if(attr.get('label') == 'source'):
                record['source'] = node
            elif(attr.get('label') == 'target'):
                record['target'] = node

        if(step==1):
            for node, attr in meta.nodes(data=True):
                if((attr.get('label') == 'source') or (attr.get('label') == 'target')):
                    continue
                G = meta.copy()
                G_before_deletion = G.copy()
                node_dict_before_deletion = {node:attr for node,attr in G.nodes(data=True)}
                try:neighbors = list(G.neighbors(node))
                except:pdb.set_trace()
                # Remove the specified node from the graph
                G.remove_node(node)
                # Get node index mapping before & after deletion
                num_reduced = len(G_before_deletion.nodes) - len(G.nodes)
                # Construct a dict to map neighbors to their new indices after deletion
                index_mapping = {this_node: this_node-num_reduced if this_node>node else this_node for this_node in G.nodes()}
                # Connect neighbors
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        try:
                            if(node_dict_before_deletion[neighbors[i]]['type'][0] + '-' + node_dict_before_deletion[neighbors[j]]['type'][0] in self.edge_types_with_reverse):    
                                G.add_edge(neighbors[i], neighbors[j])
                        except:
                            print('Something wrong in get_delete_meta_candidates(). Please check.')
                            pdb.set_trace()

                # Check whether G is still connected after deletion & whether source and target are still connected by some path
                success = nx.has_path(G, record['source'], record['target'])

                if(success):        
                    if(not nx.is_connected(G)):
                        # Find the connected component containing the specified nodes
                        connected_components = list(nx.connected_components(G))                    
                        target_component = None
                        for component in connected_components:
                            if record['source'] in component and record['target'] in component:
                                target_component = component
                                break
                        G = target_component
                    G = nx.relabel_nodes(G, index_mapping) # Relabel nodes using the mapping
                    for node, attr in G.nodes(data=True):
                        attr['type'] = attr['type'][0] + str(node)
                else: # Roll back to UB
                    G = self.rollback_target.copy() #G = self.components[6]
                # Remove duplicate paths and redundant branches
                record = {}
                for node, attr in G.nodes(data=True):
                    if(attr.get('label') == 'source'):
                        record['source'] = node
                    elif(attr.get('label') == 'target'):
                        record['target'] = node
                if(('source' not in record) or ('target' not in record)):
                    print('Something wrong. Cant find source or target.')
                    pdb.set_trace()
                G_cleaned = clean_nx_graph(G, record['source'], record['target'], visualize=False)
                for node,attr in G_cleaned.nodes(data=True):
                    if(node==record['source']): 
                        attr['label'] = 'source'
                    if(node==record['target']): 
                        attr['label'] = 'target'
                new_logic = graph2logic(G_cleaned, self.dataset, initialization=True)
                if('None' not in new_logic):
                    self.logic2meta.update({new_logic: G_cleaned})
                    candidates_without_prefix.append(new_logic)
                    candidates_with_prefix.append(f'Meta {count}: ' + new_logic)
                    count += 1
        
        else: # Deleting 2+ consecutive nodes
            for node1, attr1 in meta.nodes(data=True):
                if('label' in attr1):
                    if((attr1.get('label') == 'source') or (attr1.get('label') == 'target')):
                        continue
                node2_list = list(meta.neighbors(node1))
                for node2 in node2_list:
                    if('label' in meta.nodes[node2]):
                        if((meta.nodes[node2]['label'] == 'source') or (meta.nodes[node2]['label'] == 'target')):
                            continue
                    if(node2<=node1): # Avoid duplicates
                        continue
                    G = meta.copy()
                    G_before_deletion = G.copy()
                    node_dict_before_deletion = {node:attr for node,attr in G.nodes(data=True)}
                    neighbors = list(set(list(G.neighbors(node1)) + list(G.neighbors(node2))))
                    neighbors.remove(node1)
                    neighbors.remove(node2)
                    # Remove the specified nodes from the graph
                    G.remove_node(node1)
                    G.remove_node(node2)
                    # Get node index mapping before & after deletion
                    num_reduced = len(G_before_deletion.nodes) - len(G.nodes)
                    # Construct a dict to map neighbors to their new indices after deletion
                    index_mapping = {this_node: this_node-num_reduced if this_node>node2 else this_node for this_node in G.nodes()}
                    # Connect neighbors
                    for i in range(len(neighbors)):
                        for j in range(i + 1, len(neighbors)):
                            try:
                                if(node_dict_before_deletion[neighbors[i]]['type'][0] + '-' + node_dict_before_deletion[neighbors[j]]['type'][0] in self.edge_types_with_reverse):    
                                    G.add_edge(neighbors[i], neighbors[j])
                            except:
                                print('Something wrong in get_delete_meta_candidates(). Please check.')
                                pdb.set_trace()
                    # Check whether G is still connected after deletion & whether source and target are still connected by some path
                    success = nx.has_path(G, record['source'], record['target'])
                    if(success):    
                        if(not nx.is_connected(G)):
                            # Find the connected component containing the specified nodes
                            connected_components = list(nx.connected_components(G))                    
                            target_component = None
                            for component in connected_components:
                                if record['source'] in component and record['target'] in component:
                                    target_component = component
                                    break
                            G = target_component
                        G = nx.relabel_nodes(G, index_mapping) # Relabel nodes using the mapping
                        for node, attr in G.nodes(data=True):
                            attr['type'] = attr['type'][0] + str(node)
                    else: # Roll back to UB
                        G = self.rollback_target.copy() #G = self.components[6]
                    # Remove duplicate paths and redundant branches
                    record = {}
                    for node, attr in G.nodes(data=True):
                        if(attr.get('label') == 'source'):
                            record['source'] = node
                        elif(attr.get('label') == 'target'):
                            record['target'] = node
                    if(('source' not in record) or ('target' not in record)):
                        print('Something wrong. Cant find source or target.')
                        pdb.set_trace()
                    G_cleaned = clean_nx_graph(G, record['source'], record['target'], visualize=False)
                    for node,attr in G_cleaned.nodes(data=True):
                        if(node==record['source']): 
                            attr['label'] = 'source'
                        if(node==record['target']): 
                            attr['label'] = 'target'
                    new_logic = graph2logic(G_cleaned, self.dataset, initialization=True)
                    if('None' not in new_logic):
                        self.logic2meta.update({new_logic: G_cleaned})
                        candidates_without_prefix.append(new_logic)
                        candidates_with_prefix.append(f'Meta {count}: ' + new_logic)
                        count += 1
        
        return candidates_without_prefix, candidates_with_prefix


    def save_dialogs(self, dialogs, save_name='dialogs.txt', mode='w'):
        if not os.path.exists(self.dialogs_save_path):
            os.makedirs(self.dialogs_save_path)
            
        with open(os.path.join(self.dialogs_save_path, save_name), mode) as f:
            if mode == 'a':
                f.write(f'''************************************ START ************************************\n''')
            for dialog in dialogs:
                msg = f'''>>> {dialog['role']}: {dialog['content']}'''
                for k in dialog.keys():
                    if k not in ['role', 'content']:
                        msg = f'''
                            {msg}
                            {k}: {dialog[k]}
                            '''
                f.write(f'''{msg}\n''')
            if mode == 'a':
                f.write(f'''************************************ END ************************************\n''')
        
    
    def evaluate_similarity(self, logic, full_logic_perf_dict):
        dialogs_save_path = 'similarity-evaluation.txt'
        dialogs = []
        system_prompt = f'''
                        You are an expert in heterogeneous information networks (HIN) and specialize in identifying meta-structures, which are crucial, maybe high-order, logical structures between various types of nodes. 
                        You have an HIN from {self.dataset} with node types {self.nt_str} and edge types {self.edge_types}.
                        The semantic meanings of each node type are {self.nt_semantic_str}.
                        The semantic meanings of each valid edge type are {self.et_str}.
                        The ultimate goal is to improve meta-structures from {self.source_nt} to {self.target_nt} for a {self.downstream_task} for a recommendation model based on this HIN.
                        Each meta-structure is represented by a logic flow, which is nested attributed clauses with "THAT".
                        Specifically, starting from a {self.source_nt} node, during the process of traversing the entire graph, the connection between every two neighboring nodes is converted to a logical relationship contained in the edge types. 
                        Multiple logical relationships are combined using THAT; when a node is connected to multiple other nodes, logic flows are combined using AND.
                        For example, the logic flow of one meta-structure may be {self.example_logic}.
                        The logic flow of the candidate meta-structure is: {logic}.
                        In each round, you will receive the logic flow of another meta-structure that has been evaluated before, and your goal is to quantify the similarity between the candidate meta-structure and the given evaluated meta-structure, based on their logic flows.
                        Please think carefully and return a value ranging from 0 to 1 indicating the similarity, where 0 means absolute difference and 1 means identicality. 
                        Some intuitions you may refer to: Meta-structures with closer numbers of nodes, larger proportion of overlapping node types, larger proportion of overlapping edge types, or carrying closer semantic meanings are more likely to be more similar.
                        '''
        dialogs.append(encap_msg(system_prompt, 'system'))

        full_logic_sim_dict = dict()
        for evaluated_logic, evaluated_perf in full_logic_perf_dict.items():
            prompt = f'''
                        The logic flow of the evaluated meta-structure is : {evaluated_logic}.
                        Please return only ONE value after step-by-step reasoning. 
                        Start your response with 'Similarity ' and append your chosen index. After that, you can briefly and comprehensively summarize your reasons in one single sentence.
                        Example: Similarity 0.8.
                    '''
            dialogs.append(encap_msg(prompt))
            msg, this_cost = get_gpt_completion(dialogs, client=self.client, tools=None) 
            self.total_cost += this_cost
            dialogs.append(encap_msg(msg.content, 'assistant'))
            try:
                this_similarity = extract_similarity_value(msg.content)
            except:
                print(f'Cannot extract similarity_value. msg.content: {msg.content}')
                self.save_dialogs(dialogs, dialogs_save_path, mode='a')
                pdb.set_trace()
            if(this_similarity is None):
                print(f'Cannot extract similarity_value. msg.content: {msg.content}')
                self.save_dialogs(dialogs, dialogs_save_path, mode='a')
                pdb.set_trace()
            full_logic_sim_dict[evaluated_logic] = this_similarity
        self.save_dialogs(dialogs, dialogs_save_path, mode='a')
        print(f'Similarity evaluation completed. Total cost: {self.total_cost}')
        return full_logic_sim_dict
        

    def retrieve_most_similar_logics(self, logic, full_logic_perf_dict):
        dialogs_save_path = 'similarity-retrieval.txt'
        dialogs = []
        system_prompt = f'''
                        You are an expert in heterogeneous information networks (HIN) and specialize in identifying meta-structures, which are crucial, maybe high-order, logical structures between various types of nodes. 
                        You have an HIN from {self.dataset} with node types {self.nt_str} and edge types {self.edge_types}.
                        The semantic meanings of each node type are {self.nt_semantic_str}.
                        The semantic meanings of each valid edge type are {self.et_str}.
                        The ultimate goal is to improve meta-structures from {self.source_nt} to {self.target_nt} for a {self.downstream_task} for a recommendation model based on this HIN.
                        Each meta-structure is represented by a logic flow, which is nested attributed clauses with "THAT".
                        Specifically, starting from a {self.source_nt} node, during the process of traversing the entire graph, the connection between every two neighboring nodes is converted to a logical relationship contained in the edge types. 
                        Multiple logical relationships are combined using THAT; when a node is connected to multiple other nodes, logic flows are combined using AND.
                        For example, the logic flow of one meta-structure may be {self.example_logic}.
                        The logic flow of the candidate meta-structure is: {logic}.
                        Your target is to retrieve {self.num_similar_retrieval} logics from a given list that you think are most similar to this candidate. 
                        Please think carefully and return a list of {self.num_similar_retrieval} indices. 
                        Some intuitions you may refer to: Meta-structures with closer numbers of nodes, larger proportion of overlapping node types, larger proportion of overlapping edge types, or carrying closer semantic meanings are more likely to be more similar.
                        '''
        dialogs.append(encap_msg(system_prompt, 'system'))

        evaluated_logic_list = list(full_logic_perf_dict.keys())
        evaluated_logic_list_prompt = [f'Candidate {i}: ' + evaluated_logic_list[i] for i in range(len(evaluated_logic_list))]
        prompt = f'''
                        The given list of all logics is : {evaluated_logic_list_prompt}.
                        Please return exactly {self.num_similar_retrieval} indices after step-by-step reasoning. 
                        Indicate each logic you choose with a prefix "Candidate ", followed by a similarity metric between 0 and 1, where 0 means total difference and 1 means identicality.
                        Example: Candidate 1: 0.8, Candidate 2: 0.75, Candidate 6: 0.64, etc.
                    '''
        dialogs.append(encap_msg(prompt))
        msg, this_cost = get_gpt_completion(dialogs, gpt_model="gpt-3.5-turbo-1106", tools=None)  #msg, this_cost = get_gpt_completion(dialogs, tools=None) 
        self.total_cost += this_cost
        dialogs.append(encap_msg(msg.content, 'assistant'))
        try:
            top_index_sim_dict = extract_similarity_dict(msg.content)
            top_logic_sim_dict = {evaluated_logic_list[key] : value for key, value in top_index_sim_dict.items()}
        except:
            print(f'Cannot extract similarity_list. msg.content: {msg.content}')
            self.save_dialogs(dialogs, dialogs_save_path, mode='a')
            pdb.set_trace()

        top_dict = {}
        for key, value_sim in top_logic_sim_dict.items():
            value_perf = full_logic_perf_dict[key]
            top_dict[key] = [value_sim, value_perf]

        self.save_dialogs(dialogs, dialogs_save_path, mode='a')
        return top_dict


    def evaluate_similarity_algorithm(self, logic, full_logic_perf_dict):
        full_logic_sim_dict = dict()
        meta = self.logic2meta[logic]
        for evaluated_logic in full_logic_perf_dict:
            evaluated_meta = self.logic2meta[evaluated_logic]
            sim = graph_similarity(meta, evaluated_meta)
            full_logic_sim_dict[evaluated_logic] = sim
        return full_logic_sim_dict


    def generate_explanation(self, rank, main_structure, main_perf, neighbor_structure_list, neighbor_perf_list, grammar=True, diff=True):
        # both main_structure and neighbor_structure input are represented in matrices
        num_neighbors = len(neighbor_perf_list)
        
        prompt_name = ''
        if(grammar):
            prompt_name += '_Grammar'
        else:
            prompt_name += '_Matrix'
        if(diff):
            prompt_name += '_Differentiate'
        else:
            prompt_name += '_Nondifferentiate'
        
        dialogs_save_path = f'explanation_test_{rank}.txt' #f'explanation_new{prompt_name}_top_{rank}_{num_neighbors}neighbors.txt'
        dialogs = []
            
        if(grammar):
            example_prompt = ''' 'U (0) THAT Is Friend Of U (1) THAT Visits B (1)' performs well, because it captures the similar interests shared by friends, which is a strong indicator for successful recommendation.'''
            main_structure_prompt = f'The given meta-structure that achieves optimal performance is: {matrix2logic(main_structure, self.dataset)}, whose performance is {main_perf}. '
        
        else:
            example_prompt = '''[[U,U,B],[[0,0,1],[0,0,1],[1,1,0]]] performs well, because it captures the similar interests shared by friends, which is a strong indicator for successful recommendation.'''
            main_structure_prompt = f'The given meta-structure that achieves optimal performance is: {main_structure}, whose performance is {main_perf}. '

        system_prompt = f'''
                        You are an expert in heterogeneous information networks (HIN) and specialize in identifying meta-structures, which are crucial, maybe high-order, logical structures between various types of nodes. 
                        You have an HIN from {self.dataset} with node types {self.nt_str} and edge types {self.edge_types}.
                        The semantic meanings of each node type are {self.nt_semantic_str}.
                        The semantic meanings of each valid edge type are {self.et_str}.
                        The ultimate goal is to improve meta-structures from {self.source_nt} to {self.target_nt} for a {self.downstream_task} model based on this HIN.
                        Your current goal is to provide explanations to humans on why a given meta-structure performs well.
                        Example: {example_prompt}
                        ''' 
        dialogs.append(encap_msg(system_prompt, 'system'))
        
        # Construct prompt        
        if(grammar):
            prompt = f'''
                    A meta-structure is represented by a logic flow, which is nested attributed clauses with "THAT".
                    Specifically, starting from a {self.source_nt} node, during the process of traversing the entire graph, the connection between every two neighboring nodes is converted to a logical relationship contained in the edge types. 
                    Multiple logical relationships are combined using THAT; when a node is connected to multiple other nodes, logic flows are combined using AND.
                    For example, the logic flow of one meta-structure may be {self.example_logic}.
                    {main_structure_prompt}
                    '''
        else:
            prompt = f'''
                        A meta-structure is represented in this format:
                        {{'nodes': a ordered list of node types, 'edges': an adjacency matrix of node connections, where the entry can be 0 (no connection) or 1 (has connection)}}.
                        {main_structure_prompt}
                     '''

        if(not diff):
            prompt += '''
                        Please provide an explanation of why the given meta-structure performs well in a clear and concise manner following the provided example. 
                        Do not include any break downs of the sub-structures in your final output.
                        Think step by step, but only output your final summary.
                      ''' 
            dialogs.append(encap_msg(prompt))
            msg, this_cost = get_gpt_completion(dialogs, client=self.client, tools=None) 
            self.total_cost += this_cost
            dialogs.append(encap_msg(msg.content, 'assistant'))
            self.save_dialogs(dialogs, dialogs_save_path, mode='a')
            explanation = msg.content
            return explanation

        else:
            # Step 1
            prompt += f'''
                        To understand why the given meta-structure performs well, let's do the following 2-step analysis.
                        Step 1: Neighbor structure analysis.
                        Here are {num_neighbors} one-step neighbors of the given meta-structure: {[matrix2logic(neigh, self.dataset) for neigh in neighbor_structure_list]}.
                        Please conduct a thorough analysis of both the given meta-structure and all its one-step neighbors, by breaking down each of them into meaningful sub-structures and identify the sub-structure functions.
                      '''
            dialogs.append(encap_msg(prompt))
            msg, this_cost = get_gpt_completion(dialogs, client=self.client, tools=None) 
            self.total_cost += this_cost
            dialogs.append(encap_msg(msg.content, 'assistant'))

            # Step 2
            prompt = f'''
                        Step 2: Neighbor performance analysis.
                        Here are the performances measured by AUC of each neighbor provided above, with the same order: {neighbor_perf_list}.
                        Jointly considering these performances and your structural analysis from the previous step, please identify what good sub-structures the given meta-structure have, and what bad sub-structures are absent from it.
                        Please reply following this format: 
                            1. Includes: X THAT Y THAT xxx: This sub-structure xxx
                            2. Includes: X THAT Z THAT Y THAT xxx: This sub-structure xxx
                            3. Not includes: X THAT M THAT xxx: This sub-structure fails to xxx
                            4. ....
                            5. ....
                      ''' 
            dialogs.append(encap_msg(prompt))
            msg, this_cost = get_gpt_completion(dialogs, client=self.client, tools=None) 
            self.total_cost += this_cost
            dialogs.append(encap_msg(msg.content, 'assistant'))

            self.save_dialogs(dialogs, dialogs_save_path, mode='a')
            explanation = msg.content

        print('Cost: ', self.total_cost)
        return explanation


def main(trial='_component_v3'):
    dialogs_save_path = f'./data/gpt4{trial}'
    llm4meta = LLM4Meta(downstream_task='recommendation', dialogs_save_path=dialogs_save_path)
    UBUB = [['U', 'B', 'U', 'B'], [[0., -1., 0., 1.], [0., 0., 1., -1.], [0., 0., 0., 1.], [0., 0., 0., 0.]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 1]
    UBU = [['U', 'B', 'U'], [[0., -1., 1.], [0., 0., 1.], [0., 0., 0.]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 1]
    UBAB = [['U', 'B', 'A', 'B'], [[0., -1., 0., 1.], [0., 0., 1., -1.], [0., 0., 0., 1.], [0., 0., 0., 0.]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 1]
    UBIB = [['U', 'B', 'I', 'B'], [[0., -1., 0., 1.], [0., 0., 1., -1.], [0., 0., 0., 1.], [0., 0., 0., 0.]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 1]
    UB = [['U', 'B'], [[0, 1], [0, 0]]]

    gene_pools = [UB]

    for i in range(10):
        perfs = np.around(np.random.rand(2), decimals=5)
        gene_pools, new_nxs = llm4meta.modify_metas(gene_pools, perfs)

    client.close()
    pdb.set_trace()
    
if __name__ == "__main__":
    fire.Fire(main)
                        
        