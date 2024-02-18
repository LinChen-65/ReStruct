import ast
import re
from openai import OpenAI
import openai
import requests
import httpx
import gzip
import pickle
import json
#import joblib
import scipy.sparse as sp
import pdb
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
import random
from copy import deepcopy
import time
import math


api_key= 'PUT-YOUR-API-KEY-HERE'
GPT_MODEL = "gpt-4-1106-preview"
client = OpenAI(api_key=api_key, http_client=httpx.Client(
        #proxies=proxies['https'],
    ),)

prompt_cost_1k, completion_cost_1k = 0.01, 0.03 #0.001, 0.002

def get_gpt_completion(dialogs, gpt_model=GPT_MODEL, client=client, tools=None, tool_choice=None, temperature=0.6, max_tokens=1000):
    max_retries = 5 #1

    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=gpt_model, 
                messages=dialogs,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens
            )
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            this_cost = prompt_tokens/1000*prompt_cost_1k + completion_tokens/1000*completion_cost_1k
            return response.choices[0].message, this_cost
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(8)
            else:
                print(f"An error of type {type(e).__name__} occurred: {e}")
                return "Error"
            

def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }
    json_data = {"model": model, "messages": messages}
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
            proxies=proxies
        )
        response = response.json()
        prompt_tokens = response["usage"]["prompt_tokens"]
        completion_tokens = response["usage"]["completion_tokens"]
        this_cost = prompt_tokens/1000*prompt_cost_1k + completion_tokens/1000*completion_cost_1k
        return response["choices"][0]["message"], this_cost
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

            
def encap_msg(msg, role='user', **kwargs):
    dialog = {'role': role, 'content': msg}
    dialog.update(kwargs)
    return dialog

def unique_nested_list(nested_list):
    
    def convert_to_tuple(nested):
        if isinstance(nested, list):
            return tuple(convert_to_tuple(item) for item in nested)
        else:
            return nested

    # 使用集合去重
    unique_items_set = set(convert_to_tuple(item) for item in nested_list)

    # 将集合转换回列表
    unique_items_list = [list(item) for item in unique_items_set]
    return unique_items_list

def create_adjacency_matrix(nodes, edges, E):
    # 创建节点到索引的映射
    node_index = {node: i for i, node in enumerate(nodes)}
    
    # 初始化邻接矩阵
    n = len(nodes)
    matrix = [[0] * n for _ in range(n)]
    
    # 填充有边的部分
    for edge in edges:
        a, b = edge.split('-')
        matrix[node_index[a]][node_index[b]] = 1
        if a != b:
            matrix[node_index[b]][node_index[a]] = 1  # 对于无向图，需要填充两个方向

    # 填充不在E中的边
    for i in range(n):
        for j in range(n):
            if f"{nodes[i]}-{nodes[j]}" not in E and f"{nodes[j]}-{nodes[i]}" not in E:
                matrix[i][j] = -1

    # 设置下三角为 -1
    for i in range(n):
        for j in range(i):
            matrix[i][j] = -1

    return matrix

def extract_between_strings(text, start_str, end_str):
    start_index = text.find(start_str)
    end_index = text.find(end_str, start_index + len(start_str))
    
    if start_index != -1 and end_index != -1:
        extracted_text = text[start_index + len(start_str):end_index]
        return extracted_text
    else:
        return None

def extract_dicts_from_string(s):
    """
    从字符串中提取字典。
    """
    # 正则表达式匹配简单字典格式
    dict_pattern = r'\{[^{}]*\}'
    matches = re.findall(dict_pattern, s)

    # 将匹配到的字符串转换为字典
    extracted_dicts = []
    for match in matches:
        try:
            # 使用 ast.literal_eval 安全地评估字符串
            dict_obj = ast.literal_eval(match)
            if isinstance(dict_obj, dict):  # 确保是字典
                extracted_dicts.append(dict_obj)
        except (SyntaxError, ValueError):
            # 忽略评估失败的字符串
            pass

    return extracted_dicts

def extract_lists_from_string(text):
    pattern = r'\[.*?\]'
    list_strings = re.findall(pattern, text)
    lists = [ast.literal_eval(lst) for lst in list_strings]
    return lists


def compare_graphs(graph1, graph2):
    # 首先将边列表中的边转换为无向边
    def convert_to_undirected_edges(edges):
        undirected_edges = set()
        for edge in edges:
            parts = edge.split('-')
            if len(parts) == 2:
                undirected_edges.add('-'.join(sorted(parts)))
        return undirected_edges

    # 比较节点数
    nodes1 = set(graph1['nodes'])
    nodes2 = set(graph2['nodes'])
    if nodes1 != nodes2:
        return False
    
    # 比较边数，将边转换为无向边
    edges1 = convert_to_undirected_edges(graph1['edges'])
    edges2 = convert_to_undirected_edges(graph2['edges'])
    if edges1 != edges2:
        return False

    # 如果节点和边都相同，则图相同
    return True

def deduplicate_graphs(graphs):
    unique_graphs = []
    for graph in graphs:
        # 检查当前图是否已存在于 unique_graphs 中
        if not any(compare_graphs(graph, existing_graph) for existing_graph in unique_graphs):
            unique_graphs.append(graph)
    return unique_graphs

def print_metas(metas):
    for meta in metas:
        print(f'''<<< Meta >>> {meta}\n''')

def dict2json(some_dict):
    return [json.dumps(item) for item in some_dict]

def json2dict(some_json):
    return [json.loads(item) for item in some_json]

def record_new_metas(already_proposed_metas, metas):
    already_proposed_metas = dict2json(already_proposed_metas)
    metas = dict2json(metas)
    result_set = set(already_proposed_metas) | set(metas)

    # Convert back to list of dicts
    result_list = list(result_set)
    result_list = json2dict(result_list)

    return result_list



def plot_graphs(graphs, n_cols, file_name='./data/graph.jpg'):
    color_map = {}
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'pink', 'grey', 'brown', 'cyan']

    def assign_colors(node_types):
        for node_type in node_types:
            if node_type not in color_map:
                color_map[node_type] = colors[len(color_map) % len(colors)]

    # Define the draw_graph_with_labels function
    def draw_graph_with_labels(graph_dict, ax, node_size=300, font_color='white'):
        # Create a new graph
        G = nx.Graph()

        # Assign colors to node types
        assign_colors(graph_dict['nodes'])

        # Add nodes to the graph with unique identifiers and labels being their types
        for i, node_type in enumerate(graph_dict['nodes']):
            G.add_node(i, label=node_type, color=color_map[node_type])

        # Add edges to the graph based on the adjacency matrix
        size = len(graph_dict['nodes'])
        for i in range(size):
            for j in range(size):
                if graph_dict['edges'][i][j] == 1:
                    G.add_edge(i, j)

        # Draw the graph with labels
        pos = nx.spring_layout(G, seed=42)  # positions for all nodes
        labels = {i: G.nodes[i]['label'] for i in G.nodes()}  # Create a label mapping from node identifier to type
        nx.draw(G, pos, 
                node_color=[G.nodes[i]['color'] for i in G.nodes()], 
                with_labels=True, 
                labels=labels, 
                ax=ax, 
                node_size=node_size, 
                font_color=font_color,
                font_weight='bold')

    # Plot the graphs in subplots with titles and a clear separation between them
    fig, axes = plt.subplots(len(graphs)//n_cols, n_cols, figsize=(5*n_cols, 5*len(graphs)//n_cols))

    # Draw each graph in a subplot with specified node size and font color
    node_size = 700  # Adjust the node size as needed
    font_color = 'white'  # Set font color to white
    for i, graph in enumerate(graphs):
        # If there's only one graph, axes will not be an array, so we wrap it in a list
        if len(graphs) == 1:
            ax = axes[0, 0]
        else:
            ax = axes[i//n_cols, i%n_cols]
        ax.set_title(f'Graph {i%n_cols+1}', size=16)  # Set title for each subplot
        ax.set_xlabel(f'Combination {i//n_cols+1}')
        draw_graph_with_labels(graph, ax, node_size=node_size, font_color=font_color)
        # Draw a rectangle around the subplot to clearly separate them
        rect = plt.Rectangle((0, 0), 1, 1, fill=False, color="k", lw=2, transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)

    # Adjust layout to prevent overlap
    plt.tight_layout(pad=4)
    plt.savefig(file_name, dpi=400, bbox_inches='tight')
    
def remove_nodes_by_index(graph_dict, indices_to_remove):
    """
    Removes nodes and their related edges from a graph represented as a dictionary by specifying node indices to remove.

    Parameters:
    - graph_dict (dict): A dictionary representing the graph with 'nodes' and 'edges' keys.
    - indices_to_remove (list): A list of node indices to remove.

    Returns:
    - dict: The modified graph dictionary after node removal.
    """
    nodes = graph_dict['nodes']
    edges = graph_dict['edges']

    # Remove nodes and their related edges by index
    for index in sorted(indices_to_remove, reverse=True):
        nodes.pop(index)
        edges.pop(index)
        for row in edges:
            row.pop(index)

    return {'nodes': nodes, 'edges': edges}



def is_graph_valid(graph, node_types, valid_edges):
    """
    Check if a graph is valid based on given node types and valid edges.

    Parameters:
    graph (dict): A graph represented as a dictionary with 'nodes' and 'edges'.
    node_types (list): A list of valid node types.
    valid_edges (list): A list of valid edge types in the format 'Type1-Type2'.

    Returns:
    bool: True if the graph is valid, False otherwise.
    """
    nodes = graph['nodes']
    edges = graph['edges']
    graph_cp = deepcopy(graph)
    
    nodes2remove = set()
    num_error = 0
    error = ''
    
    # Check if all nodes are of valid types
    wrong_nodes = []
    for idx, node in enumerate(nodes):
        if node not in node_types:
            num_error += 1
            wrong_nodes.append(node)
            nodes2remove.add(idx)
    if len(wrong_nodes) > 0:
        error = f'These nodes are not allowed: {wrong_nodes}.'

    # Check for self-loops and invalid edges
    wrong_edges = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            # Self-loop check
            if i == j and edges[i][j] == 1:
                num_error += 1
                wrong_edges.append(f'self-loop for node {i}, where the type is {nodes[i]}.')
                graph_cp['edges'][i][j] = -1
                graph_cp['edges'][j][i] = -1

            # Invalid edge check
            if edges[i][j] == 1 and f"{nodes[i]}-{nodes[j]}" not in valid_edges and f"{nodes[j]}-{nodes[i]}" not in valid_edges:
                num_error += 1
                wrong_edges.append(f'{nodes[i]}-{nodes[j]} and {nodes[j]}-{nodes[i]} are not allowed.')
                graph_cp['edges'][i][j] = -1
                graph_cp['edges'][j][i] = -1
    if len(wrong_edges) > 0:
        error = f'''
                {error}
                These edges are invalid: 
                {wrong_edges}'''
                
    # Check isolated nodes
    nx_graph = convert_to_networkx_graph(graph_cp)
    comp = max(nx.connected_components(nx_graph), key=len)
    if len(comp) < len(nx_graph):
        num_error += 1
        error = f'''
            {error}
            This meta-graph is not connected!'''
        disconnected_nodes2remove = set(list(range(len(nodes)))) - set(list(comp)) 
        nodes2remove.update(set(disconnected_nodes2remove))
    graph_cp = remove_nodes_by_index(graph_cp, nodes2remove)

    return bool(num_error==0), error, graph_cp




def convert_to_networkx_graph(graph):
    """
    Convert a custom graph format to a NetworkX graph.

    Parameters:
    graph (dict): A graph with 'nodes' and 'edges'.

    Returns:
    G (networkx.Graph): A NetworkX graph.
    """
    G = nx.Graph()
    for idx, node in enumerate(graph['nodes']):
        G.add_node(idx, type=f'{node}{idx}')
    for i, row in enumerate(graph['edges']):
        for j, edge in enumerate(row):
            if edge == 1:
                G.add_edge(i, j)
    return G

def dict2nx(graph):
    G = nx.Graph()
    for idx, node in enumerate(graph['nodes']):
        G.add_node(idx, type=node)
    for edge in graph['edges']:
        G.add_edge(graph['nodes'].index(edge.split('-')[0]), graph['nodes'].index(edge.split('-')[1]))
    return G

def are_graph_same(graph1, graph2):
    
    def node_match(node1_attrs, node2_attrs):
        return node1_attrs['type'] == node2_attrs['type']

    return nx.is_isomorphic(convert_to_networkx_graph(graph1), \
                        convert_to_networkx_graph(graph2), \
                            node_match=node_match)
    
def are_nx_graph_same(graph1, graph2):
    # Check if graphs are isomorphic
    if not nx.is_isomorphic(graph1, graph2):
        return False

    # Check if node orders are the same
    nodes_graph1 = list(graph1.nodes(data=True))
    nodes_graph2 = list(graph2.nodes(data=True))

    return nodes_graph1 == nodes_graph2
    

def remove_duplicate_graphs(graphs):
    """
    Removes duplicate graphs from a list of graphs.

    Parameters:
    - graphs (list): A list of graphs represented in any suitable format.

    Returns:
    - list: A list of unique graphs with duplicates removed.
    """
    unique_graphs = []
    duplicated_idx = []
    for idx, graph in enumerate(graphs):
        # Check if the current graph is unique by comparing with existing unique graphs
        is_unique = True
        for unique_graph in unique_graphs:
            if are_nx_graph_same(graph, unique_graph):
                #print('graph: ', graph.nodes(data=True))
                #print('unique_graph: ', unique_graph.nodes(data=True))
                is_unique = False
                duplicated_idx.append(idx)
                break
        
        # If the graph is unique, add it to the list of unique graphs
        if is_unique:
            unique_graphs.append(graph)
    
    return unique_graphs, duplicated_idx

def jaccard_graphs(group1, group2):
    from itertools import product
    
    num_intersection = 0
    for g1, g2 in product(group1, group2):
        num_intersection += int(are_graph_same(g1, g2))
    
    num_union = len(group1)
    for g2 in group2:
        for idx, g1 in enumerate(group1):
            if are_graph_same(g2, g1):
                break
            if idx == len(group1)-1:
                num_union += 1

    return num_intersection/num_union



def extract_outer_list(text):
    left_bracket_index = text.find('[')
    if left_bracket_index == -1:
        return None  # 没有找到列表

    # 开始查找匹配的右方括号
    bracket_count = 0
    for i in range(left_bracket_index, len(text)):
        if text[i] == '[':
            bracket_count += 1
        elif text[i] == ']':
            bracket_count -= 1
            if bracket_count == 0:
                return eval(text[left_bracket_index:i+1])

    return None  # 没有找到匹配的右方括号


# Reimplementing DFS function to correctly handle single paths and multiple branches.

def get_relationship(G, dataset_string, node1, node2):
    if(dataset_string=='yelp'):
        relationships = {
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
    elif(dataset_string=='amazon'):
        relationships = {
                'U-I': 'Purchased', 
                'I-U': 'Is Purchased By', 
                'I-V': 'Is Viewed By', 
                'V-I': 'Views', 
                'I-C': 'Is Categorized Under', 
                'C-I': 'Encompasses', 
                'I-B': 'Is Branded As', 
                'B-I': 'Incoporates'
        }
    elif(dataset_string=='douban_movie'):
        relationships = {
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
    elif(dataset_string=='acm'):
        relationships = {
            'A-P': 'Authors', 
            'P-A': 'Is Authored By', 
            'P-S': 'Studies', 
            'S-P': 'Is Studied In'
        }
    elif(dataset_string=='imdb'):
        relationships = {
            'M-D': 'Is Directed By', 
            'D-M': 'Directs', 
            'M-A': 'Features Actor', 
            'A-M': 'Stars In'
        }    
    else:
        print(f'{dataset_string}, not implemented.')
        pdb.set_trace()

    attr1 = G.nodes[node1]['type']
    attr2 = G.nodes[node2]['type']
    rel = relationships.get(f'{attr1[0]}-{attr2[0]}', None)
    return f'''{attr1[0]} ({attr1[1]}) THAT {rel} {attr2[0]} ({attr2[1]})'''


def graph2logic(G, dataset_string, initialization=False, source_node=None, target_node=None, visited=None, logic_flow=''):
    if len(G.nodes) == 1:
        attr = G.nodes[0]['type']
        return f'''{attr[0]} ({attr[1]})'''
    
    if(initialization): # Label source node and target node for a component (must be a path, not structure)
        G.nodes[0]['label'] = 'source'
        G.nodes[1]['label'] = 'target'
    if(source_node is None):
        source_node = next(node for node, attr in G.nodes(data=True) if attr.get('label') == 'source')
    if(target_node is None):
        target_node = next(node for node, attr in G.nodes(data=True) if attr.get('label') == 'target')
    
    if visited is None:
        visited = set()
    
    visited.add(source_node)
        
    # Get all the neighbors for the node
    neighbors = [neighbor for neighbor in G.neighbors(source_node) if neighbor not in visited]
    
    # Initialize paths for branching
    paths = []
    
    for idx, neighbor in enumerate(neighbors):
        
        visited.add(source_node)

        # Get the relationship based on the attributes of the nodes
        relationship = get_relationship(G, dataset_string, source_node, neighbor)

        # Append the relationship to the current logic flow
        new_logic_flow = relationship.split(')')[1][1:]+')' if logic_flow or idx > 0 else relationship
        
        if(neighbor==target_node):
            # If the neighbor is the target node, add the path and return
            paths.append(new_logic_flow)
            #print('new_logic_flow: ', new_logic_flow)
        else:
            visited.add(neighbor)
            # Recursively visit the neighbor
            sub_path = graph2logic(G, dataset_string, initialization=False, source_node=neighbor, target_node=target_node, visited=visited.copy(), logic_flow=new_logic_flow)
            paths.append(sub_path)
            #print('sub_path: ', sub_path)

    # If there are multiple paths, connect them with 'AND', else just return the path
    if len(paths) > 1:
        formatted_paths = [format_path(path) for path in paths] # Add proper parentheses
        if logic_flow:
            return f'{logic_flow} ' + ' AND '.join(formatted_paths)
        else:
            return ' AND '.join(formatted_paths)
    elif paths:
        if logic_flow:
            return f'{logic_flow} ' + paths[0]
        else:
            return paths[0]
    else:
        return logic_flow


def generate_graphs_with_paths(node_attributes, valid_edge_types):
    """
    Generate graphs with paths based on given node attributes and valid edge types.

    Args:
    node_attributes (list): A list of node attributes.
    valid_edge_types (list): A list of valid edge types.

    Returns:
    list: A list of generated graphs.
    """
    from itertools import permutations
    # Generate all permutations of node attributes
    node_permutations = list(permutations(node_attributes))
    
    # Create graphs with valid paths
    valid_graphs = []
    for node_permutation in node_permutations:
        G = nx.Graph()
        G.add_nodes_from([(i, {'type': node_type+str(i)}) for i, node_type in enumerate(node_permutation)])
        
        # Generate all possible edges
        for i, j in zip(range(len(node_permutation)-1), range(1, len(node_permutation))):
            edge_type = node_permutation[i] + '-' + node_permutation[j]
            if edge_type in valid_edge_types:
                G.add_edge(i, j)
            edge_type = node_permutation[j] + '-' + node_permutation[i]
            if edge_type in valid_edge_types:
                G.add_edge(j, i)
        
        if nx.is_connected(G):
            valid_graphs.append(G)
    
    return remove_duplicate_graphs(valid_graphs)[0]




def draw_multiple_graphs(graphs, layout_type='spring', save_filename=None, dpi=400, bbox_inches='tight'):
    """
    Draw multiple NetworkX graphs with separators between them.

    Args:
    graphs (list of nx.Graph): List of NetworkX graphs to be drawn.
    layout_type (str): Layout type for arranging graphs ('spring', 'circular', 'kamada_kawai', etc.).
    save_filename (str): File name to save the combined image (optional).
    dpi (int): Dots per inch for saving the image (optional).
    bbox_inches (str): Bounding box option for saving the image (optional).
    """
    num_graphs = len(graphs)
    rows = int(math.ceil(math.sqrt(num_graphs)))  # Number of rows in the grid
    cols = int(math.ceil(num_graphs / rows))  # Number of columns in the grid
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust the space between subplots
    
    for i, graph in enumerate(graphs):
        ax = axes[i // cols, i % cols]
        
        # Choose layout type for arranging nodes
        if layout_type == 'spring':
            pos = nx.spring_layout(graph)
        elif layout_type == 'circular':
            pos = nx.circular_layout(graph)
        elif layout_type == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph)  # Default to spring layout
        
        node_labels = nx.get_node_attributes(graph, 'type')
        # labels = nx.get_node_attributes(graph, 'label')
        colors = ['mediumseagreen' if attr.get('label') == 'source' else 'pink' if attr.get('label') == 'target' else 'lightblue' for node, attr in graph.nodes(data=True)]
        nx.draw(graph, pos, with_labels=True, labels=node_labels, node_color=colors, font_weight='bold', ax=ax)
        rect = plt.Rectangle((0, 0), 1, 1, fill=False, color="k", lw=2, transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)

    
    # Hide empty subplots
    for i in range(num_graphs, rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis('off')
    
    # Save the combined image if specified
    if save_filename:
        plt.savefig(save_filename, dpi=dpi, bbox_inches=bbox_inches)
    
    # plt.show()
    plt.close()



def extract_location_index(input_string):
    if input_string.isdigit():
        return int(input_string)
    elif(input_string.strip('.').isdigit()):
        return int(input_string.strip('.'))
    else:
        match = re.search(r'Location (\d+)', input_string)
        if match:
            return int(match.group(1))
        else:
            return None


def extract_meta_index(input_string):
    if input_string.isdigit():
        return int(input_string)
    elif(input_string.strip('.').isdigit()):
        return int(input_string.strip('.'))
    else:
        match = re.search(r'Meta (\d+)', input_string)
        if match:
            return int(match.group(1))
        else:
            return None


def extract_candidate_index(input_string):
    if input_string.isdigit():
        return int(input_string)
    elif(input_string.strip('.').isdigit()):
        return int(input_string.strip('.'))
    else:
        match = re.search(r'Candidate (\d+)', input_string)
        if match:
            return int(match.group(1))
        else:
            return None

def extract_similarity_dict(input_string):
    pattern = r'Candidate (\d+): (\d+\.\d+)'
    matches = re.findall(pattern, input_string)
    result_dict = {int(match[0]): float(match[1]) for match in matches}
    return result_dict

def extract_similarity_value(input_string):
    try:
        float_value = float(input_string)
        return float_value
    except:
        # If not a direct float, use regular expression to find a float in the string
        match = re.search(r'Similarity (\d+(\.\d+)?)', input_string) #re.search(r'Similarity (\d+\.\d+)', input_string)
        if match:
            return float(match.group(1))
        else:
            return None

def extract_predicted_performance_confidence(input_string):
    pattern = r'(?:Performance: |Confidence: )(\d+\.\d+)'
    perf_conf_list = [float(match) for match in re.findall(pattern, input_string)]
    return perf_conf_list

def extract_batch_predicted_performance_confidence(input_string):
    pattern = r"Candidate (\d+).*?Performance: (\d+\.\d+).*?Confidence: (\d+\.\d+)"
    matches = re.findall(pattern, input_string)
    result_dict = {}
    for match in matches:
        result_dict[int(match[0])] = [float(match[1]), float(match[2])] 
    return result_dict


def format_path(path): # Add proper parentheses
    if(path[:4])=='THAT':
        return '(' + path + ')'
    else: # Insert the left bracket before the first THAT
        index = path.find('THAT')
        if index != -1:
            result = path[:index] + '(' + path[index:] + ')'
            #result += ')'  # Add right bracket at the end
            return result
        else:
            # No 'THAT' found, return the original string
            return path
        
def nx2dict(G):  
    nodes = [G.nodes[node]['type'][0] for node in G.nodes]
    # 获取邻接矩阵并转换为列表格式
    adj_matrix = nx.adjacency_matrix(G).toarray().tolist()
    # 创建结果字典
    graph_dict = {
        'nodes': nodes,
        'edges': adj_matrix
    }
    return graph_dict

def swap_elements_in_groups(arrs, group_size):
    arr = arrs[0]
    if len(arr) % group_size != 0:
        raise ValueError("数组长度必须是组大小的整数倍")

    # 确定组的数量
    num_groups = len(arr) // group_size

    # 随机选择两个不同的组进行交换
    group1, group2 = random.sample(range(num_groups), 2, )

    # 在所选组内随机选择一个元素进行交换
    index1 = group1 * group_size + random.randint(0, group_size - 1)
    index2 = group2 * group_size + random.randint(0, group_size - 1)

    # 交换元素
    for arr in arrs:
        arr[index1], arr[index2] = arr[index2], arr[index1]

    return arrs



def calculate_graph_similarity(group1, group2):
    """
    Calculate the similarity between two NetworkX graphs based on approximate graph edit distance.
    """
    total_similarity = 0
    
    # Compare each graph in group1 with each graph in group2
    for g1 in group1:
        best_similarity = max(graph_similarity(g1, g2) for g2 in group2)
        total_similarity += best_similarity
    return total_similarity / len(group1) if group1 else None

def node_match(node1_attrs, node2_attrs):
        return node1_attrs['type'][0] == node2_attrs['type'][0]
    
def graph_similarity(G1, G2):
    ged = nx.graph_edit_distance(G1, G2, node_match)
    # Convert to a similarity measure (smaller distances mean more similar)
    max_possible_ged = max(len(G1.nodes) + len(G2.nodes), len(G1.edges) + len(G2.edges))
    similarity = 1 - (ged / max_possible_ged)
    return similarity


def gload(filename):
    file = gzip.GzipFile(filename, 'rb')
    res = pickle.load(file)
    #res = joblib.load(file)
    file.close()
    return res

def gdump(obj, filename):
    file = gzip.GzipFile(filename, 'wb')
    pickle.dump(obj, file, -1)
    file.close()


# Function to read dictionaries from a JSON file
def read_from_json_file(filename):
    result = []
    with open(filename, 'r') as file:
        for line in file:
            data_entry = json.loads(line)
            result.append(data_entry)
    return result


def normalize_sym(adj): # From DiffMG/lp/preprocess.py
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_row(mx): # From DiffMG/lp/preprocess.py
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx): # From DiffMG/lp/preprocess.py
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def clean_nx_graph(G, source, target, visualize=False):
    path_subgraph = nx.Graph()
    # 找到所有从start_node到end_node的简单路径
    all_paths = nx.all_simple_paths(G, source=source, target=target)
    unique_paths = []
    # 用于存储已经出现过的节点类型序列
    seen_type_sequences = set()
    # 遍历所有路径
    for path in all_paths:
        # 获取当前路径上的节点类型序列
        #type_sequence = tuple(G.nodes[node]['type'] for node in path)
        type_sequence = tuple(G.nodes[node]['type'][0] for node in path)
        # 检查该类型序列是否已经出现过
        if type_sequence not in seen_type_sequences:
            # 如果这是一个新的类型序列，添加到结果列表中
            unique_paths.append(path)
            # 标记该类型序列已经出现过
            seen_type_sequences.add(type_sequence)
    # 遍历所有路径
    node_count = 0
    for path in unique_paths:
        # 将路径中的节点和边添加到子图中
        # nx.add_path(path_subgraph, path)
        for node in path:
            # 添加节点及其属性
            node_type = G.nodes[node]['type']
            #path_subgraph.add_node(node, **G.nodes[node], type=f'{node_type}{int(node)}')    
            path_subgraph.add_node(node, type=f'{node_type[0]}{int(node)}')    
            node_count += 1
        # 遍历路径中的每对相邻节点以添加边
        for i in range(len(path) - 1):
            # 添加边，不需要指定属性，因为边通常不携带与节点相同的属性
            path_subgraph.add_edge(path[i], path[i+1])  
    if(visualize):
        draw_multiple_graphs([G,path_subgraph,path_subgraph], save_filename='clean_graph.jpg')
    path_subgraph = sort_graph_nodes(path_subgraph)
    return path_subgraph


def sort_graph_nodes(G):
    # Sort nodes in the graph
    sorted_nodes = sorted(G.nodes())
    # Create a new graph with sorted nodes
    sorted_graph = nx.Graph()
    for node in sorted_nodes:
        node_type = G.nodes[node]['type']
        # Add node and its attributes to the new graph
        #sorted_graph.add_node(node, **G.nodes[node])
        sorted_graph.add_node(node, type=f'{node_type[0]}{int(node)}')
    # Add edges to the new graph, ensuring they go from smaller to larger nodes
    for edge in G.edges():
        sorted_graph.add_edge(min(edge), max(edge))
    return sorted_graph


def get_max_path_len(G):
    try:
        source = next(node for node, attr in G.nodes(data=True) if attr.get('label') == 'source')
        target = next(node for node, attr in G.nodes(data=True) if attr.get('label') == 'target')
    except:
        print('Error in get_max_path_len(). Please ensure that G contains one node with label \'source\' and one node with label \'target\'.')
    all_paths = nx.all_simple_paths(G, source, target)
    max_path_len = np.array([len(path) for path in all_paths]).max()
    return max_path_len


def temp_edge_type_lookup(dataset_string):
    if(dataset_string=='yelp'):
        edge_type_lookup_dict = {
            'UU': 0,
            'UB': 1,
            'BU': 2,
            'UO': 3,
            'OU': 4,
            'BA': 5,
            'AB': 6,
            'BI': 7,
            'IB': 8,
            'I': 9,
            'O': 10
        }
    elif(dataset_string=='amazon'):
        edge_type_lookup_dict = {
            'UI': 0, #1,
            'IU': 1, #2,
            'IB': 2, #3,
            'BI': 3, #4,
            'IC': 4, #5,
            'CI': 5, #6,
            'IV': 6, #7,
            'VI': 7, #8,
            'I': 8, #9,
            'O': 9, #10
        }
    elif(dataset_string=='douban_movie'):
        edge_type_lookup_dict = {
            'UU': 0,
            'UM': 1,
            'MU': 2,
            'UG': 3,
            'GU': 4,
            'MA': 5,
            'AM': 6,
            'MD': 7,
            'DM': 8,
            'MT': 9,
            'TM': 10,
            'I': 11,
            'O': 12
        }
    elif(dataset_string=='acm'):
        edge_type_lookup_dict = {
            'PA': 0, 
            'AP': 1, 
            'PS': 2, 
            'SP': 3,
            'I': 4,
            'O': 5
        }
    elif(dataset_string=='imdb'):
        edge_type_lookup_dict = {
            'MD': 0, 
            'DM': 1, 
            'MA': 2, 
            'AM': 3,
            'I': 4,
            'O': 5
        }
    else:
        print(f'dataset: {dataset_string}, not implemented yet.')
        pdb.set_trace()
    return edge_type_lookup_dict


def matrix2logic(matrix, dataset_string):
    nx_meta = convert_to_networkx_graph({'nodes': matrix[0], 'edges': matrix[1]})
    logic = graph2logic(nx_meta, dataset_string, initialization=True)        
    return logic