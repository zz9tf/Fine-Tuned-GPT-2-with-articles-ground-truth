import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import random
import itertools
import pandas as pd

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.gridspec import GridSpec


# def load_4_sample_dataset():
#     all_embeddings = []
#     all_levels = []

#     files = os.listdir(os.path.abspath('../step_1_get_embedding_value/contexts'))
#     files = sorted(files, key=lambda x: int(x.split('.')[0].split('_')[-1]))

#     for file_name in files[:10]:
#         embeddings_file_path = f"../step_1_get_embedding_value/contexts/{file_name}"
#         print(embeddings_file_path)
#         embeddings, levels = load_jsonl(embeddings_file_path)
        
#         # Append to the combined lists
#         all_embeddings.append(embeddings)
#         all_levels.append(levels)

#     # Combine all_embeddings and all_levels into a single list of tuples for easy random selection
#     num_samples = 4
#     selected_indices = random.sample(range(len(all_embeddings)), num_samples)
#     print(selected_indices)
#     selected_indices = [1,4,8,9]

#     # Randomly select 4 tuples
#     selected_embeddings = [all_embeddings[i] for i in selected_indices]
#     selected_levels = [all_levels[i] for i in selected_indices]

#     selected_embeddings = list(itertools.chain.from_iterable(selected_embeddings))
#     selected_levels = list(itertools.chain.from_iterable(selected_levels))

#     df_total = pd.DataFrame({
#         'embeddings': [tuple(embedding) for embedding in selected_embeddings],
#         'levels': selected_levels
#     })
#     return df_total


from component.io import load_nodes_jsonl
from anytree import Node, RenderTree
def load_5_documents_nodes():
    database_node_file_path = os.path.abspath('../../database/gpt-4o-batch-all-target/0.jsonl')
    nodes = load_nodes_jsonl(database_node_file_path)
    # Assuming 'nodes' is a list of nodes that each have metadata like 'level' and 'ref_doc_id'
    # and 'id_' which is an identifier of the node.
    level2nodes = {'document': []}

    # Build a dictionary to store nodes at different levels
    for node in nodes:
        level = node.metadata['level']
        if level == 'document':
            level2nodes[level].append(node)
        else:
            if level not in level2nodes:
                level2nodes[level] = {}
            if node.ref_doc_id not in level2nodes[level]:
                level2nodes[level][node.ref_doc_id] = []
            level2nodes[level][node.ref_doc_id].append(node)

    # Get the top-level nodes (root) to build the tree
    target_ids = [node.id_ for node in level2nodes['document'][:5]]
    target_nodes = level2nodes['document'][:5]

    # Create a root node for visualization
    root = Node("Root")  # this is the root node of the entire structure

    # Dictionary to store anytree Node objects for reference
    tree_nodes = {}

    # Create nodes in the tree for the 'document' level
    for doc_node in target_nodes:
        tree_nodes[doc_node.id_] = Node(f"Document {doc_node.id_}", parent=root)

    # Traverse the remaining levels and create child nodes
    for level in level2nodes.keys():
        if level == 'document':
            continue
        new_target_ids = []
        for target_id in target_ids:
            new_target_nodes = level2nodes[level][target_id]
            for new_node in new_target_nodes:
                tree_nodes[new_node.id_] = Node(f"{level} Node {new_node.id_}", parent=tree_nodes[target_id])
            new_target_ids.extend([node.id_ for node in new_target_nodes])
            target_nodes.extend(new_target_nodes)
        target_ids = new_target_ids

    # Print the tree structure
    for pre, fill, node in RenderTree(root):
        print(f"{pre}{node.name}")
    
    return target_nodes