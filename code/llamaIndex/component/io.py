import os
import json
from typing import List, Dict
from tqdm import tqdm
from llama_index.core.schema import BaseNode, TextNode

def save_nodes_jsonl(file_path: str, nodes: List[BaseNode]):
    try:
        with open(file_path, 'w') as file:
            for node in tqdm(nodes, desc=f'Saving {file_path.split(os.path.sep)[-1]}'):
                json.dump(node.to_dict(), file)
                file.write('\n')
    except Exception as e:
        print(f"An error occurred while saving nodes: {e}")

def load_nodes_jsonl(file_path: str, break_num: int=None) -> List[TextNode]:
    nodes = []
    
    try:
        # Get the total file size
        file_size = os.path.getsize(file_path)
        
        # Read the file and track progress based on bytes read
        with open(file_path, 'r') as file:
            with tqdm(total=file_size, desc=f'Loading {file_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for i, line in enumerate(file):
                    if break_num is not None and i == break_num:
                        break
                    node_data = json.loads(line)
                    nodes.append(TextNode.from_dict(node_data))
                    # Update progress bar based on bytes read
                    pbar.update(len(line))
    except Exception as e:
        print(f"An error occurred while loading nodes: {e}")
    
    return nodes

def load_nodes_jsonl_corresponding_levels(file_path: str, break_num: int=None) -> Dict[str, List[TextNode]]:
    level_to_nodes = {}
    
    try:
        # Get the total file size
        file_size = os.path.getsize(file_path)
        
        # Read the file and track progress based on bytes read
        with open(file_path, 'r') as file:
            with tqdm(total=file_size, desc=f'Loading {file_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for i, line in enumerate(file):
                    if break_num is not None and i == break_num:
                        break
                    node_data = json.loads(line)
                    node = TextNode.from_dict(node_data)
                    if node.metadata['level'] not in level_to_nodes:
                        level_to_nodes[node.metadata['level']] = []
                    level_to_nodes[node.metadata['level']].append(node)
                    # Update progress bar based on bytes read
                    pbar.update(len(line))
    except Exception as e:
        print(f"An error occurred while loading nodes: {e}")
    
    return level_to_nodes
