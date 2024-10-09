import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from component.index.index import get_retriever_from_nodes
from configs.load_config import load_configs
from llama_index.core.schema import QueryBundle

total_config, perfix_config = load_configs()

# Load retriever
root_path = os.path.abspath('../../../..')
index_name = total_config['document_preprocessing']['index_pipelines'][0]
index_dir_path = os.path.abspath(os.path.join(root_path, total_config['indexes_dir_path'], index_name))
index_id = 'all'
retriever = get_retriever_from_nodes(index_dir_path=index_dir_path, index_id=index_id)

# Load questions


nodes = retriever.retrieve()
for i, node in enumerate(nodes):
    print(f"{i}, {node.text}")