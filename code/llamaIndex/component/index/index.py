import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from typing import List
from llama_index.core import VectorStoreIndex, PropertyGraphIndex
from component.io import save_storage_context
from component.models.embed.get_embedding_model import get_embedding_model
from component.index.get_a_store import get_a_store
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from tqdm import tqdm

def get_an_index_generator(index_type):
        if index_type == 'VectorStoreIndex':
            return VectorStoreIndex
        else:
            raise Exception("Invalid embedding model name. Please provide embedding models {}".format())

def storage_nodes_for_index(embedding_config):
    # Load embedding model
    embed_model = get_embedding_model(embedding_config)

    # Save index
    # if index_dir_path != None:
    #     if index_id != None:
    #         index.set_index_id(index_id)
    #     save_storage_context(index.storage_context, index_dir_path)
        
    # return index

def get_index_from_nodes(nodes, config, index_dir_path=None, index_id=None):
    # Generate index for nodes
    index_generator = get_an_index_generator(config['index_generator'])
    index = index_generator(
        nodes=nodes,
        storage_context=StorageContext.from_defaults(
            docstore=get_a_store(config['docstore']),
            vector_store=get_a_store(config['vector_store']),
            index_store=get_a_store(config['index_store']),
            property_graph_store=get_a_store(config['property_graph_store'])
        ),
        show_progress=True
    )

    # Save index
    if index_dir_path != None:
        if index_id != None:
            index.set_index_id(index_id)
        save_storage_context(index.storage_context, index_dir_path)
        
    return index
    
if __name__ == "__main__":
    from configs.load_config import load_configs
    _, prefix_config = load_configs()
    
    config = prefix_config['storage']['simple']
    
    merge_index(
        config=config,
        index_dir_path=os.path.abspath(os.path.join('../../database', 'gpt-4o-batch-all-target')),
        index_ids=['1','2'],
        target_id='test'
    )