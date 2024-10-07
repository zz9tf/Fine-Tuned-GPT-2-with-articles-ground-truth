import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from typing import List
from llama_index.core import VectorStoreIndex, PropertyGraphIndex
from component.models.embed.get_embedding_model import get_embedding_model
from component.index.get_a_store import get_a_store
from component.index.store_io import load_storage_from_persist_dir, load_index_from_storage, save_storage_context
from llama_index.core import Settings
from llama_index.core import StorageContext
# from llama_index.core import load_index_from_storage

from tqdm import tqdm

def get_an_index_generator(index_type):
        if index_type == 'VectorStoreIndex':
            return VectorStoreIndex
        elif index_type == 'PropertyGraphIndex':
            return PropertyGraphIndex
        else:
            raise Exception("Invalid embedding model name. Please provide embedding models {}".format())
        
def generate_index(embedding_config, config, nodes, index_dir_path=None, index_id=None):
    # Load embedding model
    Settings.embed_model = get_embedding_model(embedding_config)

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

def merge_index(
    config: dict, index_dir_path: str, index_ids: List[str], target_id: str
):
    # Create a new StorageContext
    merged_storage_context = StorageContext.from_defaults(
            docstore=get_a_store(config['docstore']),
            vector_store=get_a_store(config['vector_store']),
            index_store=get_a_store(config['index_store']),
            property_graph_store=get_a_store(config['property_graph_store'])
        )
    
    print(merged_storage_context.index_store.index_structs())
    # Load each index into the new StorageContext
    for index_id in tqdm(index_ids, desc="merging index..."):
        # storage_context = StorageContext.from_defaults(persist_dir=os.path.join(index_dir_path, index_id))
        
        storage_context = load_storage_from_persist_dir(persist_dir=os.path.join(index_dir_path, index_id))
        index = load_index_from_storage(storage_context, index_id=index_id)
        for v in index.storage_context.index_store.index_structs():
            print(v[0])
        exit()
        
    
    print(merged_storage_context.index_store.index_structs())
    exit()
    # add the new index struct
    storage_context.index_store.add_index_struct(target_id)
    save_storage_context(merged_storage_context, os.path.join(index_dir_path, target_id))
    print(f"New merged index has been saved at {os.path.join(index_dir_path, target_id)}")
    
if __name__ == "__main__":
    from configs.load_config import load_configs
    _, prefix_config = load_configs()
    
    config = prefix_config['storage']['simple']
    
    merge_index(
        config=config,
        index_dir_path=os.path.abspath(os.path.join('../../database', 'gpt-4o-batch-all-target')),
        index_ids=['3','5'],
        target_id='test'
    )