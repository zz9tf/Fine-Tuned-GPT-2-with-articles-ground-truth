import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from llama_index.core import VectorStoreIndex, PropertyGraphIndex
from component.io import save_storage_context
from component.models.embed.get_embedding_model import get_embedding_model
from component.index.get_a_store import get_a_store
from llama_index.core import Settings
from llama_index.core import StorageContext

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
    
if __name__ == "__main__":
    pass