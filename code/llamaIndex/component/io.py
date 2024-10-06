import os
import json
from typing import List
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

def load_nodes_jsonl(file_path: str) -> List[TextNode]:
    nodes = []
    
    try:
        # Get the total file size
        file_size = os.path.getsize(file_path)
        
        # Read the file and track progress based on bytes read
        with open(file_path, 'r') as file:
            with tqdm(total=file_size, desc=f'Loading {file_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for line in file:
                    node_data = json.loads(line)
                    nodes.append(TextNode.from_dict(node_data))
                    # Update progress bar based on bytes read
                    pbar.update(len(line))
    except Exception as e:
        print(f"An error occurred while loading nodes: {e}")
    
    return nodes

from pathlib import Path
def save_storage_context(storage_context, persist_dir):
    # TODO: dealing with large file
    persist_dir = Path(persist_dir)
    docstore_path = str(persist_dir / "docstore.json")
    index_store_path = str(persist_dir / "index_store.json")
    graph_store_path = str(persist_dir / "graph_store.json")
    pg_graph_store_path = str(persist_dir / "property_graph_store.json")

    storage_context.docstore.persist(persist_path=docstore_path)
    storage_context.index_store.persist(persist_path=index_store_path)
    storage_context.graph_store.persist(persist_path=graph_store_path)

    if storage_context.property_graph_store:
        storage_context.property_graph_store.persist(persist_path=pg_graph_store_path)

    # save each vector store under it's namespace
    for vector_store_name, vector_store in storage_context.vector_stores.items():
        vector_store_path = str(
                Path(persist_dir)
                / f"{vector_store_name}{'__'}{'vector_store.json'}"
            )

        vector_store.persist(persist_path=vector_store_path)
        
        
import logging
from typing import Any, List, Optional
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.registry import INDEX_STRUCT_TYPE_TO_INDEX_CLASS

logger = logging.getLogger(__name__)
def load_index_from_storage(
    storage_context: StorageContext,
    index_id: Optional[str],
    **kwargs: Any,
) -> List[BaseIndex]:
    """Load multiple indices from storage context.

    Args:
        storage_context (StorageContext): storage context containing
            docstore, index store and vector store.
        index_id (Optional[Sequence[str]]): IDs of the indices to load.
            Defaults to None, which loads all indices in the index store.
        **kwargs: Additional keyword args to pass to the index constructors.
    """
    logger.info(f"Loading indices with ids: {index_id}")
    index_structs = []
    
    index_struct = storage_context.index_store.get_index_struct(index_id)
    if index_struct is None:
        raise ValueError(f"Failed to load index with ID {index_id}")
    index_structs.append(index_struct)

    type_ = index_struct.get_type()
    index_cls = INDEX_STRUCT_TYPE_TO_INDEX_CLASS[type_]
    index = index_cls(
        index_struct=index_struct, storage_context=storage_context, **kwargs
    )
    
    return index