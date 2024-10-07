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
        
import json
def load_kvstore_from_persist_path(
        persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
):
        """Load a SimpleKVStore from a persist path and filesystem."""
        fs = fs or fsspec.filesystem("file")
        logger.debug(f"Loading {__name__} from {persist_path}.")
        with fs.open(persist_path, "rb") as f:
            data = json.load(f)
        return cls(data)
        
import logging
from typing import Any, List, Optional
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.registry import INDEX_STRUCT_TYPE_TO_INDEX_CLASS

logger = logging.getLogger(__name__)
def load_index_from_storage(
    storage_context: StorageContext,
    index_id: str,
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
    
    index_struct = storage_context.index_store.get_index_struct(index_id)
    if index_struct is None:
        raise ValueError(f"Failed to load index with ID {index_id}")

    type_ = index_struct.get_type()
    index_cls = INDEX_STRUCT_TYPE_TO_INDEX_CLASS[type_]
    index = index_cls(
        index_struct=index_struct, storage_context=storage_context, **kwargs
    )
    
    return index

from tqdm import tqdm
import fsspec
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore
from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.vector_stores.simple import SimpleVectorStore

def load_storage_from_persist_dir(
    persist_dir: Optional[str] = None,
    show_process: bool = True
) -> "StorageContext":
    """Load a StorageContext from persist_dir."""
    
    # Function to load with progress updates
    def load_with_progress(load_func, description):
        if show_process:
            pbar.set_postfix_str(description)
            result = load_func(persist_dir, fs=None)
            pbar.update(1)
            return result
        else:
            return load_func(persist_dir, fs=None)

    # Initialize progress bar if required
    total_steps = 4  # 4 stores + 1 for vector store if needed
    pbar = tqdm(total=total_steps, desc="Loading storage...") if show_process else None

    # Load components with progress
    docstore = load_with_progress(SimpleDocumentStore.from_persist_dir, "Loading docstore")
    index_store = load_with_progress(SimpleIndexStore.from_persist_dir, "Loading index store")
    graph_store = load_with_progress(SimpleGraphStore.from_persist_dir, "Loading simple graph store")

    # Attempt to load property graph store
    property_graph_store = None
    vector_stores = None
    try:
        property_graph_store = load_with_progress(SimplePropertyGraphStore.from_persist_dir, "Loading property graph store")
    except FileNotFoundError:
        vector_stores=load_with_progress(SimpleVectorStore.from_namespaced_persist_dir, "Loading simple vector store")

    # Close the progress bar if it was used
    if pbar:
        pbar.close()

    # Return the loaded storage context
    return StorageContext(
        docstore=docstore,
        index_store=index_store,
        vector_stores=vector_stores,
        graph_store=graph_store,
        property_graph_store=property_graph_store
    )