import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.graph_stores.simple import SimpleGraphStore

def get_a_store(store_type):
    if store_type == 'SimpleDocumentStore':
        return SimpleDocumentStore()
    elif store_type == 'SimpleIndexStore':
        return SimpleIndexStore()
    elif store_type == 'SimpleVectorStore':
        return SimpleVectorStore()
    elif store_type == 'SimpleGraphStore':
        return SimpleGraphStore()