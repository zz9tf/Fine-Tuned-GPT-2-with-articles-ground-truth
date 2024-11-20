import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import json
from llama_index.core import VectorStoreIndex
import re
from component.models.embed.get_embedding_model import get_embedding_model
from component.io import load_nodes_jsonl, load_nodes_jsonl_corresponding_levels
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.schema import MetadataMode, TextNode
from tqdm import tqdm
from typing import List

def get_an_index_generator(index_type):
        if index_type == 'VectorStoreIndex':
            return VectorStoreIndex
        else:
            raise Exception("Invalid index generator name. Please provide name in [{}]".format('VectorStoreIndex'))

def storing_nodes_for_index(embedding_config: dict, index_dir_path: str, index_id: str, device: str = None, nodes: List[TextNode]=None, input_file_path: str=None):
    assert (nodes or input_file_path) is not None, 'Provide at least input_file_path for nodes, or directly provide nodes'
    print(f"save to: {index_dir_path}")
    if not os.path.exists(index_dir_path):
        os.makedirs(index_dir_path)
    if input_file_path is not None:
        nodes = load_nodes_jsonl(input_file_path)
    save_path = os.path.join(index_dir_path, index_id) + '_not_finish.jsonl'
    finished_nodesIds = set()
    if os.path.exists(save_path):
        finished_nodes = load_nodes_jsonl(save_path)
        finished_nodesIds = {node.id_ for node in finished_nodes}
    # Load embedding model
    embed_model = get_embedding_model(embedding_config, device)
        
    with open(save_path, 'w') as file:
        for node in tqdm(nodes, desc='generating embeddings...'):
            if node.id_ in finished_nodesIds: continue
            embedding = embed_model._get_text_embedding(node.get_content(MetadataMode.EMBED))
            node.embedding = embedding
            json.dump(node.to_dict(), file)
            file.write('\n')
    final_save_path = os.path.join(index_dir_path, index_id) + '.jsonl'
    os.rename(save_path, final_save_path)
    print(f"File: {index_id+'.jsonl'} has been saved at {os.path.abspath(index_dir_path)}")

def embedding_addtional_requiring_embeddings_key(embedding_config: dict, input_file_path, index_dir_path: str, index_id: str):
    if not os.path.exists(index_dir_path):
        os.makedirs(index_dir_path)
    
    nodes = load_nodes_jsonl(input_file_path)
    save_path = os.path.join(index_dir_path, index_id) + '_not_finish.jsonl'
    
    finished_nodesIds = set()
    if os.path.exists(save_path):
        finished_nodes = load_nodes_jsonl(save_path)
        finished_nodesIds = {node.id_ for node in finished_nodes}
        
    # Load embedding model
    embed_model = get_embedding_model(embedding_config)
        
    with open(save_path, 'w') as file:
        for node in tqdm(nodes, desc='generating embeddings...'):
            if node.id_ in finished_nodesIds: continue
            embedding = embed_model._get_text_embedding(node.get_content(MetadataMode.EMBED))
            node.embedding = embedding
            json.dump(node.to_dict(), file)
            file.write('\n')
    
    final_save_path = os.path.join(index_dir_path, index_id) + 'with_additional_embeddings.jsonl'
    os.rename(save_path, final_save_path)
    print(f"File: {index_id+'.jsonl'} has been saved at {os.path.abspath(index_dir_path)}")

def merge_database_pid_nodes(index_dir_path: str, index_id: str):
    filenames = [f for f in os.listdir(index_dir_path) if bool(re.match(r'^\d+\.jsonl$', f))]
    filenames.sort(key=lambda x: int(x.split('.')[0]))
    print(f"Detect pid files: {len(filenames)}")
    save_path = os.path.join(index_dir_path, index_id) + '_not_finish.jsonl'
    with open(save_path, 'w') as save_file:
        for filename in filenames:
            with open(os.path.join(index_dir_path, filename), 'r', encoding='utf-8') as input_file:
                file_size = os.path.getsize(os.path.join(index_dir_path, filename))
                with tqdm(total=file_size, desc=f'merging {filename}...', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                    for i, line in enumerate(input_file):
                        try:
                            node_data = json.loads(line)
                            node = TextNode.from_dict(node_data)
                            json.dump(node.to_dict(), save_file)
                            save_file.write('\n')
                            pbar.update(len(line))
                        except:
                            print(i, line)
    new_save_path = os.path.join(index_dir_path, index_id) + '.jsonl'
    os.rename(save_path, new_save_path)

def get_retriever_from_nodes(index_dir_path, index_id, retriever_kwargs=None, break_num: int=None, nodes=None):
    # Generate index for nodes
    if nodes == None:
        nodes = load_nodes_jsonl(os.path.join(index_dir_path, index_id) + '.jsonl', break_num)
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    v = VectorStoreIndex(
        index_struct=VectorStoreIndex.index_struct_cls(index_id=index_id),
        storage_context=storage_context,
        embed_model='skip'
    )
    
    nodes_batch = []
    for node in nodes:
        nodes_batch.append(node)
        if len(nodes_batch) >= 2048:
            v._vector_store.add(nodes_batch)
            nodes_batch = []
    if len(nodes_batch) > 0:
        v._vector_store.add(nodes_batch)
    
    retriever = v.as_retriever(**retriever_kwargs)
    return retriever

def update_retriever(nodes, retriever):
    retriever._vector_store.client._client.delete_collection(retriever._vector_store.client.name)
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart")
    retriever._vector_store._collection = chroma_collection
    
    nodes_batch = []
    for node in nodes:
        nodes_batch.append(node)
        if len(nodes_batch) >= 2048:
            retriever._vector_store.add(nodes_batch)
            nodes_batch = []
    if len(nodes_batch) > 0:
        retriever._vector_store.add(nodes_batch)

def get_chroma_retriever_from_nodes(index_dir_path, index_id, retriever_kwargs=None, break_num: int=None):
    # Generate index for nodes
    file_path = os.path.join(index_dir_path, index_id) + '.jsonl'
    db_path = os.path.join(index_dir_path, index_id+'_chroma')
    chroma_client = chromadb.PersistentClient(path=db_path)
    try:
        # Check if collection already exists
        chroma_collection = chroma_client.get_collection(name='quickstart')
    except Exception as e:
        chroma_collection = chroma_client.create_collection(name='quickstart')
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Get the total file size
    file_size = os.path.getsize(file_path)
    nodes = []
    
    # Read the file and track progress based on bytes read
    with open(file_path, 'r') as file:
        with tqdm(total=file_size, desc=f'Loading {file_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for i, line in enumerate(file):
                if break_num is not None and i == break_num:
                    break
                node_data = json.loads(line)
                node = TextNode.from_dict(node_data)
                node.metadata = {k: str(v) for k, v in node.metadata.items()}
                nodes.append(node)
                if len(nodes) >= 2048:
                    vector_store.add(nodes)
                    nodes = []
                # Update progress bar based on bytes read
                pbar.update(len(line))
    if len(nodes) > 0:
        vector_store.add(nodes)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    v = VectorStoreIndex(
        index_struct=VectorStoreIndex.index_struct_cls(index_id=index_id),
        storage_context=storage_context,
        embed_model='skip'
    )
    
    retriever = v.as_retriever(**retriever_kwargs)
    return retriever
    
def get_all_chroma_level_retrievers_from_nodes(index_dir_path, index_id, retriever_kwargs=None, break_num: int=None):
    basic_db_path = os.path.join(index_dir_path, index_id)
    def get_vector_store(basic_db_path, level):
        chroma_client = chromadb.PersistentClient(path=f'{basic_db_path}_{level}_chroma')
        try:
            # Check if collection already exists
            chroma_collection = chroma_client.get_collection(name='quickstart')
        except Exception as e:
            chroma_collection = chroma_client.create_collection(name='quickstart')
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return vector_store
    
    levels = ['document', 'section', 'paragraphs', 'multi-sentences']
    
    level_to_vector_store = {level:get_vector_store(basic_db_path, level) for level in levels}
    # Generate index for nodes
    file_path = os.path.join(index_dir_path, index_id) + '.jsonl'
    
    # Get the total file size
    file_size = os.path.getsize(file_path)
    level_to_nodes = {level:[] for level in levels}
    
    # Read the file and track progress based on bytes read
    with open(file_path, 'r') as file:
        with tqdm(total=file_size, desc=f'Loading {file_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for i, line in enumerate(file):
                if break_num is not None and i == break_num:
                    break
                node_data = json.loads(line)
                node = TextNode.from_dict(node_data)
                node.metadata = {k: str(v) for k, v in node.metadata.items()}
                level_to_nodes[node.metadata['level']].append(node)
                for level, nodes in level_to_nodes.items():
                    if len(nodes) == 2048:
                        level_to_vector_store[level].add(nodes)
                        level_to_vector_store[level] = []
                # Update progress bar based on bytes read
                pbar.update(len(line))
    
    for level, nodes in level_to_nodes.items():
        if len(nodes) != 0:
            level_to_vector_store[level].add(nodes)
                
    level_to_retriever = {}
    for level, vector_store in level_to_vector_store.items():
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        v = VectorStoreIndex(
            index_struct=VectorStoreIndex.index_struct_cls(index_id=index_id),
            storage_context=storage_context,
            embed_model='skip'
        )
        level_to_retriever[level] = v.as_retriever(**retriever_kwargs)
        
    return level_to_retriever  

def get_chroma_retriever_from_storage(retriever_dir, chroma_db_name, retriever_kwargs=None):
    db_path = os.path.join(retriever_dir, chroma_db_name)
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_collection(name='quickstart')
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    v = VectorStoreIndex(
        index_struct=VectorStoreIndex.index_struct_cls(index_id=chroma_db_name),
        storage_context=storage_context,
        embed_model='skip'
    )
    
    return v.as_retriever(**retriever_kwargs)

def get_chroma_retriever_from_storage(retriever_dir, chroma_db_name, retriever_kwargs=None):
    db_path = os.path.join(retriever_dir, chroma_db_name)
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_collection(name='quickstart')
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    v = VectorStoreIndex(
        index_struct=VectorStoreIndex.index_struct_cls(index_id=chroma_db_name),
        storage_context=storage_context,
        embed_model='skip'
    )
    
    return v.as_retriever(**retriever_kwargs)

if __name__ == "__main__":
    get_chroma_retriever_from_nodes(
        index_dir_path=os.path.abspath('../../database/gpt-4o-batch-all-target'),
        index_id='all',
        retriever_kwargs=None,
        break_num=None
    )