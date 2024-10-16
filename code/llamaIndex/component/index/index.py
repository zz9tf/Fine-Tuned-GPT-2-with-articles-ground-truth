import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import json
from llama_index.core import VectorStoreIndex
import re
from component.models.embed.get_embedding_model import get_embedding_model
from component.io import load_nodes_jsonl
from llama_index.core.schema import MetadataMode, TextNode
from tqdm import tqdm

def get_an_index_generator(index_type):
        if index_type == 'VectorStoreIndex':
            return VectorStoreIndex
        else:
            raise Exception("Invalid index generator name. Please provide name in [{}]".format('VectorStoreIndex'))

def storing_nodes_for_index(embedding_config: dict, input_file_path, index_dir_path: str, index_id: str, device: str = None):
    if not os.path.exists(index_dir_path):
        os.makedirs(index_dir_path)
    
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
    with open(os.path.join(index_dir_path, index_id) + '.jsonl', 'w') as save_file:
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
                

def get_retriever_from_nodes(index_dir_path, index_id, retriever_kwargs=None):
    # Generate index for nodes
    nodes = load_nodes_jsonl(os.path.join(index_dir_path, index_id) + '.jsonl')
    
    v = VectorStoreIndex(
        index_struct=VectorStoreIndex.index_struct_cls(index_id=index_id)
    )
    v._add_nodes_to_index(
        v._index_struct,
        nodes,
        show_progress=False
    )
    
    retriever = v.as_retriever(**retriever_kwargs)
    return retriever
    
if __name__ == "__main__":
    pass