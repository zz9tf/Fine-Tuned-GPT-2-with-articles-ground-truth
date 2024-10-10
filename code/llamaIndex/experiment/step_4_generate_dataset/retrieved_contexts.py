import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import json
from tqdm import tqdm
from component.index.index import get_retriever_from_nodes
from configs.load_config import load_configs
from llama_index.core.schema import QueryBundle
from component.io import load_nodes_jsonl

def generate_retrieved_contexts(question_nodes, retriever, save_path):
    with open(save_path, 'w') as save_file:
        for node in tqdm(question_nodes, desc='Generating retrieved contexts...'):
            for q, e in node.metadata['questions_and_embeddings'].items():
                query = QueryBundle(q)
                query.embedding = e
                retrieved_nodes = retriever.retrieve(query)
                
                data = {
                    'question_node_id': node.id_,
                    'retrieved_nodes_id': [n.id_ for n in retrieved_nodes],
                    'retrieved_contexts': [n.text for n in retrieved_nodes]
                }
                save_file.write(json.dumps(data) + "\n")

def generate_contexts_with_multi_level_retriever(
    question_nodes_path: str = os.path.abspath('../step_1_get_embedding_value/questions/gpt-4o-batch-all-p_extract_1.jsonl'),
    retrieved_contexts_save_path: str = './retrieved_contexts/multi_level_retrieved_contexts.jsonl'
):
    total_config, _ = load_configs()
    root_path = os.path.abspath('../../../..')
    index_name = total_config['document_preprocessing']['index_pipelines'][0]
    index_dir_path = os.path.abspath(os.path.join(root_path, total_config['indexes_dir_path'], index_name))
    index_id = 'all'
    
    retriever = get_retriever_from_nodes(
        index_dir_path=index_dir_path, 
        index_id=index_id,
        retriever_kwargs={
            'similarity_top_k': 5
        }
    )
    
    question_nodes = load_nodes_jsonl(question_nodes_path)
     
    generate_retrieved_contexts(question_nodes, retriever, retrieved_contexts_save_path)

def load_retrieved_contexts(
    retrieved_contexts_file_path: str = './retrieved_contexts/multi_level_retrieved_contexts.jsonl'
):
    file_size = os.path.getsize(retrieved_contexts_file_path)
    contexts = []
    # Read the file and track progress based on bytes read
    with open(retrieved_contexts_file_path, 'r') as input_file:
        with tqdm(total=file_size, desc=f'Loading {retrieved_contexts_file_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for line in input_file:
                row = json.loads(line)
                contexts.append(row['retrieved_contexts'])
                # Update progress bar based on bytes read
                pbar.update(len(line))
    
    return contexts

if __name__ == "__main__":
    generate_contexts_with_multi_level_retriever()
