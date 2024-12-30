import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import json
from tqdm import tqdm
from llama_index.core.schema import QueryBundle
from component.io import load_nodes_jsonl
from component.index.index import get_retriever_from_nodes, update_retriever
from utils.generate_and_execute_slurm_job import generate_and_execute_slurm_job
from llama_index.core.schema import TextNode
import argparse
import math

def load_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--action', type=str, default='main', help='The action to generate retrieve contexts')
    parser.add_argument('--reducer_id', type=int, default=None, help='The index id')

    return parser.parse_args()

def submit_job(
    script_path: str,
    python_file_name: str,
    reducer_id: int,
    action: str
):
    """Submit a job to Slurm and return the job ID."""
    job_name = f'redu_{reducer_id}'
    log_file_path = os.path.abspath(os.path.join(script_path, 'out/{job_name}.out'))
    script_path = os.path.abspath(os.path.join(script_path, 'execute/execute.sh'))
    
    job_name = generate_and_execute_slurm_job(
            python_start_script=f"{python_file_name} --action {action} --reducer_id {reducer_id}",
            account='guest',
            partition='guest-compute',
            job_name=job_name,
            qos='low',
            time='24:00:00',
            num=20,
            log_file_path=log_file_path,
            script_path=script_path
        )
    print(f"[Job: {job_name}] is submitted!")

def generate_cache_retrieved_nodes_dict(filenames, cache_dir='./.cache', needLevel=False):
    retrieved_nodes_dict = {}
    with tqdm(total=len(filenames), desc='loading ...') as pbar:
        for filename in filenames:
            file_path = os.path.join(cache_dir, filename)
            # with open(file_path, 'r') as f:
            #     total_lines = sum(1 for _ in f)
            # file_size = os.path.getsize(file_path)
            with open(file_path, 'r') as input_file: #, tqdm(total=file_size, desc=f"Processing {filename}", unit='B', unit_scale=True, unit_divisor=1024) as line_pbar
                for line in input_file:
                    data = json.loads(line)
                    if data['question_node_id'] not in retrieved_nodes_dict:
                        retrieved_nodes_dict[data['question_node_id']] = {}
                    if data['question_id'] not in retrieved_nodes_dict[data['question_node_id']]:
                        retrieved_nodes_dict[data['question_node_id']][data['question_id']] = {} if needLevel else []
                        
                    if needLevel:
                        for level in ['document', 'section', 'paragraph', 'multi-sentences']:
                            if level not in retrieved_nodes_dict[data['question_node_id']][data['question_id']]:
                                retrieved_nodes_dict[data['question_node_id']][data['question_id']][level] = []
                            nodes = [TextNode.from_dict(node_data) for node_data in data['retrieved_nodes'][level]]
                            retrieved_nodes_dict[data['question_node_id']][data['question_id']][level].extend(nodes)
                    else:
                        nodes = [TextNode.from_dict(node_data) for node_data in data['retrieved_nodes']]
                        retrieved_nodes_dict[data['question_node_id']][data['question_id']].extend(nodes)
                    # line_pbar.update(len(line))
            pbar.update(1)
    
    return retrieved_nodes_dict

def retrieved_node_cache_reducer(question_nodes, retrieved_nodes_dict, retriever_kwargs, reducer_id, cache_prefix, save_dir='./.cache', needLevel=False):
    levels = ['document', 'section', 'paragraph', 'multi-sentences']
    retriever = get_retriever_from_nodes(index_dir_path=None, index_id=None, nodes=[], retriever_kwargs=retriever_kwargs)

    save_path = os.path.join(save_dir, f"{cache_prefix}_{reducer_id}_chroma_not_finish.jsonl")
    # generate retrieved nodes
    with open(save_path, 'w') as save_file:
        with tqdm(total=sum(len(node.metadata['questions_and_embeddings']) for node in question_nodes), desc="retrieving nodes ...") as pbar:
            for node in question_nodes:
                for i, (q, e) in enumerate(node.metadata['questions_and_embeddings'].items()):
                    query_bundle = QueryBundle(query_str=q, embedding=e)
                    if needLevel:
                        data = {
                            'question_node_id': node.id_,
                            'question_id': i,
                            'retrieved_nodes': {}
                        }
                        for level in levels:
                            query = retriever._build_vector_store_query(query_bundle)
                            nodes = retrieved_nodes_dict[node.id_][i][level]
                            update_retriever(nodes, retriever)
                            retrieved_nodes = retriever._vector_store.query(query, include=retriever_kwargs['include']).nodes
                            data['retrieved_nodes'][level] = [node.to_dict() for node in retrieved_nodes]
                        save_file.write(json.dumps(data) + "\n")
                    else: # One
                        query = retriever._build_vector_store_query(query_bundle)
                        nodes = retrieved_nodes_dict[node.id_][i]
                        update_retriever(nodes, retriever)
                        retrieved_nodes = retriever._vector_store.query(query, include=retriever_kwargs['include']).nodes
                        data = {
                            'question_node_id': node.id_,
                            'question_id': i,
                            'retrieved_nodes': [node.to_dict() for node in retrieved_nodes]
                        }
                        save_file.write(json.dumps(data) + "\n")
                    pbar.update(1)
                    
    os.rename(save_path, os.path.join(save_dir, f"{cache_prefix}_{reducer_id}_chroma.jsonl"))

if __name__ == "__main__":
    args = load_args()
    prefix = "gpt-4o-batch-all-target_one" # modify each time
    cache_dir = './.cache'
    action = 'thread'
    
    filenames = [filename for filename in os.listdir(cache_dir) if prefix in filename]
    filenames.sort(key=lambda x: int(x.split('_')[-2]))
    file_batch_size = 15 # modify each time
    
    retriever_kwargs = {
        'similarity_top_k': 50,
        'mode': 'default',
        'break_num': None, # 400000, 100000
        'batch_size': None, # 200000
        'worker': 5,
        "include": ["metadatas", "documents", "embeddings", "distances"]
        # 'worker': None
    }
    
    if args.action == 'main':
        task_num = math.ceil(len(filenames)/file_batch_size)
        for i in range(task_num):
            submit_job(
                script_path=os.getcwd(),
                python_file_name='retrieved_node_cache_reducer.py',
                reducer_id=i,
                action=action
            )
    elif args.action == 'thread':
        question_nodes_path = os.path.abspath('../step_1_get_embedding_value/questions/gpt-4o-batch-all-p_pid_0.jsonl')
        question_nodes = load_nodes_jsonl(question_nodes_path)
        
        retrieved_nodes_dict = generate_cache_retrieved_nodes_dict(
            filenames=filenames[args.reducer_id*file_batch_size:(args.reducer_id+1)*file_batch_size],
            cache_dir=cache_dir,
            needLevel=False
        )
        retrieved_node_cache_reducer(
            question_nodes,
            retrieved_nodes_dict,
            retriever_kwargs=retriever_kwargs,
            reducer_id=args.reducer_id,
            cache_prefix=prefix,
            save_dir='./.reducer_cache',
            needLevel=False
        )
        
    elif args.action == 'thread_level':
        question_nodes_path = os.path.abspath('../step_1_get_embedding_value/questions/gpt-4o-batch-all-p_pid_0.jsonl')
        question_nodes = load_nodes_jsonl(question_nodes_path)
        
        retrieved_nodes_dict = generate_cache_retrieved_nodes_dict(
            filenames=filenames[args.reducer_id*file_batch_size:(args.reducer_id+1)*file_batch_size],
            cache_dir=cache_dir,
            needLevel=True
        )
        retrieved_node_cache_reducer(
            question_nodes,
            retrieved_nodes_dict,
            retriever_kwargs=retriever_kwargs,
            reducer_id=args.reducer_id,
            cache_prefix=prefix,
            save_dir='./.reducer_cache',
            needLevel=True
        )
        