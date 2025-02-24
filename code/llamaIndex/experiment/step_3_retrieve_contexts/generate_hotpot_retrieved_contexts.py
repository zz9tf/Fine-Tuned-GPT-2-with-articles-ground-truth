import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import argparse
import json
from tqdm import tqdm
import re
from configs.load_config import load_configs
from component.io import load_nodes_jsonl
from component.index.index import get_chroma_retriever_from_storage
from utils.generate_and_execute_slurm_job import generate_and_execute_slurm_job
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
from llama_index.core.vector_stores.types import (
    VectorStoreQuery
)
import torch.nn.functional as F  # For softmax
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
import numpy as np

def load_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--action', type=str, default='main', help='The action to generate retrieve contexts')
    parser.add_argument('--index_id', type=str, default=None, help='The index id')
    parser.add_argument('--index_dir', type=str, default=None, help='The index id dir name')
    parser.add_argument('--retrieved_mode', type=str, default=None, help='The retrieved_mode to be used in retrieve')
    parser.add_argument('--top_k', type=str, default=None, help='The top_k similarities to be retrieved')
    parser.add_argument('--need_level', type=lambda x: x.lower() == 'true', default=True, help='If apply multiple level mode or not')

    return parser.parse_args()

def submit_job(
    script_path: str,
    python_file_name: str,
    index_id: str,
    index_dir: str,
    retrieved_mode: str,
    top_k: str,
    action: str,
    need_level: bool
):
    """Submit a job to Slurm and return the job ID."""
    job_name = f'{retrieved_mode}_{index_id}_{index_dir}'
    log_file_path = os.path.abspath(os.path.join(script_path, 'out/{job_name}.out'))
    script_path = os.path.abspath(os.path.join(script_path, 'execute/execute.sh'))
    
    job_name = generate_and_execute_slurm_job(
            python_start_script=f"{python_file_name} --action {action} --index_id {index_id} --index_dir {index_dir} --retrieved_mode {retrieved_mode} --top_k {top_k} --need_level {need_level}",
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

def _get_retrievers(need_level, levels, chroma_db_name, database_dir):
    retrievers = {}
    
    if need_level:
        for level in levels:
            level_chroma_db_name = f"_{level}_".join(chroma_db_name.split('_'))
            db_path = os.path.join(database_dir, level_chroma_db_name)
            retrievers[level] = {
                'retriever': get_chroma_retriever_from_storage(db_path, level_chroma_db_name, retriever_kwargs),
                'db_path': db_path
            }
    else:
        db_path = os.path.join(database_dir, chroma_db_name)
        retrievers[None] = {
                'retriever': get_chroma_retriever_from_storage(db_path, chroma_db_name, retriever_kwargs),
                'db_path': db_path
        }
    return retrievers

def _count_start_line_number(save_path):
    # Count start number
    line_number = 0
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        with open(save_path, 'r') as save_file:
            with tqdm(total=file_size, desc=f'Counting lines in {save_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for line in save_file:
                    pbar.update(len(line))
                    line_number += 1
    
    print(f"Start number: {line_number}")
    return line_number

def _get_VectorStoreQuery(query_str: str, embedding: List[float], retriever):
    return VectorStoreQuery(
        query_embedding=embedding,
        similarity_top_k=retriever._similarity_top_k,
        node_ids=retriever._node_ids,
        doc_ids=retriever._doc_ids,
        query_str=query_str,
        mode=retriever._vector_store_query_mode,
        alpha=retriever._alpha,
        filters=None,
        sparse_top_k=retriever._sparse_top_k,
        hybrid_top_k=retriever._hybrid_top_k,
    )

def _generate_node_rank_info(q, e, retrievers, level):
    query = _get_VectorStoreQuery(query_str=q, embedding=e, retriever=retrievers[level]['retriever'])
    results = retrievers[level]['retriever']._vector_store.query(query)
    
    return [
        {
            'similarity': s, 
            'id': node_id, 
            'db_path': retrievers[level]['db_path']
        }
        for s, node_id in zip(results.similarities, results.ids)
    ]

def generate_retrieved_hotpot_cache(
    hotpot_question_path, database_dir, chroma_db_name, cache_prefix, save_dir='./.cache', need_level=True
):
    levels = ['document', 'section', 'paragraph', 'multi-sentences']
    retrievers = _get_retrievers(need_level, levels, chroma_db_name, database_dir)
    save_path = os.path.join(save_dir, f"{cache_prefix}_{chroma_db_name}_not_finish.jsonl")
    line_number = _count_start_line_number(save_path)
    
    file_size = os.path.getsize(hotpot_question_path)
    with open(save_path, 'a') as save_file, open(hotpot_question_path, 'r') as file:
        with tqdm(total=file_size, desc=f'Loading {hotpot_question_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for i, line in enumerate(file):
                if i < line_number:
                    continue
                qa = json.loads(line)
                data = {
                    '_id': qa['_id'],
                    'node_rank_info': {}
                }
                if need_level:
                    for level in levels:
                        data['node_rank_info'][level] = _generate_node_rank_info(qa['question'], qa['embedding'], retrievers, level)
                else:
                    data['node_rank_info'][None] = _generate_node_rank_info(qa['question'], qa['embedding'], retrievers, None)
                save_file.write(json.dumps(data) + "\n")
                pbar.update(len(line))
    os.rename(save_path,  os.path.join(save_dir, f"{cache_prefix}_{chroma_db_name}.jsonl"))

def _merge_rank_info_from_different_files(filenames, cache_dir):
    rank_info_dict = {}
    
    with tqdm(total=len(filenames), desc='loading ...') as pbar:
        for filename in filenames:
            pbar.set_description_str(filename)
            input_file = open(os.path.join(cache_dir, filename), 'r')
            for line in input_file:
                data = json.loads(line)
                if data['_id'] not in rank_info_dict:
                    rank_info_dict[data['_id']] = {}
                for level in data['node_rank_info']:
                    if level not in rank_info_dict[data['_id']]:
                        rank_info_dict[data['_id']][level] = []
                    rank_info_dict[data['_id']][level].extend(data['node_rank_info'][level])
            pbar.update(1)
            input_file.close()
    return rank_info_dict

def _shrink_to_top_k(rank_info_dict):
    vectors = {}
    for _id in rank_info_dict:
        for level in rank_info_dict[_id]:
            rank_info_dict[_id][level].sort(key= lambda x: x['similarity'], reverse=True)
            rank_info_dict[_id][level] = rank_info_dict[_id][level][: retriever_kwargs['similarity_top_k']]
            for rank_info_of_one in rank_info_dict[_id][level]:
                if rank_info_of_one['db_path'] not in vectors:
                    chroma_client = chromadb.PersistentClient(path=rank_info_of_one['db_path'])
                    chroma_collection = chroma_client.get_collection(name='quickstart')
                    vectors[rank_info_of_one['db_path']] = ChromaVectorStore(chroma_collection=chroma_collection)
    return vectors

def get_rank_info_dict_and_vectors(prefix, cache_dir='./.cache'):
    
    filenames = [filename for filename in os.listdir(cache_dir) if prefix in filename]
    filenames.sort(key=lambda x: int(x.split('_')[-2]))
    # merge top k results from different files
    rank_info_dict = _merge_rank_info_from_different_files(filenames, cache_dir)

    # Select top k results
    vectors = _shrink_to_top_k(rank_info_dict)
    
    return rank_info_dict, vectors
     
def _generate_retrieved_contexts(qas, rank_info_list_generator, save_path, vectors):
    with open(save_path, 'w') as save_file:
        with tqdm(total=len(qas), desc="retrieving nodes ...") as pbar:
            for qa in qas:
                rank_info_list = rank_info_list_generator(qa['_id'])
                retrieved_nodes_id = []
                retrieved_contexts = []
                similarity = 0
                # Go over all groups
                for rank_infos in rank_info_list:
                    one_group_nodes = []
                    # Go over rank info for each node in one group
                    for rank_info in rank_infos:
                        one_group_nodes.extend(vectors[rank_info['db_path']].get_nodes(node_ids=[rank_info['id']]))
                        similarity += rank_info['similarity']
                    retrieved_nodes_id.append([n.id_ for n in one_group_nodes])
                    retrieved_contexts.append([n.text for n in one_group_nodes])
                    
                data = {
                    '_id': qa['_id'],
                    'retrieved_nodes_id': retrieved_nodes_id,
                    'retrieved_contexts': retrieved_contexts,
                    'similarity': similarity
                }
                save_file.write(json.dumps(data) + "\n")
                pbar.update(1)

def get_topk_for_topp(rank_info_list, p_threshold):
    similarities = np.array([
        np.array([rank_info['similarity'] for rank_info in rank_info_raw])
        for rank_info_raw in zip(*rank_info_list)
    ])
    
    flattened = similarities.flatten()
    softmax_flattened = np.exp(flattened) / np.sum(np.exp(flattened))
    probabilities = softmax_flattened.reshape(similarities.shape)
    
    total_p = 0
    top_k = 0
    for probability_row in probabilities:
        for p in probability_row:
            total_p += p
        if total_p > p_threshold:
            return top_k
        top_k += 1

    return top_k

def generate_contexts(
    hotpot_question_path: str, 
    retrieved_contexts_save_path: str,
    retrieve_mode: str,
    retriever_kwargs: dict,
    cache_prefix: str,
    retrieved_cache_dir = './cache'
):
    load_configs()
    qas = []
    with open(hotpot_question_path,'r') as question_file:
        for line in question_file:
            qa = json.loads(line)
            qas.append(qa)    
    rank_info_dict, vectors = get_rank_info_dict_and_vectors(prefix=cache_prefix, cache_dir=retrieved_cache_dir)
    
    if retrieve_mode == 'one':
        def retriever_nodes_list_generator(_id):
            rank_info_for_one = []
            for level in rank_info_dict[_id]:
                rank_info_for_one.extend(rank_info_dict[_id][level])
            rank_info_for_one.sort(key= lambda x: x['similarity'], reverse=True)
            # Select top k
            rank_info_for_one = rank_info_for_one[:retriever_kwargs['similarity_top_k']]
            return [rank_info_for_one]
        
    elif retrieve_mode == 'one_TopP':
        def retriever_nodes_list_generator(_id):
            rank_info_for_one = []
            for level in rank_info_dict[_id]:
                rank_info_for_one.extend(rank_info_dict[_id][level])
            rank_info_for_one.sort(key= lambda x: x['similarity'], reverse=True)
            # Get rank info list
            rank_info_list = [rank_info_for_one]
            rank_info_list = [rank_info_one_level[:retriever_kwargs['similarity_top_k']] for rank_info_one_level in rank_info_list]
            # Select top k
            top_k = get_topk_for_topp(rank_info_list, retriever_kwargs['probability_threshold'])
            rank_info_for_one = rank_info_for_one[:top_k]
            
            return [rank_info_for_one]
        
    _generate_retrieved_contexts(qas, retriever_nodes_list_generator, retrieved_contexts_save_path, vectors)

def check_nonstart_cache_file_with_level(index_dir, save_dir, notIncludeNotFinishCache):
    index_ids = {}
    for filename in os.listdir(os.path.abspath(f'../../database/{index_dir}')):
        match = re.search(r'(\d+)_([\w-]+)_chroma', filename)
        if match:
            if match.group(1) not in index_ids:
                index_ids[match.group(1)] = set()
            index_ids[match.group(1)].add(match.group(2))
    # calculate required ids
    index_ids = sorted(
        (k for k, v in index_ids.items() if len(v) == 4),
        key=lambda x: int(x)
    )
    
    # remove already generated cache id
    for filename in os.listdir(os.path.abspath(save_dir)):
        pattern = index_dir + r'_(\d+)\_chroma.jsonl'
        match = re.search(pattern, filename)
        if match:
            task_id = match.group(1)
            index_ids.remove(task_id)
        if notIncludeNotFinishCache:
            pattern = index_dir+ r'_(\d+)\_chroma_not_finish.jsonl'
            match = re.search(pattern, filename)
            if match:
                task_id = match.group(1)
                index_ids.remove(task_id)
    print(f'leave cache task: {len(index_ids)}')
    return index_ids
    
def check_nonstart_cache_file_without_level(index_dir, save_dir, notIncludeNotFinishCache):
    index_ids = []
    for filename in os.listdir(os.path.abspath(f'../../database/{index_dir}')):
        match = re.search(r'(\d+)\_chroma', filename)
        if match:
            index_ids.append(match.group(1))
    # calculate required ids
    index_ids.sort(key=lambda x: int(x))
    
    # remove already generated cache id
    for filename in os.listdir(os.path.abspath(save_dir)):
        pattern = index_dir + r'_(\d+)\_chroma.jsonl'
        match = re.search(pattern, filename)
        if match:
            task_id = match.group(1)
            index_ids.remove(task_id)
        if notIncludeNotFinishCache:
            pattern = index_dir + r'_(\d+)\_chroma_not_finish.jsonl'
            match = re.search(pattern, filename)
            if match:
                task_id = match.group(1)
                index_ids.remove(task_id)
    print(f'leave cache task: {len(index_ids)}')
    return index_ids

if __name__ == "__main__":
    args = load_args()
    
    hotpot_question_path = os.path.abspath('../step_2_get_embedding_value/questions/hotpot_questions.jsonl')
    save_dir = './.cache'
    notIncludeNotFinishCache = False # modify each time
    retriever_kwargs = {
        'similarity_top_k': int(args.top_k) if args.top_k else args.top_k,
        "probability_threshold": 0.5
    }
    
    if args.action == 'main':
        configs = [ # modify each time
            # Top K
            # ['wikipedia-mal-rag', 'one', '10', False],
            ['wikipedia-mal-rag', 'one_TopP', '10', False],
        ]
        
        for (index_dir, retrieved_mode, top_k, need_level) in configs:
            if need_level:
                index_ids = check_nonstart_cache_file_with_level(index_dir, save_dir, notIncludeNotFinishCache)
            else:
                index_ids = check_nonstart_cache_file_without_level(index_dir, save_dir, notIncludeNotFinishCache)
            
            if len(index_ids) > 0:
                for index_id in index_ids:
                    submit_job(
                        script_path=os.getcwd(),
                        python_file_name='generate_hotpot_retrieved_contexts.py',
                        index_id=index_id,
                        index_dir=index_dir,
                        retrieved_mode=retrieved_mode,
                        top_k=top_k,
                        action='thread_cache',
                        need_level=need_level
                    )
            else:
                print('start final retrieve')
                submit_job(
                    script_path=os.getcwd(),
                    python_file_name='generate_hotpot_retrieved_contexts.py',
                    index_id='total',
                    index_dir=index_dir,
                    retrieved_mode=retrieved_mode,
                    top_k=top_k,
                    action='thread_retrieve',
                    need_level=need_level
                )
            
    elif args.action == 'thread_cache':
        index_id = args.index_id
        index_dir_path = os.path.abspath(f'../../database/{args.index_dir}')
        retrieved_mode = args.retrieved_mode

        generate_retrieved_hotpot_cache(
            hotpot_question_path=hotpot_question_path,
            database_dir=index_dir_path,
            chroma_db_name=f'{args.index_id}_chroma', # if level is None else f'{index_id}_{level}_chroma'
            cache_prefix=f'{args.index_dir}', # if level else f'{args.index_dir}_{retrieved_mode}',
            save_dir=save_dir,
            need_level=args.need_level
        )
    
    elif args.action == 'thread_retrieve':
        index_dir = args.index_dir
        retrieved_mode = args.retrieved_mode
        retrieved_contexts_save_path = f'./retrieved_contexts/{index_dir}_{retrieved_mode}_retrieved_contexts.jsonl'
        
        generate_contexts(
            hotpot_question_path=hotpot_question_path, 
            retrieved_contexts_save_path=retrieved_contexts_save_path,
            retrieve_mode=retrieved_mode,
            retriever_kwargs=retriever_kwargs,
            cache_prefix=f'{args.index_dir}',
            retrieved_cache_dir=save_dir
        )
