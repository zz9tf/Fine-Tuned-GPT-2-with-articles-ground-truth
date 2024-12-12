import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import argparse
import json
from tqdm import tqdm
import re
from configs.load_config import load_configs
from llama_index.core.schema import QueryBundle
from component.io import load_nodes_jsonl
from component.index.index import get_chroma_retriever_from_storage, get_retriever_from_nodes, update_retriever
from utils.generate_and_execute_slurm_job import generate_and_execute_slurm_job
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sqlite3
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    VectorStoreQuery,
    MetadataFilter,
    FilterOperator,
    FilterCondition
)
from llama_index.core.schema import TextNode

def load_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--action', type=str, default='main', help='The action to generate retrieve contexts')
    parser.add_argument('--index_id', type=str, default=None, help='The index id')
    parser.add_argument('--index_dir', type=str, default=None, help='The index id dir name')
    parser.add_argument('--retrieved_mode', type=str, default=None, help='The retrieved_mode to be used in retrieve')
    parser.add_argument('--top_k', type=str, default=None, help='The top_k similarities to be retrieved')

    return parser.parse_args()

def submit_job(
    script_path: str,
    python_file_name: str,
    index_id: str,
    index_dir: str,
    retrieved_mode: str,
    top_k: str,
    action: str
):
    """Submit a job to Slurm and return the job ID."""
    job_name = f'retri_con_{retrieved_mode}_{index_dir}_{index_id}'
    log_file_path = os.path.abspath(os.path.join(script_path, 'out/{job_name}.out'))
    script_path = os.path.abspath(os.path.join(script_path, 'execute/execute.sh'))
    
    job_name = generate_and_execute_slurm_job(
            python_start_script=f"{python_file_name} --action {action} --index_id {index_id} --index_dir {index_dir} --retrieved_mode {retrieved_mode} --top_k {top_k}",
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

def generate_retrieved_node_cache(question_nodes_path, database_dir, chroma_db_name, retriever_kwargs, cache_prefix, save_dir='./.cache', needLevel=False):
    question_nodes = load_nodes_jsonl(question_nodes_path)
    levels = ['document', 'section', 'paragraph', 'multi-sentences']
    retriever = None
    retrievers = None
    if needLevel:
        retrievers = {}
        for level in levels:
            level_chroma_db_name = f"_{level}_".join(chroma_db_name.split('_'))
            retrievers[level] = get_chroma_retriever_from_storage(database_dir, level_chroma_db_name, retriever_kwargs)
    else:
        retriever = get_chroma_retriever_from_storage(database_dir, chroma_db_name, retriever_kwargs)
        
    save_path = os.path.join(save_dir, f"{cache_prefix}_{chroma_db_name}_not_finish.jsonl")
    
    line_number = 0
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        with open(save_path, 'r') as save_file:
            with tqdm(total=file_size, desc=f'Counting lines in {save_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for line in save_file:
                    pbar.update(len(line))
                    line_number += 1
    
    # generate retrieved nodes
    with open(save_path, 'a') as save_file:
        with tqdm(total=sum(len(node.metadata['questions_and_embeddings']) for node in question_nodes), desc="retrieving nodes ...") as pbar:
            for node in question_nodes:
                for i, (q, e) in enumerate(node.metadata['questions_and_embeddings'].items()):
                    if i <= line_number:
                        continue
                    query_bundle = QueryBundle(query_str=q, embedding=e)
                    if needLevel:
                        data = {
                            'question_node_id': node.id_,
                            'question_id': i,
                            'retrieved_nodes': {}
                        }
                        for level in levels:
                            query = VectorStoreQuery(
                                query_embedding=query_bundle.embedding,
                                similarity_top_k=retrievers[level]._similarity_top_k,
                                node_ids=retrievers[level]._node_ids,
                                doc_ids=retrievers[level]._doc_ids,
                                query_str=query_bundle.query_str,
                                mode=retrievers[level]._vector_store_query_mode,
                                alpha=retrievers[level]._alpha,
                                filters=None,
                                sparse_top_k=retrievers[level]._sparse_top_k,
                                hybrid_top_k=retrievers[level]._hybrid_top_k,
                            )
                            # print("searching " + level)
                            retrieved_nodes = retrievers[level]._vector_store.query(query, include=retriever_kwargs['include']).nodes
                            # print([node.id_ for node in retrieved_nodes])
                            data['retrieved_nodes'][level] = [node.to_dict() for node in retrieved_nodes]
                            # for l in data['retrieved_nodes'].keys():
                            #     print(l)
                            #     print([node['id_'] for node in data['retrieved_nodes'][l]])
                        save_file.write(json.dumps(data) + "\n")
                    else: # One
                        query = VectorStoreQuery(
                            query_embedding=query_bundle.embedding,
                            similarity_top_k=retriever._similarity_top_k,
                            node_ids=retriever._node_ids,
                            doc_ids=retriever._doc_ids,
                            query_str=query_bundle.query_str,
                            mode=retriever._vector_store_query_mode,
                            alpha=retriever._alpha,
                            filters=None,
                            sparse_top_k=retriever._sparse_top_k,
                            hybrid_top_k=retriever._hybrid_top_k,
                        )
                        retrieved_nodes = retriever._vector_store.query(query, include=retriever_kwargs['include']).nodes
                        data = {
                            'question_node_id': node.id_,
                            'question_id': i,
                            'retrieved_nodes': [node.to_dict() for node in retrieved_nodes]
                        }
                        save_file.write(json.dumps(data) + "\n")
                    pbar.update(1)
                    
    os.rename(save_path,  os.path.join(save_dir, f"{cache_prefix}_{chroma_db_name}.jsonl"))

def generate_cache_retrieved_nodes_dict(prefix, cache_dir='./.cache', needLevel=False):
    retrieved_nodes_dict = {}
    filenames = [filename for filename in os.listdir(cache_dir) if prefix in filename]
    filenames.sort(key=lambda x: int(x.split('_')[-2]))
    with tqdm(total=len(filenames), desc='loading ...') as pbar:
        for filename in filenames:
            pbar.set_description_str(filename)
            input_file = open(os.path.join(cache_dir, filename), 'r')
            
            # file_path = os.path.join(cache_dir, filename)
            # with open(file_path, 'r') as f:
            #     total_lines = sum(1 for _ in f)
            
            # with open(file_path, 'r') as input_file, tqdm(total=total_lines, desc=f"Processing {filename}", leave=False, unit='line') as line_pbar:
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
                # line_pbar.update(1)
            # print("Memory size of the dictionary (in Bytes):", sys.getsizeof(retrieved_nodes_dict))
            pbar.update(1)
            input_file.close()
    return retrieved_nodes_dict
     
def generate_retrieved_contexts(question_nodes, retriever, retriever_nodes_list_generator, save_path):
    with open(save_path, 'w') as save_file:
        with tqdm(total=sum(len(node.metadata['questions_and_embeddings']) for node in question_nodes), desc="retrieving nodes ...") as pbar:
            for node in question_nodes:
                for question_id, (q, e) in enumerate(node.metadata['questions_and_embeddings'].items()):
                    retrieved_nodes = []
                    query_bundle = QueryBundle(query_str=q, embedding=e)
                    retrieve_nodes_list = retriever_nodes_list_generator(query_bundle.query_str, node.id_, question_id)
                    for retrieve_nodes in retrieve_nodes_list:
                        update_retriever(retrieve_nodes, retriever)
                        query = retriever._build_vector_store_query(query_bundle)
                        result = retriever._vector_store.query(query, include=retriever_kwargs['include'])
                        retrieved_nodes.extend(result.nodes)
                        
                    data = {
                        'question_node_id': node.id_,
                        'question_id': question_id,
                        'retrieved_nodes_id': [n.id_ for n in retrieved_nodes],
                        'retrieved_contexts': [n.text for n in retrieved_nodes]
                    }
                    save_file.write(json.dumps(data) + "\n")
                    pbar.update(1)

def generate_contexts(
    question_nodes_path: str, 
    retrieved_contexts_save_path: str,
    retrieve_mode: str,
    retriever_kwargs: dict,
    cache_prefix: str,
    retrieved_cache_dir = './cache',
    needLevel=False
):
    load_configs()
    question_nodes = load_nodes_jsonl(question_nodes_path)
    # db_path = 'retrieved_nodes.db'
    # generate_cache_retrieved_nodes_sqlite(prefix=cache_prefix, cache_dir=retrieved_cache_dir, needLevel=needLevel, db_path=db_path)
    retrieved_nodes_dict = generate_cache_retrieved_nodes_dict(prefix=cache_prefix, cache_dir=retrieved_cache_dir, needLevel=needLevel)
    retriever = get_retriever_from_nodes(index_dir_path=None, index_id=None, nodes=[], retriever_kwargs=retriever_kwargs)
    
    # conn = sqlite3.connect(db_path)
    # cursor = conn.cursor()
    
    if retrieve_mode == 'one':
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            retrieved_nodes = retrieved_nodes_dict[question_node_id][question_id]
            # cursor.execute('''
            #     SELECT node_data FROM retrieved_nodes 
            #     WHERE question_node_id = ? AND question_id = ? AND level = NULL
            #     ''', (question_node_id, question_id))
            
            # # Fetch all nodes at this level and deserialize them
            # retrieved_nodes = [TextNode.from_dict(json.loads(row[0])) for row in cursor.fetchall()]

            return [retrieved_nodes]
    elif retrieve_mode == 'all-level':
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            retrieve_nodes_list = []
            for level in ['document', 'section', 'paragraph', 'multi-sentences']:
                retrieved_nodes = retrieved_nodes_dict[question_node_id][question_id][level]
                retrieve_nodes_list.append(retrieved_nodes)
            # retrieve_nodes_list = []
            # for level in ['document', 'section', 'paragraph', 'multi-sentences']:
            #     cursor.execute('''
            #         SELECT node_data FROM retrieved_nodes 
            #         WHERE question_node_id = ? AND question_id = ? AND level = ?
            #     ''', (question_node_id, question_id, level))
                
            #     # Fetch all nodes at this level and deserialize them
            #     retrieved_nodes = [TextNode.from_dict(json.loads(row[0])) for row in cursor.fetchall()]
            #     retrieve_nodes_list.append(retrieved_nodes)
            return retrieve_nodes_list
    elif retrieved_mode == 'with_predictor':
        model_path = os.path.abspath('../step_3_level_predictor/SciFive-base-PMC_results/checkpoint-750')
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            inputs = tokenizer(query, return_tensors="pt", truncation=True)
            # Put the model in evaluation mode
            model.eval()
            # Get the prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()
            num_to_label = {i:label for i, label in enumerate(['document', 'section', 'paragraph', 'multi-sentences'])}
            return [retrieved_nodes_dict[question_node_id][question_id][num_to_label[predicted_class]]]
        
    elif retrieved_mode == 'top2_predictor':
        model_path = os.path.abspath('../step_3_level_predictor/SciFive-base-PMC_results/checkpoint-750')
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            inputs = tokenizer(query, return_tensors="pt", truncation=True)
            # Put the model in evaluation mode
            model.eval()
            # Get the prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                _, top_indices = torch.topk(logits, k=2, dim=-1)
            num_to_label = {i:label for i, label in enumerate(['document', 'section', 'paragraph', 'multi-sentences'])}
            return [retrieved_nodes_dict[question_node_id][question_id][num_to_label[index]] for index in top_indices]
        
    elif retrieve_mode == 'top3_predictor':
        model_path = os.path.abspath('../step_3_level_predictor/SciFive-base-PMC_results/checkpoint-750')
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            inputs = tokenizer(query, return_tensors="pt", truncation=True)
            # Put the model in evaluation mode
            model.eval()
            # Get the prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                _, top_indices = torch.topk(logits, k=3, dim=-1)
            num_to_label = {i:label for i, label in enumerate(['document', 'section', 'paragraph', 'multi-sentences'])}
            return [retrieved_nodes_dict[question_node_id][question_id][num_to_label[index]] for index in top_indices]
    
    elif retrieved_mode == 'top2_predictor':
        model_path = os.path.abspath('../step_3_level_predictor/SciFive-base-PMC_results/checkpoint-750')
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            inputs = tokenizer(query, return_tensors="pt", truncation=True)
            # Put the model in evaluation mode
            model.eval()
            # Get the prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                _, top_indices = torch.topk(logits, k=2, dim=-1)
            num_to_label = {i:label for i, label in enumerate(['document', 'section', 'paragraph', 'multi-sentences'])}
            return [retrieved_nodes_dict[question_node_id][question_id][num_to_label[index]] for index in top_indices]
        
    generate_retrieved_contexts(question_nodes, retriever, retriever_nodes_list_generator, retrieved_contexts_save_path)
    # conn.close()

if __name__ == "__main__":
    args = load_args()
    
    question_nodes_path = os.path.abspath('../step_1_get_embedding_value/questions/gpt-4o-batch-all-p_pid_0.jsonl')
    save_dir = './.cache'
    notIncludeNotFinishCache = False # modify each time
    retriever_kwargs = {
        'similarity_top_k': int(args.top_k) if args.top_k else args.top_k,
        'mode': 'default',
        'break_num': 100000, # 400000
        'batch_size': None, # 200000
        'worker': 5,
        "include": ["metadatas", "documents", "embeddings", "distances"]
        # 'worker': None
    }
    
    if args.action == 'main':
        configs = [ # modify each time
            # ['gpt-4o-batch-all-target', 'one', '50'],
            ['gpt-4o-batch-all-target', 'all-level', '25'],
            # ['gpt-4o-batch-all-target', 'with_predictor', '25'],
            # ['gpt-4o-batch-all-target', 'top2_predictor', '15'],
            # ['gpt-4o-batch-all-target', 'top3_predictor', '10'],
            # ['gpt-4o-batch-all-target', '30rOver50_prediction', '30'],
            # ['sentence-splitter-rag', 'one', '10']
        ]
        
        for (index_dir, retrieved_mode, top_k) in configs:
            # TODO count index id in database
            if retrieved_mode == 'one':
                index_ids = []
                for filename in os.listdir(os.path.abspath(f'../../database/{index_dir}')):
                    match = re.search(r'(\d+)\_chroma', filename)
                    if match:
                        index_ids.append(match.group(1))
                        
                index_ids.sort(key=lambda x: int(x))
                
                for filename in os.listdir(os.path.abspath(save_dir)):
                    pattern = f'{index_dir}_{retrieved_mode}' + r'_(\d+)\_chroma.jsonl'
                    match = re.search(pattern, filename)
                    if match:
                        task_id = match.group(1)
                        index_ids.remove(task_id)
                    if notIncludeNotFinishCache:
                        pattern = f'{index_dir}_{retrieved_mode}' + r'_(\d+)\_chroma_not_finish.jsonl'
                        match = re.search(pattern, filename)
                        if match:
                            task_id = match.group(1)
                            index_ids.remove(task_id)
                print(f'leave cache task: {len(index_ids)}')
            else:
                index_ids = {}
                for filename in os.listdir(os.path.abspath(f'../../database/{index_dir}')):
                    match = re.search(r'(\d+)_([\w-]+)_chroma', filename)
                    if match:
                        if match.group(1) not in index_ids:
                            index_ids[match.group(1)] = set()
                        index_ids[match.group(1)].add(match.group(2))
                index_ids = sorted(
                    (k for k, v in index_ids.items() if len(v) == 4),
                    key=lambda x: int(x)
                )
                
                for filename in os.listdir(os.path.abspath(save_dir)):
                    pattern = f'{index_dir}_{retrieved_mode}' + r'_(\d+)\_chroma.jsonl'
                    match = re.search(pattern, filename)
                    if match:
                        task_id = match.group(1)
                        index_ids.remove(task_id)
                    if notIncludeNotFinishCache:
                        pattern = f'{index_dir}_{retrieved_mode}' + r'_(\d+)\_chroma_not_finish.jsonl'
                        match = re.search(pattern, filename)
                        if match:
                            task_id = match.group(1)
                            index_ids.remove(task_id)
                print(f'leave cache task: {len(index_ids)}')

            # TODO: comment here
            # index_ids = index_ids[:1]
            
            if len(index_ids) > 0:
                for index_id in index_ids:
                    submit_job(
                        script_path=os.getcwd(),
                        python_file_name='generate_retrieved_contexts.py',
                        index_id=index_id,
                        index_dir=index_dir,
                        retrieved_mode=retrieved_mode,
                        top_k=top_k,
                        action='thread_cache'
                    )
            else:
                print('start final retrieve')
                submit_job(
                    script_path=os.getcwd(),
                    python_file_name='generate_retrieved_contexts.py',
                    index_id='total',
                    index_dir=index_dir,
                    retrieved_mode=retrieved_mode,
                    top_k=top_k,
                    action='thread_retrieve'
                )
            
    elif args.action == 'thread_cache':
        index_id = args.index_id
        index_dir_path = os.path.abspath(f'../../database/{args.index_dir}')
        retrieved_mode = args.retrieved_mode
        generate_retrieved_node_cache(
            question_nodes_path=question_nodes_path,
            database_dir=index_dir_path,
            chroma_db_name=f'{args.index_id}_chroma', # if level is None else f'{index_id}_{level}_chroma'
            retriever_kwargs=retriever_kwargs,
            cache_prefix=f'{args.index_dir}_{retrieved_mode}', # if level else f'{args.index_dir}_{retrieved_mode}',
            save_dir=save_dir,
            needLevel=retrieved_mode!='one'
        )
    
    elif args.action == 'thread_retrieve':
        index_dir = args.index_dir
        retrieved_mode = args.retrieved_mode
        retrieved_contexts_save_path = f'./retrieved_contexts/{index_dir}_{retrieved_mode}_retrieved_contexts.jsonl'
        
        generate_contexts(
            question_nodes_path=question_nodes_path, 
            retrieved_contexts_save_path=retrieved_contexts_save_path,
            retrieve_mode=retrieved_mode,
            retriever_kwargs=retriever_kwargs,
            cache_prefix=f'{args.index_dir}_{retrieved_mode}',
            retrieved_cache_dir=save_dir,
            needLevel=retrieved_mode!='one'
        )
