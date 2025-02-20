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
    # print("searching " + level)
    results = retrievers[level]['retriever']._vector_store.query(query)
    # print(len(list(set(node.id_ for node in results.nodes))))
    # print(sum([len(node.text) for node in results.nodes]))
    
    return [
        {
            'similarity': s, 
            'id': node_id, 
            'db_path': retrievers[level]['db_path']
        }
        for s, node_id in zip(results.similarities, results.ids)
    ]

def generate_retrieved_node_cache(
    question_nodes_path, database_dir, chroma_db_name, retriever_kwargs, cache_prefix, save_dir='./.cache', need_level=True
):
    levels = ['document', 'section', 'paragraph', 'multi-sentences']
    question_nodes = load_nodes_jsonl(question_nodes_path)
    retrievers = _get_retrievers(need_level, levels, chroma_db_name, database_dir)
    
    save_path = os.path.join(save_dir, f"{cache_prefix}_{chroma_db_name}_not_finish.jsonl")
    line_number = _count_start_line_number(save_path)
    
    # generate retrieved nodes
    with open(save_path, 'a') as save_file:
        with tqdm(total=sum(len(node.metadata['questions_and_embeddings']) for node in question_nodes), desc="retrieving nodes ...") as pbar:
            for node in question_nodes:
                for i, (q, e) in enumerate(node.metadata['questions_and_embeddings'].items()):
                    if i < line_number:
                        continue
                    # Update cache file with one question data
                    data = {
                            'question_node_id': node.id_,
                            'question_id': i,
                            'node_rank_info': {}
                    }
                    if need_level:
                        for level in levels:
                            data['node_rank_info'][level] = _generate_node_rank_info(q, e, retrievers, level)
                    else:
                        data['node_rank_info'][None] = _generate_node_rank_info(q, e, retrievers, None)
                    save_file.write(json.dumps(data) + "\n")
                    pbar.update(1)
    os.rename(save_path,  os.path.join(save_dir, f"{cache_prefix}_{chroma_db_name}.jsonl"))

def _merge_rank_info_from_different_files(filenames, cache_dir):
    rank_info_dict = {}
    
    with tqdm(total=len(filenames), desc='loading ...') as pbar:
        for filename in filenames:
            pbar.set_description_str(filename)
            input_file = open(os.path.join(cache_dir, filename), 'r')
            for line in input_file:
                data = json.loads(line)
                if data['question_node_id'] not in rank_info_dict:
                    rank_info_dict[data['question_node_id']] = {}
                if data['question_id'] not in rank_info_dict[data['question_node_id']]:
                    rank_info_dict[data['question_node_id']][data['question_id']] = {}
                for level in data['node_rank_info']:
                    if level not in rank_info_dict[data['question_node_id']][data['question_id']]:
                        rank_info_dict[data['question_node_id']][data['question_id']][level] = []
                    rank_info_dict[data['question_node_id']][data['question_id']][level].extend(data['node_rank_info'][level])
            pbar.update(1)
            input_file.close()
    return rank_info_dict

def _shrink_to_top_k(rank_info_dict):
    vectors = {}
    for question_node_id in rank_info_dict:
        for question_id in rank_info_dict[question_node_id]:
            for level in rank_info_dict[question_node_id][question_id]:
                rank_info_dict[question_node_id][question_id][level].sort(key= lambda x: x['similarity'], reverse=True)
                rank_info_dict[question_node_id][question_id][level] = rank_info_dict[question_node_id][question_id][level][: retriever_kwargs['similarity_top_k']]
                for rank_info_of_one in rank_info_dict[question_node_id][question_id][level]:
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
     
def _generate_retrieved_contexts(question_nodes, rank_info_list_generator, save_path, vectors):
    with open(save_path, 'w') as save_file:
        with tqdm(total=sum(len(node.metadata['questions_and_embeddings']) for node in question_nodes), desc="retrieving nodes ...") as pbar:
            for node in question_nodes:
                for question_id, (q, _) in enumerate(node.metadata['questions_and_embeddings'].items()):
                    rank_info_list = rank_info_list_generator(q, node.id_, question_id)
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
                        'question_node_id': node.id_,
                        'question_id': question_id,
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
    question_nodes_path: str, 
    retrieved_contexts_save_path: str,
    retrieve_mode: str,
    retriever_kwargs: dict,
    cache_prefix: str,
    retrieved_cache_dir = './cache'
):
    load_configs()
    question_nodes = load_nodes_jsonl(question_nodes_path)
    rank_info_dict, vectors = get_rank_info_dict_and_vectors(prefix=cache_prefix, cache_dir=retrieved_cache_dir)
    model_path = os.path.abspath(retriever_kwargs['model_path'])
    
    if retrieve_mode == 'one':
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            rank_info_for_one = []
            for level in rank_info_dict[question_node_id][question_id]:
                rank_info_for_one.extend(rank_info_dict[question_node_id][question_id][level])
            rank_info_for_one.sort(key= lambda x: x['similarity'], reverse=True)
            # Select top k
            rank_info_for_one = rank_info_for_one[:retriever_kwargs['similarity_top_k']]
            return [rank_info_for_one]
        
    elif retrieve_mode == 'document':
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            # All infos are already top k
            return [rank_info_dict[question_node_id][question_id]['document']]
    
    elif retrieve_mode == 'section':
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            # All infos are already top k
            return [rank_info_dict[question_node_id][question_id]['section']]
    
    elif retrieve_mode == 'paragraph':
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            # All infos are already top k
            return [rank_info_dict[question_node_id][question_id]['paragraph']]
    
    elif retrieve_mode == 'multi-sentences':
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            # All infos are already top k
            return [rank_info_dict[question_node_id][question_id]['multi-sentences']]
        
    elif retrieve_mode == 'all-level':
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            # All infos are already top k
            return [
                rank_info_dict[question_node_id][question_id][level] 
                    for level in rank_info_dict[question_node_id][question_id]
            ]
            
    elif retrieved_mode == 'predictor_top1':
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
            # All infos are already top k
            return [rank_info_dict[question_node_id][question_id][num_to_label[predicted_class]]]
        
    elif retrieved_mode == 'predictor_top2':
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
            # All infos are already top k
            return [rank_info_dict[question_node_id][question_id][num_to_label[index.item()]] for index in top_indices[0]]
        
    elif retrieve_mode == 'predictor_top3':
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
            # All infos are already top k
            return [rank_info_dict[question_node_id][question_id][num_to_label[index.item()]] for index in top_indices[0]]
    
    elif retrieved_mode == 'predictor_over25_percent':
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
                probabilities = F.softmax(logits, dim=-1)
                selected_indices = (probabilities > 0.25).nonzero(as_tuple=True)[0].tolist()
            num_to_label = {i:label for i, label in enumerate(['document', 'section', 'paragraph', 'multi-sentences'])}
            # All infos are already top k
            return [rank_info_dict[question_node_id][question_id][num_to_label[index]] for index in selected_indices]
        
    elif retrieved_mode == 'predictor_top2_depending_on_similarity':
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
            # All infos are already top k
            rank_info_in_one = []
            for index in top_indices[0]:
                label = num_to_label[index.item()]
                rank_info_in_one.extend(rank_info_dict[question_node_id][question_id][label])
                
            rank_info_in_one.sort(key= lambda x: x['similarity'], reverse=True)
            # Select top k
            rank_info_in_one = rank_info_in_one[:retriever_kwargs['similarity_top_k']]
            
            rank_info_list = [rank_info_in_one]
            
            return rank_info_list
        
    elif retrieve_mode == 'one_TopP':
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            rank_info_for_one = []
            for level in rank_info_dict[question_node_id][question_id]:
                rank_info_for_one.extend(rank_info_dict[question_node_id][question_id][level])
            rank_info_for_one.sort(key= lambda x: x['similarity'], reverse=True)
            # Get rank info list
            rank_info_list = [rank_info_for_one]
            rank_info_list = [rank_info_one_level[:retriever_kwargs['similarity_top_k']] for rank_info_one_level in rank_info_list]
            # Select top k
            top_k = get_topk_for_topp(rank_info_list, retriever_kwargs['probability_threshold'])
            rank_info_for_one = rank_info_for_one[:top_k]
            
            return [rank_info_for_one]
        
    elif retrieve_mode == 'document_TopP':
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            # All infos are already top k
            rank_info_list = [rank_info_dict[question_node_id][question_id]['document']]
            top_k = get_topk_for_topp(rank_info_list, retriever_kwargs['probability_threshold'])
            rank_info_list = [rank_info_one_level[:top_k] for rank_info_one_level in rank_info_list]
            
            return rank_info_list
    
    elif retrieve_mode == 'section_TopP':
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            # All infos are already top k
            rank_info_list = [rank_info_dict[question_node_id][question_id]['section']]
            top_k = get_topk_for_topp(rank_info_list, retriever_kwargs['probability_threshold'])
            rank_info_list = [rank_info_one_level[:top_k] for rank_info_one_level in rank_info_list]
            
            return rank_info_list
    
    elif retrieve_mode == 'paragraph_TopP':
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            # All infos are already top k
            rank_info_list = [rank_info_dict[question_node_id][question_id]['paragraph']]
            top_k = get_topk_for_topp(rank_info_list, retriever_kwargs['probability_threshold'])
            rank_info_list = [rank_info_one_level[:top_k] for rank_info_one_level in rank_info_list]
            
            return rank_info_list
    
    elif retrieve_mode == 'multi-sentences_TopP':
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            # All infos are already top k
            rank_info_list = [rank_info_dict[question_node_id][question_id]['multi-sentences']]
            top_k = get_topk_for_topp(rank_info_list, retriever_kwargs['probability_threshold'])
            rank_info_list = [rank_info_one_level[:top_k] for rank_info_one_level in rank_info_list]
            
            return rank_info_list
    
    elif retrieve_mode == 'all-level_TopP':
        def retriever_nodes_list_generator(query, question_node_id, question_id):
            # All infos are already top k
            rank_info_list = [
                rank_info_dict[question_node_id][question_id][level] 
                    for level in rank_info_dict[question_node_id][question_id]
            ]
            top_k = get_topk_for_topp(rank_info_list, retriever_kwargs['probability_threshold'])
            rank_info_list = [rank_info_one_level[:top_k] for rank_info_one_level in rank_info_list]
            
            return rank_info_list

    elif retrieved_mode == 'predictor_top1_TopP':
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
            # All infos are already top k
            rank_info_list = [rank_info_dict[question_node_id][question_id][num_to_label[predicted_class]]]
            top_k = get_topk_for_topp(rank_info_list, retriever_kwargs['probability_threshold'])
            rank_info_list = [rank_info_one_level[:top_k] for rank_info_one_level in rank_info_list]
            
            return rank_info_list
        
    elif retrieved_mode == 'predictor_top2_TopP':
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
            # All infos are already top k
            rank_info_list = [rank_info_dict[question_node_id][question_id][num_to_label[index.item()]] for index in top_indices[0]]
            top_k = get_topk_for_topp(rank_info_list, retriever_kwargs['probability_threshold'])
            rank_info_list = [rank_info_one_level[:top_k] for rank_info_one_level in rank_info_list]
            
            return rank_info_list
        
    elif retrieve_mode == 'predictor_top3_TopP':
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
            # All infos are already top k
            rank_info_list = [rank_info_dict[question_node_id][question_id][num_to_label[index.item()]] for index in top_indices[0]]
            top_k = get_topk_for_topp(rank_info_list, retriever_kwargs['probability_threshold'])
            rank_info_list = [rank_info_one_level[:top_k] for rank_info_one_level in rank_info_list]
            
            return rank_info_list
    
    elif retrieved_mode == 'predictor_over25_percent_TopP':
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
                probabilities = F.softmax(logits, dim=-1)
                selected_indices = (probabilities > 0.25).nonzero(as_tuple=True)[0].tolist()
            num_to_label = {i:label for i, label in enumerate(['document', 'section', 'paragraph', 'multi-sentences'])}
            # All infos are already top k
            rank_info_list = [rank_info_dict[question_node_id][question_id][num_to_label[index]] for index in selected_indices]
            top_k = get_topk_for_topp(rank_info_list, retriever_kwargs['probability_threshold'])
            rank_info_list = [rank_info_one_level[:top_k] for rank_info_one_level in rank_info_list]
            
            return rank_info_list
    
    # elif retrieved_mode == 'predictor_top2_depending_on_similarity':
    #     model = AutoModelForSequenceClassification.from_pretrained(model_path)
    #     tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    #     def retriever_nodes_list_generator(query, question_node_id, question_id):
    #         inputs = tokenizer(query, return_tensors="pt", truncation=True)
    #         # Put the model in evaluation mode
    #         model.eval()
    #         # Get the prediction
    #         with torch.no_grad():
    #             outputs = model(**inputs)
    #             logits = outputs.logits
    #             _, top_indices = torch.topk(logits, k=2, dim=-1)
    #         num_to_label = {i:label for i, label in enumerate(['document', 'section', 'paragraph', 'multi-sentences'])}
    #         # All infos are already top k
    #         rank_info_in_one = []
    #         for index in top_indices[0]:
    #             label = num_to_label[index.item()]
    #             rank_info_in_one.extend(rank_info_dict[question_node_id][question_id][label])
                
    #         rank_info_in_one.sort(key= lambda x: x['similarity'], reverse=True)
    #         # Select top k
    #         rank_info_in_one = rank_info_in_one[:retriever_kwargs['similarity_top_k']]
            
    #         rank_info_list = [rank_info_in_one]
            
    #         return rank_info_list
        
    _generate_retrieved_contexts(question_nodes, retriever_nodes_list_generator, retrieved_contexts_save_path, vectors)

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
    
    question_nodes_path = os.path.abspath('../step_1_get_embedding_value/questions/gpt-4o-batch-all-p_pid_0.jsonl')
    save_dir = './.cache'
    notIncludeNotFinishCache = False # modify each time
    retriever_kwargs = {
        # 'break_num': 100000, # 400000 not use
        # 'batch_size': None, # 200000 not use
        # 'worker': 5, # not use
        # "include": ["metadatas", "documents", "embeddings", "distances"], # not use
        'mode': 'default',
        'similarity_top_k': int(args.top_k) if args.top_k else args.top_k,
        "probability_threshold": 0.5,
        'model_path': '../step_3_level_predictor/scibert_scivocab_uncased_results/checkpoint-180'
        # 'worker': None
    }
    
    if args.action == 'main':
        configs = [ # modify each time
            # Top K
            # ['gpt-4o-batch-all-target', 'one', '10', True],
            # ['gpt-4o-batch-all-target', 'document', '10', True],
            # ['gpt-4o-batch-all-target', 'section', '10', True],
            # ['gpt-4o-batch-all-target', 'paragraph', '10', True],
            # ['gpt-4o-batch-all-target', 'multi-sentences', '10', True],
            # ['gpt-4o-batch-all-target', 'all-level', '10', True],            
            # ['gpt-4o-batch-all-target', 'predictor_top1', '10', True],
            ['gpt-4o-batch-all-target', 'predictor_top2', '10', True],
            # ['gpt-4o-batch-all-target', 'predictor_top3', '10', True],
            # ['gpt-4o-batch-all-target', 'predictor_over25_percent', '10', True],
            # Top P
            # ['gpt-4o-batch-all-target', 'one_TopP', '10', True],
            # ['gpt-4o-batch-all-target', 'document_TopP', '10', True],
            # ['gpt-4o-batch-all-target', 'section_TopP', '10', True],
            # ['gpt-4o-batch-all-target', 'paragraph_TopP', '10', True],
            # ['gpt-4o-batch-all-target', 'multi-sentences_TopP', '10', True],
            # ['gpt-4o-batch-all-target', 'all-level_TopP', '10', True],            
            # ['gpt-4o-batch-all-target', 'predictor_top1_TopP', '10', True],
            # ['gpt-4o-batch-all-target', 'predictor_top2_TopP', '10', True],
            # ['gpt-4o-batch-all-target', 'predictor_top3_TopP', '10', True],
            # ['gpt-4o-batch-all-target', 'predictor_over25_percent_TopP', '10', True],
            # Better strategies with predictor
            # ['gpt-4o-batch-all-target', 'predictor_top2_depending_on_similarity', '10', True],
            # Other dataset
            ['sentence-splitter-rag', 'one', '10', False]
        ]
        
        for (index_dir, retrieved_mode, top_k, need_level) in configs:
            # TODO count index id in database
            if need_level:
                index_ids = check_nonstart_cache_file_with_level(index_dir, save_dir, notIncludeNotFinishCache)
            else:
                index_ids = check_nonstart_cache_file_without_level(index_dir, save_dir, notIncludeNotFinishCache)

            # TODO: comment here
            # index_ids = index_ids[:4]
            
            if len(index_ids) > 0:
                for index_id in index_ids:
                    submit_job(
                        script_path=os.getcwd(),
                        python_file_name='generate_retrieved_contexts.py',
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
                    python_file_name='generate_retrieved_contexts.py',
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

        generate_retrieved_node_cache(
            question_nodes_path=question_nodes_path,
            database_dir=index_dir_path,
            chroma_db_name=f'{args.index_id}_chroma', # if level is None else f'{index_id}_{level}_chroma'
            retriever_kwargs=retriever_kwargs,
            cache_prefix=f'{args.index_dir}', # if level else f'{args.index_dir}_{retrieved_mode}',
            save_dir=save_dir,
            need_level=args.need_level
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
            cache_prefix=f'{args.index_dir}',
            retrieved_cache_dir=save_dir
        )