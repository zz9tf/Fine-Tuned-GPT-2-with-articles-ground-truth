import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import argparse
import json
from tqdm import tqdm
# from component.index.custom_retriever import CustomRetriever
from configs.load_config import load_configs
from llama_index.core.schema import QueryBundle
from component.io import load_nodes_jsonl
from component.index.index import get_chroma_retriever_from_nodes
from utils.generate_and_execute_slurm_job import generate_and_execute_slurm_job
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--action', type=str, default='main', help='The action to generate retrieve contexts')
    parser.add_argument('--prefix', type=str, default=None, help='The prefix of retrieve contexts storage file')
    parser.add_argument('--index_id', type=str, default=None, help='The index id')
    parser.add_argument('--index_dir', type=str, default=None, help='The index id dir name')
    parser.add_argument('--retrieved_mode', type=str, default=None, help='The retrieved_mode to be used in retrieve')
    parser.add_argument('--top_k', type=str, default=None, help='The top_k similarities to be retrieved')

    return parser.parse_args()

def submit_job(
    script_path: str,
    python_file_name: str,
    prefix: str,
    index_id: str,
    index_dir: str,
    retrieved_mode: str,
    top_k: str    
):
    """Submit a job to Slurm and return the job ID."""
    job_name = f'retri_con_{prefix}'
    log_file_path = os.path.abspath(os.path.join(script_path, 'out/{job_name}.out'))
    script_path = os.path.abspath(os.path.join(script_path, 'execute/execute.sh'))
    job_name = generate_and_execute_slurm_job(
            python_start_script=f"{python_file_name} --action thread --prefix {prefix} --index_id {index_id} --index_dir {index_dir} --retrieved_mode {retrieved_mode} --top_k {top_k}",
            job_name=job_name,
            num=20,
            log_file_path=log_file_path,
            script_path=script_path
        )
    
    print(f"[Job: {job_name}] is submitted!")
    return job_name

def generate_retrieved_contexts(question_nodes, retriever_selector, save_path):
    with open(save_path, 'w') as save_file:
        with tqdm(total=sum(len(node.metadata['questions_and_embeddings']) for node in question_nodes), desc="retrieving nodes ...") as pbar:
            for node in question_nodes:
                for i, (q, e) in enumerate(node.metadata['questions_and_embeddings'].items()):
                    retrieved_nodes = []
                    query = QueryBundle(query_str=q, embedding=e)
                    retrieves = retriever_selector(query.query_str)
                    for retriever in retrieves:
                        retrieved_nodes.extend(retriever.retrieve(query))
                        
                    data = {
                        'question_node_id': node.id_,
                        'question_id': i,
                        'retrieved_nodes_id': [n.id_ for n in retrieved_nodes],
                        'retrieved_contexts': [n.text for n in retrieved_nodes]
                    }
                    save_file.write(json.dumps(data) + "\n")
                    pbar.update(1)

def generate_retrieved_contexts_from_custom(question_nodes, retriever, save_path):
    queries = []
    for node in question_nodes:
        for q, e in node.metadata['questions_and_embeddings'].items():
            queries.append(QueryBundle(query_str=q, embedding=e))
    
    query_id_to_result = retriever.retrieve(queries)
    
    query_id = 0
    with open(save_path, 'w') as save_file:
        for node in tqdm(question_nodes, desc='Generating retrieved contexts...'):
            for q, e in node.metadata['questions_and_embeddings'].items():
                print([n.id_ for n in query_id_to_result[query_id].nodes])
                exit()
                data = {
                    'question_node_id': node.id_,
                    'question_id': query_id,
                    'retrieved_nodes_id': [n.id_ for n in query_id_to_result[query_id].nodes],
                    'retrieved_contexts': [n.text for n in query_id_to_result[query_id].nodes]
                }
                save_file.write(json.dumps(data) + "\n")
                query_id += 1

def generate_contexts(
    question_nodes_path: str, 
    retrieved_contexts_save_path: str, 
    index_id: str, 
    index_dir_path: str, 
    retriever_kwargs,
):
    load_configs()
    
    question_nodes = load_nodes_jsonl(question_nodes_path)
    file_path = os.path.join(index_dir_path, index_id) + '.jsonl'
    
    # retriever = CustomRetriever(
    #     file_path=file_path,
    #     **retriever_kwargs
    # )
    if retriever_kwargs['retrieve_mode'] == 'one':
        # retriever = CustomRetriever(
        #     file_path=file_path,
        #     retrieve_mode=retrieve_mode,
        #     **retriever_kwargs
        # )
        retrieve = get_chroma_retriever_from_nodes(
            index_dir_path,
            index_id,
            retriever_kwargs=retriever_kwargs,
            break_num=retriever_kwargs['break_num']
        )
        retriever_selector = lambda _: [retrieve]
    # elif retrieve_mode == 'all-level':
    #     level_to_retriever = CustomRetriever(
    #         file_path=file_path,
    #         **retriever_kwargs
    #     )
    #     retrievers = list(level_to_retriever.values())
    #     retriever_selector = lambda _: retrievers
    # elif retrieved_mode == 'with_predictor':
    #     level_to_retriever = CustomRetriever.from_nodes_file_with_all_levels(
    #         index_dir_path=index_dir_path, 
    #         index_id=index_id,
    #         retriever_kwargs=retriever_kwargs,
    #         break_num=break_num,
    #         worker=2
    #     )
    #     model_path = os.path.abspath('../step_3_level_predictor/results/checkpoint-945')
    #     model = AutoModelForSequenceClassification.from_pretrained(model_path)
    #     tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    #     def retriever_selector(query):
    #         inputs = tokenizer(query.query_str, return_tensors="pt", truncation=True)
            
    #         # Put the model in evaluation mode
    #         model.eval()

    #         # Get the prediction
    #         with torch.no_grad():
    #             outputs = model(**inputs)
    #             logits = outputs.logits
    #             predicted_class = torch.argmax(logits, dim=-1).item()
    #         num_to_label = {i:label for i, label in enumerate(['document', 'section', 'paragraph', 'multi-sentences'])}
    #         return [level_to_retriever[num_to_label[predicted_class]]]
        
    generate_retrieved_contexts(question_nodes, retriever_selector, retrieved_contexts_save_path)

if __name__ == "__main__":
    args = load_args()
    
    question_nodes_path = os.path.abspath('../step_1_get_embedding_value/questions/gpt-4o-batch-all-p_pid_0.jsonl')
    
    if args.action == 'main':
        configs = [
            ['multi_level', 'all', 'gpt-4o-batch-all-target', 'one', '5'],
            # ['multi_level_all_levels', 'all', 'gpt-4o-batch-all-target', 'all-level', '2'],
            # ['multi_level_with_predictor', 'all', 'gpt-4o-batch-all-target', 'with_predictor', '5'],
            # ['sentence_splitter', 'all', 'sentence-splitter-rag', 'one', '5']
        ]
        
        for (prefix, index_id, index_dir, retrieved_mode, top_k) in configs:
            submit_job(
                script_path=os.getcwd(),
                python_file_name='generate_retrieved_contexts.py',
                prefix=prefix,
                index_id=index_id,
                index_dir=index_dir,
                retrieved_mode=retrieved_mode,
                top_k=top_k
            )
        
    elif args.action == 'thread':
        prefix = args.prefix
        index_id = args.index_id
        index_dir_path = os.path.abspath(f'../../database/{args.index_dir}')
        retrieved_mode = args.retrieved_mode
        retrieved_contexts_save_path = f'./retrieved_contexts/{prefix}_retrieved_contexts.jsonl'
        
        retriever_kwargs = {
            'similarity_top_k': int(args.top_k),
            'mode': 'default',
            'break_num': 100000, # 400000
            'batch_size': None, # 200000
            'retrieve_mode': retrieved_mode,
            'worker': 5
            # 'worker': None
        }
        
        generate_contexts(
            question_nodes_path, 
            retrieved_contexts_save_path, 
            index_id, 
            index_dir_path,
            retriever_kwargs=retriever_kwargs
        )
