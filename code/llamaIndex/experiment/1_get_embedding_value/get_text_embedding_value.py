import os, sys
sys.path.insert(0, os.path.abspath('../..'))
import yaml
import gc
import torch
from tqdm import tqdm
import json
from custom.embedding import get_embedding_model
from custom.io import load_nodes_jsonl
from utils.generate_and_execute_slurm_job import generate_and_execute_slurm_job
import argparse

def load_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--output_file_path', type=str, default=None)
    parser.add_argument('--action', type=str, default='main')

    return parser.parse_args()

# Function to load the embedding model
def load_embedding_model():
    print("Loading embedding model")
    config_path = os.path.abspath('../../configs/prefix_config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    embed_config = config['embedding_model']['Linq-AI-Research/Linq-Embed-Mistral']
    embedding_model = get_embedding_model(embed_config)
    return embedding_model

# Function to load nodes
def load_nodes(file_name, cache_dir):
    print("Loading nodes")
    file_path = os.path.join(cache_dir, file_name)
    nodes = load_nodes_jsonl(file_path)
    nodeId2node ={node.id_: node for node in nodes}
    return nodes, nodeId2node

def get_text_embeddings_and_save(embedding_model, nodes, output_file_path):
    print("Getting embeddings from nodes")
    
    # Open the output file in write mode
    with open(output_file_path, 'w') as f_out:
        # Extract the node_id and objs columns
        for node in tqdm(nodes, desc="get embeddings ..."):
            text = node.get_content()
            with torch.no_grad():
                embedding = embedding_model._get_text_embedding(text)
            torch.cuda.empty_cache()
            gc.collect()
            # Prepare the data dictionary for saving
            data = {
                'id_': node.id_,
                'level': node.metadata['level'],
                'embedding': embedding,  # Convert embedding to a list for JSON serialization
            }
                
            # Write the data to the output file in JSON Lines format
            f_out.write(json.dumps(data) + '\n')  # Write JSON line and add a newline

    print(f"Embeddings successfully saved to {output_file_path}")

def text_embedding_pipeline(file_name, cache_dir, output_file_path):
    # Load models and nodes
    embedding_model = load_embedding_model()
    nodes, _ = load_nodes(file_name, cache_dir)
    # Get embeddings and save them
    get_text_embeddings_and_save(embedding_model, nodes, output_file_path)

def merge():
    pass

# Main function to execute the process
def main():
    args = load_args()
    if args.action == 'main':
        file_name_prefix = "gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_50_pid_"
        cache_path = os.path.abspath("../../.cache")
        
        exist_context_embeddings = {file_name[len('embeddings_'):].split('.')[0] for file_name in os.listdir('./contexts')}
        leave_files = []
        for file_name in os.listdir(cache_path):
            file_name_before_point = file_name.split('.')[0]
            if file_name.startswith(file_name_prefix) and\
                file_name_before_point not in exist_context_embeddings:
                leave_files.append(file_name)
                
        leave_files = sorted(leave_files, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        print(f"Leave files: {len(leave_files)}")
        
        for i in range(min(len(leave_files), 40)): # 
            input_file = leave_files[i]
            output_file_path = f"./contexts/embeddings_{input_file.split('.')[0]}.jsonl"
            if i < 30:
                generate_and_execute_slurm_job(
                    f"get_text_embedding_value.py --input_file {input_file} --cache_dir {cache_dir} --output_file_path {output_file_path} --action submit_job",
                    job_name=f"embedding-{i}",
                    gpu="V100",
                    num="1"
                )
            else:
                generate_and_execute_slurm_job(
                    f"get_text_embedding_value.py --input_file {input_file} --cache_dir {cache_dir} --output_file_path {output_file_path} --action submit_job",
                    account="guest",
                    partition="guest-gpu",
                    job_name=f"embedding-{i}",
                    qos="low-gpu",
                    time="24:00:00",
                    gpu="V100",
                    num="1"
                )
    elif args.action == 'submit_job':
        file_name = args.input_file
        cache_dir = args.cache_dir
        output_file_path = args.output_file_path
        text_embedding_pipeline(file_name, cache_dir, output_file_path)   
    elif args.action == 'merge':
        merge()

if __name__ == "__main__":
    main()
