import os, sys
sys.path.insert(0, '/home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex')
import yaml
import gc
import torch
import numpy as np
from tqdm import tqdm
from custom.embedding import get_embedding_model
from custom.io import load_nodes_jsonl

# Function to load the embedding model
def load_embedding_model(config_path):
    print("Loading embedding model")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    embed_config = config['embedding_model']['Linq-AI-Research/Linq-Embed-Mistral']
    embedding_model = get_embedding_model(embed_config)
    return embedding_model

# Function to load nodes
def load_nodes(pid_num, cache_dir):
    print("Loading nodes")
    file_name = f"gpt-4o-batch-all-p_2_parser_ManuallyHierarchicalNodeParser_8165_gpu_V100_nodeNum_200_{pid_num}.jsonl"
    file_path = os.path.join(cache_dir, file_name)
    nodes = load_nodes_jsonl(file_path)
    return nodes

# Function to get embeddings for nodes and save them to a file
def get_embeddings_and_save(embedding_model, nodes, output_file):
    print("Getting embeddings from nodes")
    embeddings = []
    for node in tqdm(nodes, desc="get embeddings ..."):
        text = node.get_content()
        with torch.no_grad():
            embedding = embedding_model._get_text_embedding(text)
        torch.cuda.empty_cache()
        gc.collect()
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    np.save(output_file, embeddings)

# Main function to execute the process
def main():
    # Configurations
    config_path = "/home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/configs/prefix_config.yaml"
    cache_dir = "/home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/.save"
    for pid_num in ['pid_14']:
        output_file = f"embeddings_{pid_num}.npy"
    
        # Load models and nodes
        embedding_model = load_embedding_model(config_path)
        nodes = load_nodes(pid_num, cache_dir)
    
        # Get embeddings and save them
        get_embeddings_and_save(embedding_model, nodes, output_file)

if __name__ == "__main__":
    main()
