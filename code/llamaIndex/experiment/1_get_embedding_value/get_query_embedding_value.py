import os, sys
sys.path.insert(0, '/home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex')
import yaml
import json
import gc
import torch
import numpy as np
from tqdm import tqdm
from custom.embedding import get_embedding_model
from custom.io import load_nodes_jsonl
import pandas as pd

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
    nodeId2node ={node.id_: node for node in nodes}
    return nodes, nodeId2node

def load_node_questions(pid_num, cache_dir):
    file_name = f"gpt-4o-batch-all-p_2_parser_ManuallyHierarchicalNodeParser_8165_gpu_V100_nodeNum_200_{pid_num}_extract_langchain.csv"
    file_path = os.path.join(cache_dir, file_name)
    question_df = pd.read_csv(file_path)
    return question_df

# Function to get embeddings for nodes and save them to a file
def get_embeddings_and_save(embedding_model, nodeId2node, question_df, output_file_path):
    print("Getting embeddings from node questions")
    
    # Open the output file in write mode
    with open(output_file_path, 'w') as f_out:
        # Extract the node_id and objs columns
        node_objs_list = question_df[['node_id', 'objs']].values.tolist()  # Converts to list of lists (node_id, objs pairs)
        
        # Iterate through each node_id and its corresponding objs
        for node_id, objs in tqdm(node_objs_list, desc="Getting embeddings..."):
            objs = json.loads(objs)  # Parse the objs as a JSON object
            for obj in objs:
                question = obj['Question']  # Get the question field
                node = nodeId2node[node_id]  # Get the corresponding node
                
                # Get the embedding using the provided model
                with torch.no_grad():
                    embedding = embedding_model._get_text_embedding(question)
                
                # Clear CUDA cache to free memory
                torch.cuda.empty_cache()
                
                # Prepare the data dictionary for saving
                data = {
                    'id_': node_id,
                    'level': node.metadata['level'],
                    'embedding': embedding,  # Convert embedding to a list for JSON serialization
                }
                
                # Write the data to the output file in JSON Lines format
                f_out.write(json.dumps(data) + '\n')  # Write JSON line and add a newline

    print(f"Embeddings successfully saved to {output_file_path}")

# Main function to execute the process
def main():
    # Configurations
    config_path = "/home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/configs/prefix_config.yaml"
    cache_dir = "/home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/.save"
    embedding_model = load_embedding_model(config_path)
    
    for pid_num in ['pid_1']:
        output_file_path = f"./data/embeddings_question_{pid_num}.npy"
    
        # Load models and nodes
        _, nodeId2node = load_nodes(pid_num, cache_dir)
        question_df = load_node_questions(pid_num, cache_dir)
    
        # Get embeddings and save them
        get_embeddings_and_save(embedding_model, nodeId2node, question_df, output_file_path)

if __name__ == "__main__":
    main()
