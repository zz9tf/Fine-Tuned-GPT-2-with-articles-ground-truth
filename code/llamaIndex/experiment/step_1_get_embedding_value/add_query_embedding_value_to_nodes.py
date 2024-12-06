import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from configs.load_config import load_configs
from component.models.embed.get_embedding_model import get_embedding_model
from component.io import load_nodes_jsonl
import pandas as pd
import torch
from tqdm import tqdm
import json

def get_queries_embeddings_and_save(embedding_model, id_2_node, question_df, output_file_path):
    print("Getting embeddings from node questions")
    
    # Open the output file in write mode
    with open(output_file_path, 'w') as f_out:
        # Extract the node_id and objs columns
        node_objs_list = question_df[['node_id', 'objs']].values.tolist()  # Converts to list of lists (node_id, objs pairs)
        
        # Iterate through each node_id and its corresponding objs
        for node_id, objs in tqdm(node_objs_list, desc="Getting embeddings..."):
            objs = json.loads(objs)  # Parse the objs as a JSON object
            if len(objs) == 0: continue
            
            node = id_2_node[node_id]  # Get the corresponding node
            node.metadata['questions_and_embeddings'] = {}
            for obj in objs:
                question = obj['Question']  # Get the question field
                # Get the embedding using the provided model
                with torch.no_grad():
                    embedding = embedding_model._get_text_embedding(question)
                node.metadata['questions_and_embeddings'][question] = embedding
            
            # Write the data to the output file in JSON Lines format            
            f_out.write(json.dumps(node.to_dict()) + '\n')  # Write JSON line and add a newline

    print(f"Embeddings successfully saved to {output_file_path}")

def add_query_embedding_values(nodes_file_path: str, question_csv_file_path: str):
    # Configurations
    _, prefix_config = load_configs()
    # node file
    nodes = load_nodes_jsonl(nodes_file_path)
    id_2_node = {node.id_: node for node in nodes}
    
    # question file
    question_df = pd.read_csv(question_csv_file_path)

    embed_config = prefix_config['embedding_model']['Linq-AI-Research/Linq-Embed-Mistral']
    embedding_model = get_embedding_model(embed_config)

    # Get embeddings and save them
    output_file_path = "./questions/gpt-4o-batch-all-p_pid_0.jsonl"
    get_queries_embeddings_and_save(embedding_model, id_2_node, question_df, output_file_path)

if __name__ == "__main__":
    node_dir = os.path.abspath('../../.save/gpt-4o-batch-all-target_1_parser/sub')
    node_file_name = "gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_50_pid_0.jsonl"
    nodes_file_path = os.path.join(node_dir, node_file_name)
    
    question_csv_dir = ("../../.save/gpt-4o-batch-all-target_1_parser/question")
    question_file_name = "gpt-4o-batch-all-target_extract_gpt-4o-QAExtractor-batch_pid_0.jsonl.csv"
    question_csv_file_path = os.path.join(question_csv_dir, question_file_name)
    
    add_query_embedding_values(nodes_file_path, question_csv_file_path)
