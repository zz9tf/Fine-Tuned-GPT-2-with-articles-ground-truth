import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))
import gc
import torch
import json
from tqdm import tqdm
from configs.load_config import load_configs
from component.models.embed.get_embedding_model import get_embedding_model
import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

def load_embedding_model(prefix_config):
    print("Loading embedding model")
    embed_config = prefix_config['embedding_model']['Linq-AI-Research/Linq-Embed-Mistral']
    embedding_model = get_embedding_model(embed_config)
    return embedding_model

def generate_document_from_str(text):   
    return Document(text=text)

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
                'embedding': embedding,  # Convert embedding to a list for JSON serialization
            }
                
            # Write the data to the output file in JSON Lines format
            f_out.write(json.dumps(data) + '\n')  # Write JSON line and add a newline

    print(f"Embeddings successfully saved to {output_file_path}")

if __name__ == '__main__':
    _, prefix_config = load_configs()
    # Get data
    df = pd.read_csv('scidocs_validation.csv')
    documents = [generate_document_from_str(text) for text in tqdm(df['positive'], desc='generating documents...')]
    
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    nodes = node_parser.get_nodes_from_documents(
        documents, show_progress=True
    )
    
    # Get embedding model
    embedding_model = load_embedding_model(prefix_config)
    
    # Generate embeddings
    get_text_embeddings_and_save(
        embedding_model=embedding_model,
        nodes=nodes,
        output_file_path=os.path.join(os.path.abspath('.'), 'scidocs_embeddings.jsonl')
    )