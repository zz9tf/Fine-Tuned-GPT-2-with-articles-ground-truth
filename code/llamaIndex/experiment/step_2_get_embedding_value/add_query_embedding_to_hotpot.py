import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from configs.load_config import load_configs
from component.models.embed.get_embedding_model import get_embedding_model
import torch
from tqdm import tqdm
import json

def get_queries_embeddings_and_save(embedding_model, questions, output_file_path):
    print("Getting embeddings from hotpot questions")
    
    # Open the output file in write mode
    with open(output_file_path, 'w') as f_out:
        # Process each question-answer pair
        for qa in tqdm(questions):
            # Generate embeddings for the question using the embedding model
            with torch.no_grad():
                question_embedding = embedding_model._get_text_embedding(qa['question'])
            
            # Prepare the data to save
            data = {
                '_id': qa['_id'],
                'question': qa['question'],
                'embedding': question_embedding  # Convert numpy array to list for JSON serialization
            }
            
            # Write the JSON object to the file, one per line
            f_out.write(json.dumps(data) + '\n')
    
    print(f"Embeddings successfully saved to {output_file_path}")

def add_query_embedding_values(question_path: str, output_file):
    # Configurations
    _, prefix_config = load_configs()

    # Open and read the JSON file
    with open(question_path, 'r') as test_file:
        questions = json.load(test_file)
    
    # Load embed model
    embed_config = prefix_config['embedding_model']['dunzhang/stella_en_400M_v5']
    embedding_model = get_embedding_model(embed_config)

    # Get embeddings and save them
    get_queries_embeddings_and_save(embedding_model, questions, f"./questions/{output_file}")

if __name__ == "__main__":
    question_dir = ("../../.cache")
    question_json_file_name = "hotpot_test_fullwiki_v1.json"
    question_path = os.path.join(question_dir, question_json_file_name)
    
    add_query_embedding_values(question_path, 'hotpot_questions.jsonl')
