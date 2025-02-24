import os
import argparse
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Download and save Hugging Face model and tokenizer.")
    parser.add_argument(
        "from_source", 
        type=str, 
        nargs='?', 
        default=None, 
        help="Specify the model source: 'AutoModelForCausalLM', 'SentenceTransformer', 'AutoModel', 'AutoModelForSequenceClassification'."
    )
    parser.add_argument("repo_name", type=str, help="Name of the repository to download from Hugging Face")
    
    return parser.parse_args()

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve environment variables
    hug_token = os.getenv('HUGGINGFACE_TOKEN')
    save_dir_path = os.path.join(os.getenv('SAVE_DIR_PATH'), '.hf_cache')

    # Parse arguments
    args = parse_args()
    repo_name = args.repo_name

    # Hugging Face login
    login(token=hug_token, add_to_git_credential=True)

    
    if args.from_source == 'AutoModelForCausalLM':
        AutoTokenizer.from_pretrained(repo_name, cache_dir=save_dir_path)
        model = AutoModelForCausalLM.from_pretrained(repo_name, cache_dir=save_dir_path)
    elif args.from_source == 'SentenceTransformer':
        model = SentenceTransformer(repo_name, cache_folder=save_dir_path)[0]
    elif args.from_source == 'AutoModel':
        AutoTokenizer.from_pretrained(repo_name, cache_dir=save_dir_path)  
        model = AutoModel.from_pretrained(repo_name, cache_dir=save_dir_path)
    elif args.from_source == 'AutoModelForSequenceClassification':
        AutoTokenizer.from_pretrained(repo_name, cache_dir=save_dir_path)  
        model = AutoModelForSequenceClassification.from_pretrained(repo_name, cache_dir=save_dir_path)
    elif args.from_source == 'AutoModelForSeq2SeqLM':
        AutoTokenizer.from_pretrained(repo_name, cache_dir=save_dir_path)  
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_name, cache_dir=save_dir_path)
        
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params}")

if __name__ == "__main__":
    main()