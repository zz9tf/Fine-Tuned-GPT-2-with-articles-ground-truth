import os
import argparse
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Download and save Hugging Face model and tokenizer.")
    parser.add_argument("repo_name", type=str, help="Name of the repository to download from Hugging Face")
    parser.add_argument("from_scource", type=str, nargs='?', default=None, help="Download model from Hugging Face /'hf/' or SentenceTransformer")
    return parser.parse_args()

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve environment variables
    hug_token = os.getenv('HUGGINGFACE_TOKEN')
    save_dir_path = os.path.join(os.getenv('SAVE_DIR_PATH'), '.hf_cache')

    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)

    # Parse arguments
    args = parse_args()
    repo_name = args.repo_name

    # Hugging Face login
    login(token=hug_token, add_to_git_credential=True)

    
    if args.from_scource == 'hf':
        # Load model and tokenizer from the specified repository
        AutoTokenizer.from_pretrained(repo_name, cache_dir=save_dir_path)
        AutoModel.from_pretrained(repo_name, cache_dir=save_dir_path)

    else:
        SentenceTransformer(repo_name, cache_folder=save_dir_path)

    

if __name__ == "__main__":
    main()