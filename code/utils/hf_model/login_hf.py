import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv('HUGGINGFACE_TOKEN'), add_to_git_credential=True)