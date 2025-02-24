import os
from dotenv import load_dotenv

def list_folders(directory):
    directory = os.path.join(directory, '.hf_cache')
    try:
        # List all files and directories in the specified directory
        items = os.listdir(directory)
        
        print(f"[Show cache] Path {directory}:")
        # Filter out only directories
        for item in items:
            if os.path.isdir(os.path.join(directory, item)):
                if item.startswith('models'):
                    print("[Show cache]  - {}".format(item))
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# Load environment variables from .env file
load_dotenv()
save_dir_path = os.getenv('SAVE_DIR_PATH')
list_folders(save_dir_path)


