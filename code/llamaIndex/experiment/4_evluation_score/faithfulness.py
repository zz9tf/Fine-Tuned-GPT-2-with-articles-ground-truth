import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7,8,9'
sys.path.insert(0, os.path.abspath('../..'))
from tqdm import tqdm
import json
import pandas as pd
from datasets import Dataset
from ragas.metrics import faithfulness
# from ragas.integrations.llama_index import evaluate
from ragas import evaluate
import yaml
from custom.llm import get_llm
from custom.embedding import get_embedding_model
import time
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper

def load_jsonl_as_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a JSON Lines (JSONL) file and return its contents as a Pandas DataFrame.
    
    Each line in the JSONL file is parsed as a dictionary and then converted to a DataFrame.
    
    :param file_path: Path to the JSONL file to load.
    :return: Pandas DataFrame representing the data in the JSONL.
    """
    data = []
    
    try:
        # Get the total file size
        file_size = os.path.getsize(file_path)
        
        # Read the file and track progress based on bytes read
        with open(file_path, 'r', encoding='utf-8') as file:
            with tqdm(total=file_size, desc=f'Loading {os.path.basename(file_path)}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for line in file:
                    # Parse each line as JSON and append the dictionary to the data list
                    data.append(json.loads(line))
                    # Update progress bar based on bytes read
                    pbar.update(len(line))
    except Exception as e:
        print(f"An error occurred while loading the JSONL file: {e}")
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    
    return df

if __name__ == '__main__':
    # Load dataset
    file_path = os.path.abspath('../3_generate_dataset/answer_with_correct_contexts.jsonl')
    dataset = load_jsonl_as_dataframe(file_path=file_path)
    dataset.rename(columns={'correct_contexts': 'contexts'}, inplace=True)
    dataset.drop(columns=['ground_true'], inplace=True)
    dataset = {col: dataset[col].tolist()[:4] for col in dataset.columns}
    dataset = Dataset.from_dict(dataset)
    
    # Get llm
    with open(os.path.join(os.path.abspath('../../configs'), 'prefix_config.yaml'), 'r') as prefix_config_file:
        prefix_config = yaml.safe_load(prefix_config_file)
    llm_config = prefix_config['llm']['lmsys/vicuna-13b-v1.5']
    llm = get_llm(None, llm_config, 'cuda:1')
    
    embedding_config = prefix_config['embedding_model']['Linq-AI-Research/Linq-Embed-Mistral']
    embed_model = get_embedding_model(embedding_config, 'cuda:2')
    
    # Evaluate
    start_time = time.time()
    score = evaluate(
        dataset=dataset,
        metrics=[faithfulness],
        llm=LlamaIndexLLMWrapper(llm),
        embeddings=LlamaIndexEmbeddingsWrapper(embed_model),
        raise_exceptions=True
    )
    print(score.to_pandas())
    end_time = time.time()  # End timing

    # Calculate and print the time taken
    elapsed_time = end_time - start_time
    print(f"Time taken to process the dataset: {elapsed_time/60:.2f} minutes")
