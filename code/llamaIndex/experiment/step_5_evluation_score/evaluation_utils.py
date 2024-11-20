import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from tqdm import tqdm
import json
from typing import Union
from datasets import Dataset
from ragas import evaluate
import time
from ragas.run_config import RunConfig
from ragas.metrics.base import Metric

def load_dataset_from_jsonl(input_path, start_num=0, end_num=None):
    dataset = {
        "question": [],
        "ground_truth": [],
        "answer": [],
        "contexts": []
    }
    file_size = os.path.getsize(input_path)
    with open(input_path, 'r') as input_file:
        with tqdm(total=file_size, desc=f'Loading {input_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for idx, line in enumerate(input_file):
                if idx >= start_num and (end_num is None or idx < end_num): # [1000] [33] [1000]
                    # print(f"Tracking row {idx}: {row}")  # Print the tracked row
                    row = json.loads(line)
                    dataset["question"].append(row["question"])
                    dataset["ground_truth"].append(row["ground_truth"])
                    dataset["answer"].append(row["answer"])
                    dataset["contexts"].append(row["context"])
                pbar.update(len(line))
                
    return Dataset.from_dict(dataset)

def evaluation_with_metrics(dataset: Dataset, metrix: Union[Metric, list[Metric]], save_file_path: str):
    # Get llm
    # with open(os.path.join(os.path.abspath('../../configs'), 'prefix_config.yaml'), 'r') as prefix_config_file:
    #     prefix_config = yaml.safe_load(prefix_config_file)
    # llm_config = prefix_config['llm']['lmsys/vicuna-13b-v1.5']
    # llm = get_llm(None, llm_config, 'cuda:1')
    
    # embedding_config = prefix_config['embedding_model']['Linq-AI-Research/Linq-Embed-Mistral']
    # embed_model = get_embedding_model(embedding_config, 'cuda:2')
    
    run_config = RunConfig(
        timeout=900,
        max_retries=20,
        max_wait=600
    )
    
    # Evaluate
    start_time = time.time()
    score = evaluate(
        dataset=dataset,
        metrics=[metrix] if isinstance(metrix, Metric) else metrix,
        # llm=LlamaIndexLLMWrapper(llm),
        # embeddings=LlamaIndexEmbeddingsWrapper(embed_model),
        run_config=run_config,
        raise_exceptions=True
    )
    print(score.to_pandas())
    end_time = time.time()  # End timing

    # Calculate and print the time taken
    elapsed_time = end_time - start_time
    print(f"Time taken to process the dataset: {elapsed_time/60:.2f} minutes")
    
    return score.to_pandas().to_csv(save_file_path, index=False)

if __name__ == '__main__':
    pass