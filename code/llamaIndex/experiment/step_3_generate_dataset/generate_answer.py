import os, sys
import yaml
import json
from tqdm import tqdm
sys.path.insert(0, os.path.abspath('../..'))
from llama_index.core import PromptTemplate
from configs.load_config import load_configs
from component.models.llm.get_llm import get_llm

qa_prompt = PromptTemplate(
    """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: \
"""
)

def generate_response(context_str, query_str, qa_prompt, llm):
    fmt_qa_prompt = qa_prompt.format(
        context_str=context_str, query_str=query_str
    )
    response = llm.complete(fmt_qa_prompt)
    return str(response), fmt_qa_prompt

def load_dataset(file_path):
    queries = []
    correct_contexts = []
    ground_truths = []

    # Open and read the .jsonl file
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)  # Parse each line as a JSON object    
                queries.append(data['question'])
                correct_contexts.append(data['correct_contexts'])
                ground_truths.append(data['ground_truth'])
            except Exception as e:
                print(f"An error occurred while loading the dataset: {e}")
    
    return queries, correct_contexts, ground_truths

def save(query, context_str, ground_true, answer, output_file):
    # Create a dictionary with the desired format
    data = {
        "question": query,
        "correct_contexts": context_str,
        "ground_true": ground_true,  # Assuming ground_true comes from CSV
        "answer": answer
    }
    # Write the dictionary as a JSON line in the file
    output_file.write(json.dumps(data) + "\n")
    
if __name__ == "__main__":
    _, prefix_config = load_configs()
    llm_config = prefix_config['llm']['lmsys/vicuna-13b-v1.5']
    llm = get_llm(llm_config)
    
    data_path = "./datasets/qcg_dataset.jsonl"  # Path to your CSV file
    # Load the CSV into a pandas DataFrame
    queries, correct_contexts, ground_truths = load_dataset(data_path)
    
    mode = "with_correct_contexts"
    cache_file = open(f'answer_{mode}.jsonl', "a")
    with tqdm(total=len(queries), desc="generating answer...") as pbar:
        for (query_str, context_str, ground_true) in zip(queries, correct_contexts, ground_truths):
            # Generate a response for each query
            response, fmt_qa_prompt = generate_response(
                context_str, query_str, qa_prompt, llm
            )
            # Save the generated response to the JSONL file
            save(query_str, context_str, ground_true, response, cache_file)
            pbar.update(1)