import pandas as pd

import os
import sys
sys.path.insert(0, '../..')
import json
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from llama_index.core import PromptTemplate
from component.models.llm.get_llm import get_llm
from configs.load_config import load_configs
from experiment.step_4_generate_dataset.retrieved_contexts import load_retrieved_contexts

def get_quetions_grountruth_correct_contexts(qar_dataset_path: str):
    """ This method will return question, ground_truths, and correct_retrieved_contexts """
    def get_context(input_text):
        lines = input_text.split('\n')
        for idx, text in enumerate(lines):
            if text.startswith('Here is the context'):
                # The next line will be the context string
                if idx + 1 < len(lines):
                    return lines[idx + 1]
    df = pd.read_csv(qar_dataset_path)
    questions = []
    ground_truths = []
    correct_contexts = []

    for row_id, obj_str in enumerate(df['objs']):
        objs = json.loads(obj_str)
        if len(objs) == 0:
            print(f"[Invalid rqa pairs] No objs found at row id: {row_id} which should have {df['qar_num'][row_id]} questions")
        for obj in objs:
            questions.append(obj['Question'])
            ground_truths.append(obj['Answer'])
            correct_contexts.append([get_context(df['input_text'][row_id])])
    return questions, ground_truths, correct_contexts

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

def generate_answer_and_save_as_jsonl(llm_config, questions, ground_truths, contexts, save_path:str='./dataset.jsonl'):
    llm = get_llm(llm_config)
    save_file = open(save_path, 'w')
    with tqdm(total=len(questions), desc="generating answer...") as pbar:
        for (question, retrieved_contexts, ground_truth) in zip(questions, contexts, ground_truths):
            # Generate a response for each query
            retrieved_contexts_str = "\n".join(retrieved_contexts)
            fmt_qa_prompt = qa_prompt.format(context_str=retrieved_contexts_str, query_str=question)
            answer = llm.complete(fmt_qa_prompt)
            # Save the generated response to the JSONL file
            data = {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "context": retrieved_contexts
            }
            save_file.write(json.dumps(data) + "\n")
            pbar.update(1)
    save_file.close()

def load_dataset_from_jsonl(input_path):
    dataset = {
        "question": [],
        "ground_truth": [],
        "answer": [],
        "contexts": []
    }
    file_size = os.path.getsize(input_path)
    with open(input_path, 'r') as input_file:
        with tqdm(total=file_size, desc=f'Loading {input_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for line in input_file:
                row = json.load(line)
                dataset["question"].append(row["question"])
                dataset["ground_truth"].append(row["ground_truth"])
                dataset["answer"].append(row["answer"])
                dataset["contexts"].append(row["context"])
                pbar.update(len(line))
                
    return Dataset.from_dict(dataset)

if __name__ == '__main__':
    qar_file_name = 'gpt-4o-batch-all-target_extract_gpt-4o-QAExtractor-batch_pid_0.jsonl.csv'
    qar_dataset_path = os.path.join(os.path.abspath('../../.cache'), qar_file_name)
    q, g, cc = get_quetions_grountruth_correct_contexts(qar_dataset_path)
    condition = 1
    save_file_name = f"dataset_condition_{condition}.jsonl"
    save_path = os.path.abspath(os.path.join('./datasets', save_file_name))
    print(f"save path: {save_path}")
    
    _, perfix_config = load_configs()
    llm_config = perfix_config['llm']['gpt-4o-mini']
    if condition == 1:
        generate_answer_and_save_as_jsonl(llm_config, q, g, cc, save_path)
    elif condition == 2:
        c = load_retrieved_contexts()
        generate_answer_and_save_as_jsonl(llm_config, q, g, c, save_path)
    else:
        pass