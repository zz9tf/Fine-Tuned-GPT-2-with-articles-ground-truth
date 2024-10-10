import os
import sys
sys.path.insert(0, '../..')
import json
import pandas as pd
from datasets import Dataset

def load_df(
    qar_file_name: str='gpt-4o-batch-all-target_extract_gpt-4o-QAExtractor-batch_pid_0.jsonl.csv'
):
    qar_dataset_path:str=os.path.join(os.path.abspath('../../.cache'), qar_file_name)
    return pd.read_csv(qar_dataset_path)
    
def get_context(input_text):
    lines = input_text.split('\n')
    for idx, text in enumerate(lines):
        if text.startswith('Here is the context'):
            # The next line will be the context string
            if idx + 1 < len(lines):
                return lines[idx + 1]

def get_quetions_answers_and_grountruth(df):
    questions = []
    ground_truths = []
    correct_contexts = []

    for row_id, obj_str in enumerate(df['objs']):
        objs = json.loads(obj_str)
        if len(objs) == 0:
            print(f"[Invalid rqa pairs] No objs found at row id: {row_id}")
        for obj in objs:
            questions.append(obj['Question'])
            ground_truths.append(obj['Answer'])
            correct_contexts.append([get_context(df['input_text'][row_id])])
    return questions, ground_truths, correct_contexts
    
def save_dataset(questions, correct_contexts, ground_truth):
    # Prepare the dataset
    dataset = [
        {
            'question': question,
            'correct_contexts': context,
            'ground_truth': truth
        }
        for question, context, truth in zip(questions, correct_contexts, ground_truth)
    ]

    # Write the dataset to a JSONL file
    file_path = os.path.abspath('./dataset/qcg_dataset.jsonl')
    with open(file_path, 'w') as f:
        for data in dataset:
            f.write(json.dumps(data) + '\n')  # Each dict as a separate line

    return file_path

if __name__ == '__main__':
    df = load_df()
    questions, ground_truths, correct_contexts = get_quetions_answers_and_grountruth(df)
    save_dataset(questions, correct_contexts, ground_truths)