import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from configs.load_config import load_configs
import argparse
import subprocess
from datetime import datetime
from evaluation_utils import load_dataset_from_jsonl, evaluation_with_metrics
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_similarity,
    answer_correctness,
    context_precision,
    context_utilization,
    context_recall,
    context_entity_recall,
    noise_sensitivity_relevant,
    noise_sensitivity_irrelevant
)

if __name__ == '__main__':
    load_configs()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, help='The condition of the experiment')
    parser.add_argument('--matrix', type=str, help='The matrix to be executed')
    parser.add_argument('--now', type=str)
    parser.add_argument('--action', type=str, default='main')
    
    args = parser.parse_args()
    log_dir_path = os.path.abspath('./log')
    
    if args.action == 'main':
        matrixes = [
            'faithfulness',
            'answer_relevancy',
            'answer_similarity',
            'answer_correctness',
            'context_precision',
            'context_utilization',
            'context_recall',
            'context_entity_recall'
        ]
        condition = '1'
        for matrix_name in matrixes:
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_file_path = os.path.join(log_dir_path, f'{matrix_name}_condition_{condition}_{now}.log')
            with open(log_file_path, 'w') as log_file:
                subprocess.Popen(
                    [sys.executable, __file__, '--condition', condition, '--matrix', matrix_name, '--now', now, '--action', 'thread'],
                    stdout=log_file,
                    stderr=log_file
                )
    elif args.action == 'thread':
        condition = args.condition
        matrix_name = args.matrix
        now = args.now
        
        dataset_dir_path = os.path.abspath('../step_4_generate_dataset/datasets')    
        dataset_name = f'dataset_condition_{condition}.jsonl'
        save_file_name = f"{dataset_name.split('.')[0]}_{matrix_name}_{now}.csv"
        save_file_path = f'./score/{save_file_name}'
        dataset = load_dataset_from_jsonl(os.path.join(dataset_dir_path, dataset_name))
        
        if matrix_name == 'faithfulness':
            evaluation_with_metrics(dataset, faithfulness, save_file_path)
        elif matrix_name == 'answer_relevancy':
            evaluation_with_metrics(dataset, answer_relevancy, save_file_path)
        elif matrix_name == 'answer_similarity':
            evaluation_with_metrics(dataset, answer_similarity, save_file_path)
        elif matrix_name == 'answer_correctness':
            evaluation_with_metrics(dataset, answer_correctness, save_file_path)
        elif matrix_name == 'context_precision':
            evaluation_with_metrics(dataset, context_precision, save_file_path)
        elif matrix_name == 'context_utilization':
            evaluation_with_metrics(dataset, context_utilization, save_file_path)
        elif matrix_name == 'context_recall':
            evaluation_with_metrics(dataset, context_recall, save_file_path)
        elif matrix_name == 'context_entity_recall':
            evaluation_with_metrics(dataset, context_entity_recall, save_file_path)
        elif matrix_name == 'noise_sensitivity_relevant':
            evaluation_with_metrics(dataset, [noise_sensitivity_relevant, noise_sensitivity_irrelevant], save_file_path)
            
        # Rename the log file here
        original_log_file_path = os.path.join(log_dir_path, f'{matrix_name}_condition_{condition}_{now}.log')
        renamed_log_file_path = os.path.join(log_dir_path, f'[done]{matrix_name}_condition_{condition}_{now}.log')

        # Rename the log file
        os.rename(original_log_file_path, renamed_log_file_path)
        