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
import time

if __name__ == '__main__':
    load_configs()
    dataset_dir_path = os.path.abspath('../step_4_generate_dataset/datasets')    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--matrix', type=str, help='The matrix to be executed')
    parser.add_argument('--now', type=str)
    parser.add_argument('--action', type=str, default='main')
    
    args = parser.parse_args()
    log_dir_path = os.path.abspath('./log')
    
    if args.action == 'main':
        dataset_names = [
            # Top k
            # 'gpt-4o-batch-all-target_one_retrieved_contexts_dataset_condition_2.jsonl',
            # 'gpt-4o-batch-all-target_document_retrieved_contexts_dataset_condition_2.jsonl',
            # 'gpt-4o-batch-all-target_section_retrieved_contexts_dataset_condition_2.jsonl',
            # 'gpt-4o-batch-all-target_paragraph_retrieved_contexts_dataset_condition_2.jsonl',
            # 'gpt-4o-batch-all-target_multi-sentences_retrieved_contexts_dataset_condition_2.jsonl',
            # 'gpt-4o-batch-all-target_predictor_top1_retrieved_contexts_dataset_condition_2.jsonl',
            # 'gpt-4o-batch-all-target_predictor_top2_retrieved_contexts_dataset_condition_2.jsonl',
            # 'gpt-4o-batch-all-target_predictor_top3_retrieved_contexts_dataset_condition_2.jsonl',
            # 'gpt-4o-batch-all-target_predictor_over25_percent_retrieved_contexts_dataset_condition_2.jsonl',
            # Top p
            # 'gpt-4o-batch-all-target_one_TopP_retrieved_contexts_dataset_condition_2.jsonl',
            # 'gpt-4o-batch-all-target_document_TopP_retrieved_contexts_dataset_condition_2.jsonl',
            # 'gpt-4o-batch-all-target_section_TopP_retrieved_contexts_dataset_condition_2.jsonl',
            # 'gpt-4o-batch-all-target_paragraph_TopP_retrieved_contexts_dataset_condition_2.jsonl',
            'gpt-4o-batch-all-target_multi-sentences_TopP_retrieved_contexts_dataset_condition_2.jsonl',
            # 'gpt-4o-batch-all-target_predictor_over25_percent_TopP_retrieved_contexts_dataset_condition_2.jsonl',
            # 'gpt-4o-batch-all-target_predictor_top1_TopP_retrieved_contexts_dataset_condition_2.jsonl',
            # 'gpt-4o-batch-all-target_predictor_top2_TopP_retrieved_contexts_dataset_condition_2.jsonl',
        ]
        
        processes = []
        log_file_paths = []
        
        for dataset_name in dataset_names:
            matrixes = [
                # 'faithfulness',
                # 'answer_relevancy',
                # 'answer_similarity',
                # 'answer_correctness',
                # 'context_precision',
                # 'context_utilization',
                # 'context_recall',
                'context_entity_recall'
            ]
            for matrix_name in matrixes:
                now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                log_file_path = os.path.join(log_dir_path, f'{dataset_name.split(".")[0]}_{matrix_name}_{now}.log')
                with open(log_file_path, 'w') as log_file:
                    process = subprocess.Popen(
                        [sys.executable, __file__, '--dataset_name', dataset_name, '--matrix', matrix_name, '--now', now, '--action', 'thread'],
                        stdout=log_file,
                        stderr=log_file
                    )
                    processes.append((process, log_file_path, matrix_name, dataset_name, now))
                    log_file_paths.append(log_file_path)
            while processes:
                for process, log_file_path, matrix_name, dataset_name, now in processes[:]:
                    if process.poll() is not None:  # Process finished
                        # Rename the log file
                        renamed_log_file_path = os.path.join(
                            log_dir_path, f'[done]{dataset_name.split(".")[0]}_{matrix_name}_{now}.log'
                        )
                        try:
                            os.rename(log_file_path, renamed_log_file_path)
                            print(f"Renamed log file: {renamed_log_file_path}")
                        except PermissionError as e:
                            print(f"Error renaming file {log_file_path}: {e}")

                        # Remove the completed process from the list
                        processes.remove((process, log_file_path, matrix_name, dataset_name, now))
                
                # Sleep briefly to avoid busy-waiting
                time.sleep(1)
            
    elif args.action == 'thread':
        matrix_name = args.matrix
        now = args.now
        save_file_name = f"{args.dataset_name.split('.')[0]}_{matrix_name}_{now}.csv"
        save_file_path = f'./score/{save_file_name}'
        dataset = load_dataset_from_jsonl(os.path.join(dataset_dir_path, args.dataset_name))
        
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
        