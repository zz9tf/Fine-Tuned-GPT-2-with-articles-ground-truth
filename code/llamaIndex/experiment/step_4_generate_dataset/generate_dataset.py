import os
import sys
sys.path.insert(0, '../..')
import pandas as pd
import json
import pandas as pd
from tqdm import tqdm
from component.models.llm.get_llm import get_llm
from configs.load_config import load_configs
from component.schema import Gen_Dataset_Temp
from component.schema import old_gen_temp
import time
from datetime import datetime
import argparse
import subprocess

def get_quetions_groundtruth_correct_contexts(qar_dataset_path: str):
    """ This method will return question, ground_truths, and correct_retrieved_contexts """
    def get_context(input_text):
        lines = input_text.split('\n')
        for idx, text in enumerate(lines):
            if text.startswith('Context'):
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
        for i, obj in enumerate(objs):
            questions.append(obj['Question'])
            ground_truths.append(obj['Answer'])
            
            correct_contexts.append([get_context(df['input_text'][row_id])])
    return questions, ground_truths, correct_contexts

def get_quetions_groundtruth_contexts(
    qar_dataset_path: str, retrieved_contexts_file_path: str
):
    """ This method will return question, ground_truths, and correct_retrieved_contexts """
                
    def load_retrieved_contexts(
        retrieved_contexts_file_path: str = './retrieved_contexts/multi_level_retrieved_contexts.jsonl'
    ):
        file_size = os.path.getsize(retrieved_contexts_file_path)
        contexts_dict = {}
        # Read the file and track progress based on bytes read
        with open(retrieved_contexts_file_path, 'r') as input_file:
            with tqdm(total=file_size, desc=f'Loading {retrieved_contexts_file_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for line in input_file:
                    row = json.loads(line)
                    if row['question_node_id'] not in contexts_dict:
                        contexts_dict[row['question_node_id']] = {} 
                    contexts_dict[row['question_node_id']][row['question_id']] = row['retrieved_contexts']
                    # Update progress bar based on bytes read
                    pbar.update(len(line))
        
        return contexts_dict
    df = pd.read_csv(qar_dataset_path)
    contexts_dict = load_retrieved_contexts(retrieved_contexts_file_path)
    questions = []
    ground_truths = []
    contexts = []
    for row_id, obj_str in enumerate(df['objs']):
        objs = json.loads(obj_str)
        if len(objs) == 0:
            print(f"[Invalid rqa pairs] No objs found at row id: {row_id} which should have {df['qar_num'][row_id]} questions")
        for i, obj in enumerate(objs):
            questions.append(obj['Question'])
            ground_truths.append(obj['Answer'])
            # ################## top k ##########################
            # context_list = contexts_dict[df['node_id'][row_id]][i]
            # # retrieve 5 for one
            # retrieve_num = 10
            # cur_contexts = [text for i, text in enumerate(context_list[0]) if i < retrieve_num]
            # if len(cur_contexts) != retrieve_num:
            #     print(row_id)
            #     print(i)
            #     break
            
            # ############################################
            
            # ################## length threshold ##########################
            text_len = 0
            cur_contexts = []
            context_list = contexts_dict[df['node_id'][row_id]][i]
            context_length_threshold = 10000
            
            for text_group in zip(*context_list):
                text_len += sum([len(text) for text in text_group])
                if text_len > context_length_threshold:
                    break
                cur_contexts.extend(text_group)
            # ############################################
            contexts.append(cur_contexts)
    return questions, ground_truths, contexts

def generate_answer_and_save_as_jsonl(llm_config, questions, ground_truths, contexts, save_path:str='./dataset.jsonl'):
    llm = get_llm(llm_config)
    existing_data = {}
    # Read the current file content into memory
    if os.path.exists(save_path):
        with open(save_path, 'r') as save_file:
            for line in save_file:
                existing_data[json.loads(line)['question']] = line
    
    # qa_prompt = old_gen_temp
    qa_prompt = Gen_Dataset_Temp.prompt_template
    parser = Gen_Dataset_Temp.parser
    
    with open(save_path, 'w') as save_file, tqdm(total=len(questions), desc="Processing questions") as pbar:
        for id, (question, retrieved_contexts, ground_truth) in enumerate(zip(questions, contexts, ground_truths)):
            if question in existing_data:
                save_file.write(f"{existing_data[question]}")
            else:
                # Generate a response for each query
                retrieved_contexts_str = "".join([f" {i+1}. {s}\n" for i, s in enumerate(retrieved_contexts)])
                fmt_qa_prompt = qa_prompt.format(context_str=retrieved_contexts_str, query_str=question)
                answer = llm.complete(fmt_qa_prompt).text.strip()
                try:
                    obj = parser.parse(answer)
                    # Save the generated response to the JSONL file
                    data = {
                        "question": question,
                        "ground_truth": ground_truth,
                        "answer": obj.Answer,
                        # "answer": '',
                        "contexts": retrieved_contexts
                    }
                    json.dump(data, save_file)
                    save_file.write("\n")
                except:
                    print(f'[skip] {id}: {question}')
                    print(f'prompt: {fmt_qa_prompt}')
                    print(f'answer: {answer}')
                    print(obj)
                    input()
                    # print(answer)
            pbar.update(1)
    save_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, help='The condition of the experiement')
    parser.add_argument('--retrieved_file_name', type=str, help='The retrieved file name of the experiement')
    parser.add_argument('--now', type=str)
    parser.add_argument('--action', type=str, default='main')
    args = parser.parse_args()
    log_dir_path = os.path.abspath('./log')
    
    if args.action == 'main':
        retrieved_file_config = [
            # Top k
            # ['gpt-4o-batch-all-target_one_retrieved_contexts.jsonl', 2],
            # ['gpt-4o-batch-all-target_document_retrieved_contexts.jsonl', 2],
            # ['gpt-4o-batch-all-target_section_retrieved_contexts.jsonl', 2],
            # ['gpt-4o-batch-all-target_paragraph_retrieved_contexts.jsonl', 2],
            # ['gpt-4o-batch-all-target_multi-sentences_retrieved_contexts.jsonl', 2],
            # ['gpt-4o-batch-all-target_predictor_top1_retrieved_contexts.jsonl', 2],
            # ['gpt-4o-batch-all-target_predictor_top2_retrieved_contexts.jsonl', 2],
            # ['gpt-4o-batch-all-target_predictor_top3_retrieved_contexts.jsonl', 2], #
            # ['gpt-4o-batch-all-target_predictor_over25_percent_retrieved_contexts.jsonl', 2],
            # Top p
            # ['gpt-4o-batch-all-target_one_TopP_retrieved_contexts.jsonl', 2],
            # ['gpt-4o-batch-all-target_document_TopP_retrieved_contexts.jsonl', 2],
            # ['gpt-4o-batch-all-target_section_TopP_retrieved_contexts.jsonl', 2],
            # ['gpt-4o-batch-all-target_paragraph_TopP_retrieved_contexts.jsonl', 2], #
            # ['gpt-4o-batch-all-target_multi-sentences_TopP_retrieved_contexts.jsonl', 2],
            # ['gpt-4o-batch-all-target_predictor_top1_TopP_retrieved_contexts.jsonl', 2],
            # ['gpt-4o-batch-all-target_predictor_top2_TopP_retrieved_contexts.jsonl', 2],
            # ['gpt-4o-batch-all-target_predictor_top3_TopP_retrieved_contexts.jsonl', 2], #
            # ['gpt-4o-batch-all-target_predictor_over25_percent_TopP_retrieved_contexts.jsonl', 2],
            # Retrival with predictor strategy
            ['gpt-4o-batch-all-target_predictor_top2_depending_on_similarity_retrieved_contexts.jsonl', 2],
        ]
        processes = []
        log_file_paths = []
        
        for (retrieved_file_name, condition) in retrieved_file_config:
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_file_path = os.path.join(log_dir_path, f'{retrieved_file_name}_{now}.log')
            with open(log_file_path, 'w') as log_file:
                process = subprocess.Popen(
                    [sys.executable, __file__, '--condition', str(condition), '--retrieved_file_name', retrieved_file_name, '--now', now, '--action', 'thread'],
                    stdout=log_file,
                    stderr=log_file
                )
                processes.append((process, log_file_path, retrieved_file_name, now))
                log_file_paths.append(log_file_path)
        while processes:
            for process, log_file_path, retrieved_file_name, now in processes[:]:
                if process.poll() is not None:  # Process finished
                    # Rename the log file
                    renamed_log_file_path = os.path.join(
                        log_dir_path, f'[done]{retrieved_file_name}_{now}.log'
                    )
                    try:
                        os.rename(log_file_path, renamed_log_file_path)
                        print(f"Renamed log file: {renamed_log_file_path}")
                    except PermissionError as e:
                        print(f"Error renaming file {log_file_path}: {e}")

                    # Remove the completed process from the list
                    processes.remove((process, log_file_path, retrieved_file_name, now))
            
            # Sleep briefly to avoid busy-waiting
            time.sleep(1)
    elif args.action == 'thread':
        qar_file_name = 'gpt-4o-batch-all-target_extract_gpt-4o-QAExtractor-batch_pid_0.jsonl.csv' # modify each time
        qar_dataset_path = os.path.join(os.path.abspath('../../.save/gpt-4o-batch-all-target_1_parser/question'), qar_file_name)
        
        _, perfix_config = load_configs()
        llm_config = perfix_config['llm']['gpt-4o-mini']
        retrieved_contexts_path = os.path.abspath(f'../step_4_0_retrieve_contexts/retrieved_contexts/{args.retrieved_file_name}')
        prefix = args.retrieved_file_name.split('.')[0] # sentence_splitter
        save_file_name = f"{prefix}_dataset_condition_{args.condition}.jsonl" # modify each time
        save_path = os.path.abspath(os.path.join('./datasets', save_file_name))
        print(f"save path: {save_path}")
        
        if args.condition == 1:
            q, g, cc = get_quetions_groundtruth_correct_contexts(qar_dataset_path)
            generate_answer_and_save_as_jsonl(llm_config, q, g, cc, save_path)
        else:
            q, g, c = get_quetions_groundtruth_contexts(qar_dataset_path, retrieved_contexts_file_path=retrieved_contexts_path)
            # input()
            generate_answer_and_save_as_jsonl(llm_config, q, g, c, save_path)