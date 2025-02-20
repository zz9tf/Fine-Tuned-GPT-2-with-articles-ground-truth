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

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieved_file_name', type=str, help='The retrieved file name of the experiement')
    parser.add_argument('--now', type=str)
    parser.add_argument('--action', type=str, default='main')
    return parser.parse_args()

def load_retrieved_contexts(
    retrieved_contexts_file_path: str = '../step_3_retrieve_contexts/retrieved_contexts/wikipedia-mal-rag_one_TopP_retrieved_contexts.jsonl'
):
    file_size = os.path.getsize(retrieved_contexts_file_path)
    contexts_dict = {}
    # Read the file and track progress based on bytes read
    with open(retrieved_contexts_file_path, 'r') as input_file:
        with tqdm(total=file_size, desc=f'Loading {retrieved_contexts_file_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for line in input_file:
                row = json.loads(line)
                if row['_id'] not in contexts_dict:
                    contexts_dict[row['_id']] = {} 
                contexts_dict[row['_id']] = row['retrieved_contexts']
                # Update progress bar based on bytes read
                pbar.update(len(line))
    
    return contexts_dict

def load_questions(questions_file_path):
    questions = []
    file_size = os.path.getsize(questions_file_path)
    with open(questions_file_path, 'r') as questions_file:
        with tqdm(total=file_size, desc=f'Load hotpot questions', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for line in questions_file:
                question = json.loads(line)
                questions.append({
                    '_id': question['_id'],
                    'question': question['question']
                })
                pbar.update(len(line))
    return questions

def generate_answer_and_save_as_jsonl(llm_config, questions, contexts, save_path:str='./dataset.jsonl'):
    llm = get_llm(llm_config)
    existing_data = {}
    # Read the current file content into memory
    if os.path.exists(save_path):
        with open(save_path, 'r') as save_file:
            for line in save_file:
                qac = json.loads(json.loads(line)) # question, answer, contexts
                existing_data[qac['question']] = line
    qa_prompt = Gen_Dataset_Temp.prompt_template
    parser = Gen_Dataset_Temp.parser
    
    with open(save_path, 'w') as save_file, tqdm(total=len(questions), desc="Processing questions") as pbar:
        for question, retrieved_contexts in zip(questions, contexts):
            if question['question'] in existing_data:
                save_file.write(existing_data[question])
            else:
                retrieved_contexts_str = "".join([f" {i+1}. {s}\n" for i, s in enumerate(retrieved_contexts)])
                fmt_qa_prompt = qa_prompt.format(context_str=retrieved_contexts_str, query_str=question)
                answer = llm.complete(fmt_qa_prompt).text.strip()
                try:
                    obj = parser.parse(answer)
                    # Save the generated response to the JSONL file
                    data = {
                        "question": question,
                        "answer": obj.Answer,
                        'answer': '',
                        "contexts": retrieved_contexts
                    }
                    json.dump(data, save_file)
                    save_file.write("\n")
                except:
                    print(f'[skip] {id}: {question}')
                    print(f'prompt: {fmt_qa_prompt}')
                    # print(f'answer: {answer}')
                    # print(obj)
                    # input()
            pbar.update(1)
    save_file.close()

if __name__ == '__main__':
    args = load_args()
    log_dir_path = './log'
    
    if args.action == 'main':
        retrieved_file_config = [
            # Top p
            'wikipedia-mal-rag_one_TopP_retrieved_contexts.jsonl',
        ]
        processes = []
        log_file_paths = []
        
        for retrieved_file_name in retrieved_file_config:
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_file_path = os.path.join(log_dir_path, f'{retrieved_file_name}_{now}.log')
            with open(log_file_path, 'w') as log_file:
                process = subprocess.Popen(
                    [sys.executable, __file__, '--retrieved_file_name', retrieved_file_name, '--now', now, '--action', 'thread'],
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
        # Load hyperparameters
        _, perfix_config = load_configs()
        llm_config = perfix_config['llm']['gpt-4o-mini']
        prefix = args.retrieved_file_name.split('.')[0] # sentence_splitter or wikipedia-mal-rag
        
        # Save path
        save_file_name = f"{prefix}_dataset.jsonl" # modify each time
        save_path = os.path.abspath(os.path.join('./datasets', save_file_name))
        print(f"save path: {save_path}")
        
        # Load contexts
        retrieved_contexts_path = os.path.abspath(f'../step_3_retrieve_contexts/retrieved_contexts/{args.retrieved_file_name}')
        contexts = load_retrieved_contexts(retrieved_contexts_path)
        
        # Load questions
        questions_file_name = 'hotpot_questions.jsonl' # modify each time
        questions_file_path = os.path.join('../step_2_get_embedding_value/questions', questions_file_name)
        questions = load_questions(questions_file_path)
        
        generate_answer_and_save_as_jsonl(llm_config, questions, contexts, save_path)