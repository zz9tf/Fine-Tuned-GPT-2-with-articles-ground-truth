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
import math
import datetime

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

    exceed_contexts_num = 0
    for row_id, obj_str in enumerate(df['objs']):
        objs = json.loads(obj_str)
        if len(objs) == 0:
            print(f"[Invalid rqa pairs] No objs found at row id: {row_id} which should have {df['qar_num'][row_id]} questions")
        for i, obj in enumerate(objs):
            questions.append(obj['Question'])
            ground_truths.append(obj['Answer'])
            text_len = 0
            used_texts = 0
            cur_contexts = []
            for contexts in contexts_dict[df['node_id'][row_id]][i]:
                for text in contexts:
                    if text_len+len(text) < 3000:
                        cur_contexts.append(text)
                        text_len += len(text)
                        used_texts += 1
                    else:
                        if text_len + len(text) - 3000 < 3000 - text_len:
                            cur_contexts.append(text)
                            text_len += len(text)
                            used_texts += 1
                            break
            if abs(text_len - 3000) > 500:
                print(f"Warning: text len exceed limitation at question {row_id}_{i}: text len {text_len} with text num {used_texts}")
                exceed_contexts_num += 1
            print(text_len)
            contexts.append(cur_contexts)
            
    print(f"Exceed limiation: {100*exceed_contexts_num/len(questions):.2f}%")
    return questions, ground_truths, contexts

def generate_answer_and_save_as_jsonl(llm_config, questions, ground_truths, contexts, save_path:str='./dataset.jsonl'):
    llm = get_llm(llm_config)
    save_file = open(save_path, 'w')
    
    # qa_prompt = old_gen_temp
    qa_prompt = Gen_Dataset_Temp.prompt_template
    parser = Gen_Dataset_Temp.parser
    
    with tqdm(total=len(questions), desc="generating answer...") as pbar:
        for id, (question, retrieved_contexts, ground_truth) in enumerate(zip(questions, contexts, ground_truths)):
            # Generate a response for each query
            retrieved_contexts_str = "".join([f" {i}. {s}\n" for i, s in enumerate(retrieved_contexts)])
            fmt_qa_prompt = qa_prompt.format(context_str=retrieved_contexts_str, query_str=question)
            answer = llm.complete(fmt_qa_prompt).text.strip()
            try:
                # Save the generated response to the JSONL file
                # data = {
                #     "question": question,
                #     "ground_truth": ground_truth,
                #     "answer": answer,
                #     "context": retrieved_contexts
                # }
                # json.dump(data, save_file)
                # save_file.write("\n")
                
                obj = parser.parse(answer)
                # Save the generated response to the JSONL file
                data = {
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": obj.Answer,
                    "context": retrieved_contexts
                }
                json.dump(data, save_file)
                save_file.write("\n")
            except:
                print(f'[skip] {id}: {question}')
                print(answer)
            pbar.update(1)
    save_file.close()

# TODO here
# def create_batch():
#     node_id = 0
#     batches = []
#     node_dict = {} # {node.id_: {'node': node, 'input_text': input_text}}
#     input_file_paths = {} # {now : input_file_path}
#     batch_info_paths = {} # {id : {"path": batch_info_path, 'now': now}}

# def generate_answers_in_batches(llm_config, questions, ground_truths, contexts, save_path: str = './dataset.jsonl', request_num: int = 45000):
#     llm = get_llm(llm_config)
#     batch_data = []
    
#     qa_prompt = Gen_Dataset_Temp.prompt_template
#     parser = Gen_Dataset_Temp.parser

#     total_batches = math.ceil(len(questions) / request_num)

#     with tqdm(total=total_batches, desc="Generating answers...") as pbar:
#         question_id = 0
#         for question, retrieved_contexts, ground_truth in zip(batch_questions, batch_contexts, batch_ground_truths):
#             # Prepare the prompt for the batch
#             if question_id % request_num == 0:
#                 now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#                 input_file_path = f"./batch_cache/{now}---batchinput.jsonl"
#                 input_file_paths[now] = input_file_path
#                 file = open(input_file_path, 'w')
#             retrieved_contexts_str = "".join([f" {i}. {s}\n" for i, s in enumerate(retrieved_contexts)])
#             fmt_qa_prompt = qa_prompt.format(context_str=retrieved_contexts_str, query_str=question)
#             batch_prompts.append((question, ground_truth, fmt_qa_prompt))

#             # Generate responses for the entire batch
#             try:
#                 responses = llm.complete_batch([prompt[2] for prompt in batch_prompts])  # Assuming llm supports batch processing
#                 for (question, ground_truth, fmt_qa_prompt), answer in zip(batch_prompts, responses):
#                     answer = answer.text.strip()  # Adjust based on your API's response format
#                     obj = parser.parse(answer)

#                     # Save the generated response to the batch data list
#                     data = {
#                         "question": question,
#                         "ground_truth": ground_truth,
#                         "answer": obj.Answer,
#                         "context": [context for context in batch_contexts]
#                     }
#                     batch_data.append(data)
#             except Exception as e:
#                 print(f"[Error] Batch {batch_id}: {e}")

#             pbar.update(1)

#     # Write all collected batch data to the JSONL file at once
#     with open(save_path, 'w') as save_file:
#         for data in batch_data:
#             json.dump(data, save_file)
#             save_file.write("\n")

if __name__ == '__main__':
    qar_file_name = 'gpt-4o-batch-all-target_extract_gpt-4o-QAExtractor-batch_pid_0.jsonl.csv' # modify each time
    qar_dataset_path = os.path.join(os.path.abspath('../../.save/gpt-4o-batch-all-target_1_parser/question'), qar_file_name)
    condition = 2
    retrieved_file_name = 'gpt-4o-batch-all-target_all-level_retrieved_contexts.jsonl' # modify each time
    retrieved_contexts_path = os.path.abspath(f'../step_4_0_retrieve_contexts/retrieved_contexts/{retrieved_file_name}')
    prefix = retrieved_file_name.split('.')[0] # sentence_splitter
    save_file_name = f"{prefix}_dataset_condition_{condition}.jsonl" # modify each time
    save_path = os.path.abspath(os.path.join('./datasets', save_file_name))
    print(f"save path: {save_path}")
    
    _, perfix_config = load_configs()
    llm_config = perfix_config['llm']['gpt-4o-mini']
    if condition == 1:
        q, g, cc = get_quetions_groundtruth_correct_contexts(qar_dataset_path)
        generate_answer_and_save_as_jsonl(llm_config, q, g, cc, save_path)
    else:
        q, g, c = get_quetions_groundtruth_contexts(qar_dataset_path, retrieved_contexts_file_path=retrieved_contexts_path)
        # input()
        generate_answer_and_save_as_jsonl(llm_config, q, g, c, save_path)