import os
import json
from tqdm import tqdm
from configs.load_config import load_configs
from component.models.llm.get_llm import get_llm
from component.schema import Gen_Dataset_Temp

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