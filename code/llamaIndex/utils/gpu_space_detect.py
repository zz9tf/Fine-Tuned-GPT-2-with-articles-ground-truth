import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
import subprocess
import torch
from transformers import BitsAndBytesConfig
from custom.custom_huggingfacellm import CustomHuggingFaceLLM
import concurrent.futures

# Function to get GPU memory information
def get_gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    gpu_memory = result.stdout.decode('utf-8').strip().split('\n')
    gpu_memory_info = []
    for info in gpu_memory:
        total, used, free = map(int, info.split(','))
        gpu_memory_info.append({
            'total': total,
            'used': used,
            'free': free
        })
    return gpu_memory_info

# Function to set up LLM on a specific GPU
def get_llm(gpu_id):
    llm_config = {'model_name': 'lmsys/vicuna-13b-v1.5', 'cache_dir': '/work/zhengzheng/.hf_cache'}
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    return CustomHuggingFaceLLM(
        model_name=llm_config['model_name'],
        model_kwargs={
            "quantization_config": quantization_config,
            "cache_dir": llm_config['cache_dir'],
            "local_files_only": True
        },
        tokenizer_name=llm_config['model_name'],
        tokenizer_kwargs={
            "cache_dir": llm_config['cache_dir'],
            "local_files_only": True
        },
        query_wrapper_prompt="System: You are an advanced language model designed to provide expert, high-quality responses. Your task is to understand the user's input and generate an appropriate response.\nUser: {query_str}\nResponse:",
        max_new_tokens=4096,
        generate_kwargs={'do_sample': True, "temperature": 0.3, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.2},
        device_map=f"cuda:{gpu_id}"
    )

# Initialize LLMs on different GPUs concurrently
def initialize_llms():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(torch.cuda.device_count()):
            print(f"Initializing LLM on GPU {i}")
            futures.append(executor.submit(get_llm, i))
        llms = [future.result() for future in concurrent.futures.as_completed(futures)]
    return llms

def one_response(i, llm):
    print(f"One response started at {i}")
    result = f"test: {llm.complete("Test").text}"
    print(f"Response completed at {i}")
    return result

# Retrieve GPU memory information
gpu_memory_info = get_gpu_memory()
for idx, info in enumerate(gpu_memory_info):
    print(f"GPU {idx}: Total Memory: {info['total']}MB, Used Memory: {info['used']}MB, Free Memory: {info['free']}MB")

# Initialize LLMs
# llms = initialize_llms()
llms = []
for i in range(torch.cuda.device_count()):
    llms.append(get_llm(i))

# Print out the initialized LLMs for confirmation
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for i, llm in enumerate(llms):
        futures.append(executor.submit(one_response, i, llm))
    for future in concurrent.futures.as_completed(futures):
        print(future.result())