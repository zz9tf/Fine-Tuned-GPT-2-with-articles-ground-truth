import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import math
import time
import argparse
import smtplib
import subprocess
from configs.load_config import load_configs
from utils.generate_and_execute_slurm_job import generate_and_execute_slurm_job
from component.parser.multiple_abstract_level_node_parser import MultipleAbstractLevelNodeParser
from component.io import save_nodes_jsonl, load_nodes_jsonl

def load_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--input_file', type=str, default=None, help='The input filename to process manually parser')
    parser.add_argument('--action', type=str, default='main', help='The action to process manually parser')
    parser.add_argument('--pid', type=int, default=None, help='The PID of the subtask to process nodes')

    return parser.parse_args()

def submit_job(
    script_path: str, 
    gpu: str,
    gn: str,
    pid: int,
    python_file_name: str,
    input_file: str
):
    """Submit a job to Slurm and return the job ID."""
    print(f"[PID: {pid}] is submitting with gpu {gpu}!")
    job_name = f'parse-{pid}'
    log_file_path = os.path.abspath(os.path.join(script_path, 'out/{job_name}.out'))
    script_path = os.path.abspath(os.path.join(script_path, 'execute/execute.sh'))
    generate_and_execute_slurm_job(
        python_start_script=f"{python_file_name} --input_file {input_file} --action thread --pid {pid}",
        job_name=job_name,
        gpu=gpu,
        num=gn,
        log_file_path=log_file_path,
        script_path=script_path
    )

def one_thread_parser(
        input_file_path: str,
        cache_path: str,
        pid: int,
        llm_config: dict,
        embedding_config: dict
):
    # Load nodes
    nodes = load_nodes_jsonl(input_file_path)

    # Load parser
    parser = MultipleAbstractLevelNodeParser.from_defaults(
        llm_config=llm_config,
        embedding_config=embedding_config,
        cache_dir_path=cache_path,
        cache_file_name=f'pid_cache_{pid}_not_finish.jsonl'
    )
    nodes = parser.get_nodes_from_documents(nodes, show_progress=True)

    # Save nodes
    nodes_cache_path = os.path.join(cache_path, f'parsed_chunk_{pid}.jsonl')
    save_nodes_jsonl(nodes_cache_path, nodes)

# def merge(cache_path):
#     prefix_file_name = ""
#     save_files = [file_name for file_name in os.listdir(cache_path) if prefix_file_name in file_name]
#     all_nodes = []
#     for save_cache_name in save_files:
#         nodes_cache_path = os.path.abspath(os.path.join(cache_path, save_cache_name))
#         all_nodes.extend(load_nodes_jsonl(nodes_cache_path))
        
#     cache_name = f".jsonl"
#     nodes_cache_path = os.path.join(cache_path, cache_name)
#     save_nodes_jsonl(nodes_cache_path, all_nodes)

if __name__ == '__main__':
    # prefix parameters
    input_dir = '../.save/finished_chunk' # modify this each time
    cache_path = '../.cache'
    python_file_name = 'step_2_parse.py'
    notIncludeNotFinishJsonl = True
    gn = 1 # gpu number
    
    args = load_args()
    if args.action == 'main':
        # calculate finished and leaving tasks
        filenames = {int(filename.split('_')[-1].split('.')[0]): filename for filename in os.listdir(input_dir) if 'finished_chunk' in filename}
        
        prefix_name = 'parsed_chunk'
        for filename in os.listdir(cache_path):
            if prefix_name in filename:
                task_id = int(filename.split('_')[-1].split(".")[0])
                if ('not_finish' not in filename or notIncludeNotFinishJsonl) and task_id in filenames:
                    del filenames[task_id]
        
        print(f"leave task: {len(filenames)}")
        
        gpus = sorted([(pid, filename, 'V100') for pid, filename in filenames.items()], key=lambda x: x[0])
        
        for pid, filename, gpu in gpus:
            submit_job(
                script_path=os.getcwd(),
                gpu=gpu,
                gn=gn,
                pid=pid,
                python_file_name=python_file_name,
                input_file=filename
            )
        
    elif args.action == 'thread':
        # Load configs
        config, prefix_config = load_configs()
        parser_config = prefix_config['parser']['MALNodeParser-hf_vicuna_13b']
        llm_config = prefix_config['llm'][parser_config['llm']]
        embedding_config = prefix_config['embedding_model'][parser_config['embedding_model']]
        
        one_thread_parser(
            input_file_path=os.path.join(input_dir, args.input_file),
            cache_path=cache_path,
            pid=args.pid,
            llm_config=llm_config,
            embedding_config=embedding_config
        )
        
    # elif args.action == 'merge':
    #     merge(cache_path)
