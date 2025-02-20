import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import re
import argparse
from configs.load_config import load_configs
from utils.generate_and_execute_slurm_job import generate_and_execute_slurm_job
from component.index.index import storing_nodes_for_index, merge_database_pid_nodes

def load_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--action', type=str, default='main', help='The action to process manually storage')
    parser.add_argument('--pid', type=int, default=None, help='The PID of the subtask to process storage')
    parser.add_argument('--input_file_name', type=str, default=None, help='The input filename to process storage')

    return parser.parse_args()

def submit_job(
    gpu, 
    gn, 
    pid,
    python_file_name, 
    input_file_name,
    job_name
):
    """Return the SLURM job script template based on GPU type."""
    generate_and_execute_slurm_job(
        script_path='./execute/execute.sh',
        log_file_path=f"./out/{job_name}_{pid}.out",
        gpu=gpu,
        num=gn,
        python_start_script=f"{python_file_name} --action thread --pid {pid} --input_file_name {input_file_name}",
        job_name=job_name
    )
    print(f"[PID: {pid}] is submitted with gpu {gpu}!")

def one_thread_store(
    embedding_config,
    input_file_path: str,
    index_dir_path: str,
    index_id: str
    ):
    storing_nodes_for_index(
        embedding_config=embedding_config, 
        input_file_path=input_file_path,
        index_dir_path=index_dir_path,
        index_id=index_id
        # device='cuda:1'
    )
    
if __name__ == "__main__":
    # Load config
    total_config, prefix_config = load_configs()
    config = prefix_config['storage']['wikipedia'] # modify this each time
    embedding_config = prefix_config['embedding_model'][config['embedding_model']]
    root_path = "../../.."
    cache_dir = "../.cache"
    print(f"cache_dir: {cache_dir}")
    
    # prefix parameters
    input_file_base = 'parsed_chunk_'
    python_file_name = 'step_4_store_and_index.py'
    index_name = 'wikipedia-mal-rag' # modify this each time
    index_id = 'all'
    gn = str(1) # gpu number
    notIncludeNotFinishJsonl = True
    
    # calculate finished and leaving tasks
    index_dir_path = os.path.abspath(os.path.join(root_path, total_config['indexes_dir_path'], index_name))
    
    args = load_args()
    if args.action == "merge":
        merge_database_pid_nodes(
            index_dir_path=index_dir_path,
            index_id=index_id
        )
    else:
        if args.action == 'main':
            leave_tasks = {}
            for filename in os.listdir(cache_dir):
                if filename.startswith(input_file_base):
                    pid = filename.split('.')[0].split('_')[-1]
                    leave_tasks[pid] = filename
            pattern = r'(\d+)\.jsonl'
            for filename in os.listdir(index_dir_path):
                match = re.search(pattern, filename)
                if match:
                    task_id = match.group(1)
                    del leave_tasks[task_id]
                else:
                    match = re.search(r'(\d+)_not_finish\.jsonl', filename)
                    if not match:
                        print(f"[Not pid file] {filename}")
                        
                    elif notIncludeNotFinishJsonl:
                        task_id = match.group(1)
                        del leave_tasks[task_id]
            print(f"leave task: {len(leave_tasks)}")
            
            gpus = [(k, v, 'V100') for k, v in leave_tasks.items()]
            gpus.sort(key=lambda x: int(x[0]))
            
            for i, (pid, input_file_name, gpu) in enumerate(gpus):
                job_name = submit_job(
                    gpu=gpu,
                    gn=gn,
                    pid=pid,
                    python_file_name=python_file_name,
                    input_file_name=input_file_name,
                    job_name=index_name
                )
                
        elif args.action == 'thread':
            one_thread_store(
                embedding_config=embedding_config,
                input_file_path=os.path.abspath(os.path.join(cache_dir, args.input_file_name)),
                index_dir_path=index_dir_path,
                index_id=str(args.pid),
            )