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

def generate_and_execute_slurm_script(
    gpu, 
    gn, 
    job_name, 
    python_file_name, 
    input_file_name,
    script_path,
    log_file_path,
    account=None
):
    """Return the SLURM job script template based on GPU type."""
    if gpu == 'L40':
        return generate_and_execute_slurm_job(
            python_start_script=f"{python_file_name} --action thread --pid {pid} --input_file_name {input_file_name}",
            account='guest',
            partition='guest-gpu',
            job_name={job_name},
            qos='low-gpu',
            time='24:00:00',
            gpu=gpu,
            num=gn,
            log_file_path=log_file_path,
            script_path=script_path
        )
    elif gpu == 'V100' and account == 'guest':
        return generate_and_execute_slurm_job(
            python_start_script=f"{python_file_name} --action thread --pid {pid} --input_file_name {input_file_name}",
            account='guest',
            partition='guest-gpu',
            job_name=job_name,
            qos='low-gpu',
            time='24:00:00',
            gpu=gpu,
            num=gn,
            log_file_path=log_file_path,
            script_path=script_path
        )
    elif gpu == 'V100' and account == 'pengyu-lab':
        return generate_and_execute_slurm_job(
            python_start_script=f"{python_file_name} --action thread --pid {pid} --input_file_name {input_file_name}",
            job_name=job_name,
            gpu=gpu,
            num=gn,
            log_file_path=log_file_path,
            script_path=script_path
        )

def submit_job(
    script_path: str, 
    gpu: str,
    gn: str,
    pid: int,
    python_file_name: str,
    input_file: str,
    account=None
):
    """Submit a job to Slurm and return the job ID."""
    job_name = f'store_pid-{pid}_account-{account}_gpu-{gpu}_gn-{gn}'
    log_file_path = os.path.abspath(os.path.join(script_path, 'out/{job_name}.out'))
    script_path = os.path.abspath(os.path.join(script_path, 'execute/execute.sh'))
    job_name = generate_and_execute_slurm_script(
        gpu=gpu, 
        gn=gn,
        job_name=job_name, 
        python_file_name=python_file_name, 
        input_file_name=input_file,
        script_path=script_path,
        log_file_path=log_file_path,
        account=account
    )
    print(f"[PID: {pid}] is submitted with gpu {gpu}!")
    return job_name

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
        index_id=index_id,
        device='cuda:1'
    )
    
if __name__ == "__main__":
    # Load config
    total_config, prefix_config = load_configs()
    config = prefix_config['storage']['simple']
    embedding_config = prefix_config['embedding_model'][config['embedding_model']]
    root_path = "../../.."
    cache_dir = "../.save/gpt-4o-batch-all-target_1_parser/sub"
    print(f"cache_dir: {cache_dir}")
    
    # prefix parameters
    input_file_base = 'gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_50_pid_'
    python_file_name = 'step_4_store_and_index.py'
    index_name = total_config['document_preprocessing']['index_pipelines'][0]
    index_id = 'all'
    gn = str(2) # gpu number
    check_interval = 2 # sconds of interval to check tasks
    notIncludeNotFinishJsonl = True
    
    # calculate finished and leaving tasks
    index_dir_path = os.path.abspath(os.path.join(root_path, total_config['indexes_dir_path'], "gpt-4o-batch-all-target"))
    
    args = load_args()
    if args.action == "merge":
        merge_database_pid_nodes(
            index_dir_path=index_dir_path,
            index_id=index_id
        )
    else:
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
        gpus = gpus[:30]
    
        if args.action == 'main':
            for i, (pid, input_file, gpu) in enumerate(gpus):
                job_name = submit_job(
                    script_path=os.getcwd(),
                    gpu=gpu,
                    gn=gn,
                    pid=pid,
                    python_file_name=python_file_name,
                    input_file=input_file,
                    account='guest' if i >= 25 else "pengyu-lab"
                )
                
        elif args.action == 'thread':
            one_thread_store(
                embedding_config=embedding_config,
                input_file_path=os.path.abspath(os.path.join(cache_dir, args.input_file_name)),
                index_dir_path=index_dir_path,
                index_id=str(args.pid),
            )