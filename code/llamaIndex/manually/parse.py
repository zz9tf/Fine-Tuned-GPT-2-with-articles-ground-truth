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
from component.parser.custom_hierarchical_node_parser import CustomHierarchicalNodeParser
from component.io import save_nodes_jsonl, load_nodes_jsonl

def load_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--input_file', type=str, default=None, help='The input filename to process manually parser')
    parser.add_argument('--action', type=str, default='main', help='The action to process manually parser')
    parser.add_argument('--pid', type=int, default=None, help='The PID of the subtask to process nodes')
    parser.add_argument('--gpu', type=str, default=None, help='The GPU to process manually parser')
    parser.add_argument('--node_number_per_process', type=int, default=None, help='The node number per process')

    return parser.parse_args()

def submit_job(
    script_path: str, 
    gpu: str,
    gid: str,
    gn: str,
    pid: int,
    python_file_name: str,
    input_file: str,
    node_number_per_process: str,
    nodes_number: int
):
    """Submit a job to Slurm and return the job ID."""
    if pid * node_number_per_process >= nodes_number:
        return None, None
    print(f"[PID: {pid}] is submitting with gpu {gpu}!")
    job_name = f'pid-{pid}_gpu-{gpu}_gn-{gn}_nnum-{node_number_per_process}'
    script_path = os.path.abspath(os.path.join(script_path, 'execute/execute.sh'))
    print(script_path)
    exit()
    job_name = generate_and_execute_slurm_script(gpu, gid, gn, job_name, python_file_name, input_file, node_number_per_process, script_path)
    # Is submitting successful?
    if job_name == None:
        return None, None
    save_file = generate_save_file_name(input_file, gpu, node_number_per_process, pid)
    return job_name, save_file

def generate_and_execute_slurm_script(
    gpu, 
    gn, 
    job_name, 
    python_file_name, 
    input_file, 
    node_number_per_process, 
    script_path
):
    """Return the SLURM job script template based on GPU type."""
    if gpu == 'L40':
        generate_and_execute_slurm_job(
            python_start_script=f"{python_file_name} --input_file {input_file} --action thread --pid {pid} --gpu {gpu} --node_number_per_process {node_number_per_process}",
            account='guest',
            partition='guest-gpu',
            job_name={job_name},
            qos='low-gpu',
            time='24:00:00',
            gpu=gpu,
            num=gn,
            script_path=script_path
        )
    elif gpu == 'V100':
        generate_and_execute_slurm_job(
            python_start_script=f"{python_file_name} --input_file {input_file} --action thread --pid {pid} --gpu {gpu} --node_number_per_process {node_number_per_process}",
            job_name=job_name,
            gpu=gpu,
            num=gn
        )

def generate_save_file_name(input_file, gpu, node_number_per_process, pid):
    """Generate a cache name based on input file, GPU, and node information."""
    return '_'.join(input_file.split('_')[:-1] + [f'gpu_{gpu}_nodeNum_{node_number_per_process}_pid_{pid}.jsonl'])

def is_job_finished(save_file_path):
    """Check if the job is finished by verifying the cache file's existence."""
    return os.path.exists(save_file_path)

def is_job_failed(status):
    """Check if the job status indicates a failure."""
    return status in ["FAILED", "NODE_FAIL", '']

def parse_job_name(job_name):
    """Parse the job name to extract job parameters."""
    return {attr.split('-')[0]: attr.split('-')[1] for attr in job_name.split('_')}

def send_email(sent_from, app_password, sent_to, subject="", email_body=""):

    smtpserver = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    smtpserver.ehlo()
    smtpserver.login(sent_from, app_password)

    # Test send mail
    smtpserver.sendmail(sent_from, sent_to, "Subject: {}\n\n{}".format(subject, email_body).encode('utf-8'))

    # Close the connection
    smtpserver.close()
    print(f"[Send email] sent {subject}")

def notify_failure(job_name, job_dict):
    """Send an email notification when a job fails."""
    slurm_out_file = f'slurm-{job_dict["pid"]}.out'
    if os.path.exists(slurm_out_file):
        with open(slurm_out_file, 'r') as out_file:
            send_email(
                sent_from="zhengzheng@brandeis.edu",
                app_password="your_password",
                sent_to="zhengzheng@brandeis.edu",
                subject=f"Failed {job_name}",
                email_body=out_file.read()
            )

def resubmit_failed_job(script_path, job_dict):
    job_name, save_file_name = submit_job(
        script_path=script_path,
        gpu=job_dict['gpu'],
        gn=int(job_dict['gn']),
        pid=int(job_dict['pid'])
    )
    return job_name, save_file_name

def handle_failed_job(job_name, save_file_path, script_path, is_restart):
    """Handle the process when a job fails, including possible resubmission."""
    time.sleep(30)
    if not is_job_finished(save_file_path):
        print(f"Job '{job_name}' failed...")
        job_dict = parse_job_name(job_name)
        notify_failure(job_name, job_dict)
        if is_restart:
            return resubmit_failed_job(script_path, job_dict)
    return None, None

def get_job_status(job_name):
    """Get the status of a job by its name."""
    try:
        result = subprocess.run(
            ['squeue', '--name', job_name, '--format=%T', '--noheader'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error checking job status: {e}")
        return None

def monitor_and_restart(
    jobs, script_path, cache_path, check_interval=60, is_restart=True
):
    """Monitor the job and restart it if it fails."""
    while len(jobs) > 0:
        unfinished = []
        for job_name, save_file in jobs:
            status = get_job_status(job_name)
            print(f"Job '{job_name}' status: {status}")

            save_file_path = os.path.join(cache_path, save_file)
            if is_job_finished(save_file_path):
                continue
            elif is_job_failed(status, save_file_path):
                job_name, save_file = handle_failed_job(job_name, save_file_path, script_path, is_restart)
                if job_name != None:
                    unfinished.append((job_name, save_file))
            else:
                unfinished.append((job_name, save_file))
        print()
        jobs = unfinished
        time.sleep(check_interval)

def one_thread_parser(
        input_file_name: str,
        cache_path: str,
        gpu: str,
        pid: int,
        node_number_per_process: int
):
    # Load nodes
    nodes_cache_path = os.path.abspath(os.path.join(cache_path, input_file_name))
    nodes = load_nodes_jsonl(nodes_cache_path)[pid*node_number_per_process: (pid+1)*node_number_per_process]

    # Load parser
    parser = CustomHierarchicalNodeParser.from_defaults(
        llm_config=llm_config,
        embedding_config=embedding_config,
        cache_dir_path=cache_path,
        cache_file_name=f'pid-{pid}.jsonl'
    )
    nodes = parser.get_nodes_from_documents(nodes, show_progress=True)

    # Save nodes
    save_file_name = generate_save_file_name(input_file, gpu, node_number_per_process)
    nodes_cache_path = os.path.abspath(os.path.join(cache_path, save_file_name))
    print(nodes_cache_path)
    save_nodes_jsonl(nodes_cache_path, nodes)

if __name__ == '__main__':
    config, prefix_config = load_configs()
    parser_config = prefix_config['parser']['ManuallyHierarchicalNodeParser']
    llm_config = prefix_config['llm'][parser_config['llm']]
    embedding_config = prefix_config['embedding_model'][parser_config['embedding_model']]
    # prefix parameters
    input_file = 'gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_processing.jsonl'
    python_file_name = 'parse.py'
    gn = 1 # gpu number
    node_number_per_process=50
    check_interval = 2
    
    nodes_number = int(input_file.split('_')[-2])
    leave_tasks = math.ceil(nodes_number/node_number_per_process)
    leave_tasks = list(range(leave_tasks))
    cache_path = os.path.abspath(os.path.join('..', config['cache']))
    print(f"cache_path: {cache_path}")
    basic_name = "_".join(input_file.split('_')[:-1])
    print(f"basic_name: {basic_name}")
    for file_name in os.listdir(cache_path):
        if basic_name in file_name:
            task_id = int(file_name.split('.')[0][len(basic_name):])
            leave_tasks.remove(task_id)
    print(f"leave task: {len(leave_tasks)}")
    print(leave_tasks)
    
    args = load_args()
    if args.action == 'main':
        jobs = []
        gpus1 = [(pid, 'V100', 3) for pid in leave_tasks[:10]]
        gpus2 = [(pid, 'V100', 2) for pid in leave_tasks[10:14]]
        gpus3 = [(pid, 'V100', 4) for pid in leave_tasks[20:30]]
        gpus4 = [(pid, 'V100', 5) for pid in leave_tasks[30:40]]

        gpus = gpus1 + gpus2 + gpus3 + gpus4
        
        for (pid, gpu, gid) in gpus:
            job_name, save_file = submit_job(
                script_path=os.getcwd(),
                gpu=gpu,
                gid=gid,
                gn=gn,
                pid=pid,
                python_file_name=python_file_name,
                input_file=input_file,
                node_number_per_process=node_number_per_process,
                nodes_number=nodes_number
            )
            if job_name != None:
                jobs.append((job_name, save_file))

        monitor_and_restart(
            jobs=jobs,
            script_path=os.getcwd(),
            cache_path=cache_path,
            check_interval=check_interval,
            is_restart=False
        )
        
    elif args.action == 'thread':
        one_thread_parser(
            input_file_name=args.input_file,
            cache_path=cache_path,
            pid=args.pid,
            gpu=args.gpu,
            node_number_per_process=args.node_number_per_process
        )
        
    elif args.action == 'merge':
        prefix_file_name = "gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_50"
        save_files = [file_name for file_name in os.listdir(cache_path) if prefix_file_name in file_name]
        all_nodes = []
        for save_cache_name in save_files:
            nodes_cache_path = os.path.abspath(os.path.join(cache_path, save_cache_name))
            all_nodes.extend(load_nodes_jsonl(nodes_cache_path))
            
        cache_name = f"gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_{len(all_nodes)}.jsonl"
        nodes_cache_path = os.path.join(cache_path, cache_name)
        save_nodes_jsonl(nodes_cache_path, all_nodes)
    
    
    