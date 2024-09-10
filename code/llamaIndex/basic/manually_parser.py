import os
import sys
"""Set up root directory and path imports."""
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
import time
import yaml
import argparse
import subprocess
import smtplib
from dotenv import load_dotenv
from custom.parser import CustomHierarchicalNodeParser
from custom.io import save_nodes_jsonl, load_nodes_jsonl


def load_configuration(root_path, config_dir_path):
    """Load YAML configuration files."""
    config_path = os.path.abspath(os.path.join(root_path, config_dir_path, 'config.yaml'))
    prefix_config_path = os.path.abspath(os.path.join(root_path, config_dir_path, 'prefix_config.yaml'))

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    with open(prefix_config_path, 'r') as prefix_config_file:
        prefix_config = yaml.safe_load(prefix_config_file)

    return config, prefix_config


def initialize_env_variables(root_path):
    """Load environment variables from .env file."""
    load_dotenv(dotenv_path=os.path.abspath(os.path.join(root_path, './code/llamaIndex/.env')))


def identify_leave_tasks(basic_name, task_cache_path):
    """Identify tasks that need to be processed."""
    leave_tasks = list(range(164))
    
    for file_name in os.listdir(task_cache_path):
        if basic_name in file_name:
            task_id = int(file_name.split('.')[0][len(basic_name):])
            leave_tasks.remove(task_id)
    
    print(f"leave task: {len(leave_tasks)}")
    print(leave_tasks)
    return leave_tasks


def split_gpus(leave_tasks):
    """Divide leave tasks into groups based on GPU availability."""
    gpus1 = [(pid, 'V100', 1) for pid in leave_tasks[:10]]
    gpus2 = [(pid, 'V100', 3) for pid in leave_tasks[10:20]]
    gpus3 = [(pid, 'V100', 4) for pid in leave_tasks[20:30]]
    gpus4 = [(pid, 'V100', 5) for pid in leave_tasks[30:40]]

    return gpus1 + gpus2 + gpus3 + gpus4


def load_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--input_file', type=str, default=None, help='The input filename to process manually parser')
    parser.add_argument('--action', type=str, default='main', help='The action to process manually parser')
    parser.add_argument('--pid', type=int, default=None, help='The PID of the subtask to process nodes')
    parser.add_argument('--gpu', type=str, default=None, help='The GPU to process manually parser')
    parser.add_argument('--node_number_per_process', type=int, default=None, help='The node number per process')

    return parser.parse_args()


def submit_job(script_path, gpu, gid, gn, pid, python_file_name, input_file, node_number_per_process, nodes_length):
    """Submit a job to Slurm and return the job ID."""
    if pid * node_number_per_process >= nodes_length:
        return None, None

    job_name = f'pid-{pid}_gpu-{gpu}_gn-{gn}_nnum-{node_number_per_process}'
    script_template = get_slurm_script(gpu, gid, gn, job_name, python_file_name, input_file, node_number_per_process)

    # Define the script path
    script_path = os.path.join(script_path, 'execute.sh')

    # Write the script to a file
    with open(script_path, 'w') as file:
        file.write(script_template)

    # Make the script executable
    os.chmod(script_path, 0o755)

    return execute_job(script_path, pid, job_name, input_file, node_number_per_process, gpu)


def get_slurm_script(gpu, gid, gn, job_name, python_file_name, input_file, node_number_per_process):
    """Return the SLURM job script template based on GPU type."""
    if gpu == 'L40':
        return f"""#!/bin/bash
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --job-name={job_name}
#SBATCH --qos=low-gpu
#SBATCH --time=72:00:00
#SBATCH --output=slurm-{job_name}.out
#SBATCH --gres=gpu:{gpu}:{gn}
#SBATCH --nodelist=gpu-{gpu}-4-{gid}

source ~/.bashrc
conda activate llm
python {python_file_name} --input_file {input_file} --action thread --pid {pid} --gpu {gpu} --node_number_per_process {node_number_per_process}
"""
    elif gpu == 'V100':
        return f"""#!/bin/bash
#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --job-name={job_name}
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --output=slurm-{job_name}.out
#SBATCH --gres=gpu:{gpu}:{gn}
#SBATCH --nodelist=gpu-1-{gid}

source ~/.bashrc
conda activate llm
python {python_file_name} --input_file {input_file} --action thread --pid {pid} --gpu {gpu} --node_number_per_process {node_number_per_process}
"""


def execute_job(script_path, pid, job_name, input_file, node_number_per_process, gpu):
    """Execute the job submission via sbatch."""
    result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)

    if result.returncode == 0:
        job_id = result.stdout.split()[-1]
        print(f"[Job ID: {job_id}] Job submitted successfully with pid {pid} at gpu {gpu}!")
        save_cache_name = generate_cache_name(input_file, gpu, node_number_per_process, pid)
        return job_name, save_cache_name

    handle_job_submission_error(result, job_name, gpu, pid, script_path, input_file, node_number_per_process)
    return None, None


def generate_cache_name(input_file, gpu, node_number_per_process, pid):
    """Generate a cache name based on input file, GPU, and node information."""
    return '_'.join(input_file.split('_')[:-1] + [f'gpu_{gpu}_nodeNum_{node_number_per_process}_pid_{pid}.jsonl'])


def handle_job_submission_error(result, job_name, gpu, pid, script_path, input_file, node_number_per_process):
    """Handle the resubmission process if job submission fails."""
    print("Error submitting job:")
    print(result.stderr)
    for _ in range(10):
        print(f'Resubmitting job with pid {pid} at gpu {gpu}')
        result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
        if result.returncode == 0:
            return


def monitor_and_restart(jobs, script_path, check_interval=60, is_restart=True):
    """Monitor the job and restart it if it fails."""
    while len(jobs) > 0:
        unfinished = []
        for job_name, save_cache_name in jobs:
            status = get_job_status(job_name)
            print(f"Job '{job_name}' status: {status}")

            save_cache_file_path = os.path.join(TASK_CACHE_PATH, save_cache_name)
            if is_job_finished(save_cache_file_path):
                continue
            elif is_job_failed(status):
                handle_failed_job(job_name, save_cache_file_path, script_path, is_restart)

            unfinished.append((job_name, save_cache_name))

        jobs = unfinished
        time.sleep(check_interval)


def is_job_finished(save_cache_file_path):
    """Check if the job is finished by verifying the cache file's existence."""
    return os.path.exists(save_cache_file_path)


def is_job_failed(status):
    """Check if the job status indicates a failure."""
    return status in ["FAILED", "NODE_FAIL", '']


def handle_failed_job(job_name, save_cache_file_path, script_path, is_restart):
    """Handle the process when a job fails, including possible resubmission."""
    time.sleep(30)
    if not os.path.exists(save_cache_file_path):
        print(f"Job '{job_name}' failed. Restarting...")
        job_dict = parse_job_name(job_name)
        notify_failure(job_name, job_dict)
        if is_restart:
            resubmit_failed_job(script_path, job_dict)


def parse_job_name(job_name):
    """Parse the job name to extract job parameters."""
    return {attr.split('-')[0]: attr.split('-')[1] for attr in job_name.split('_')}


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
