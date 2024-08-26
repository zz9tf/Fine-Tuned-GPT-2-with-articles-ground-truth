import os, sys
import time
import smtplib
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
import yaml
import argparse
import subprocess
from dotenv import load_dotenv
from custom.parser import CustomHierarchicalNodeParser
from custom.io import save_nodes_jsonl, load_nodes_jsonl

# Set paramters
input_file = 'gpt-4o-batch-all-p_2_parser_ManuallyHierarchicalNodeParser_8165_processing.jsonl'
python_file_name = 'manually_parser_exe.py'

leave_tasks = list(range(164))
task_cache_path = "/home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/.cache"
basic_name = "gpt-4o-batch-all-p_2_parser_ManuallyHierarchicalNodeParser_8165_gpu_V100_nodeNum_50_pid_"
for file_name in os.listdir(task_cache_path):
    if basic_name in file_name:
        task_id = int(file_name.split('.')[0][len(basic_name):])
        leave_tasks.remove(task_id)

gpus1 = [(pid, 'V100', 1) for pid in leave_tasks[:4]]
gpus2 = [(pid, 'V100', 2) for pid in leave_tasks[4:8]]
gpus3 = [(pid, 'V100', 3) for pid in leave_tasks[8:16]]
gpus4 = [(pid, 'V100', 4) for pid in leave_tasks[16:20]]

# gpus = gpus1 + gpus2 + gpus3 + gpus4
gpus = gpus3
gn = 2
nodes_length = int(input_file.split('_')[-2])
# node_number_per_process=max(math.ceil(nodes_length/len(gpus)), 1)
node_number_per_process=50
check_interval = 5

# Load config
root_path = '../../..'
config_dir_path='./code/llamaIndex/configs'
config_path = os.path.abspath(os.path.join(root_path, config_dir_path, 'config.yaml'))
prefix_config_path = os.path.abspath(os.path.join(root_path, config_dir_path, 'prefix_config.yaml'))
with open(config_path, 'r') as config:
    config = yaml.safe_load(config)
with open(prefix_config_path, 'r') as prefix_config:
    prefix_config = yaml.safe_load(prefix_config)
load_dotenv(dotenv_path=os.path.abspath(os.path.join(root_path, './code/llamaIndex/.env')))

cache_path = os.path.abspath(os.path.join(root_path, config['cache']))
parser_config = prefix_config['parser']['ManuallyHierarchicalNodeParser']
llm_config = prefix_config['llm'][parser_config['llm']]
embedding_config = prefix_config['embedding_model'][parser_config['embedding_model']]

def load_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--input_file', type=str, default=None,
                        help='The intput filename to process manually parser')
    parser.add_argument('--action', type=str, default='main',
                        help='The action to process manually parser')
    parser.add_argument('--pid', type=int, default=None,
                        help='The pid of the subtask to process nodes')
    parser.add_argument('--gpu', type=str, default=None,
                        help='The gpu to process manually parser')
    parser.add_argument('--node_number_per_process', type=int, default=None,
                        help='The node number per process')
    args = parser.parse_args()

    return args

# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=zhengzheng@brandeis.edu
def submit_job(
        script_path: str, gpu: str, gid: int, gn: int, pid: int,
    ):
    """Submit a job to Slurm and return the job ID."""
    if pid*node_number_per_process >= nodes_length:
        return None, None
    job_name = f'pid-{pid}_gpu-{gpu}_gn-{gn}_nnum-{node_number_per_process}'

    if gpu == 'L40':
        script_template = f"""#!/bin/bash

#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --job-name={job_name}
#SBATCH --qos=low-gpu
#SBATCH --time=72:00:00
#SBATCH --output=slurm-{pid}.out
#SBATCH --gres=gpu:{gpu}:{gn}
#SBATCH --nodelist=gpu-{gpu}-4-{gid}

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python {python_file_name} --input_file {input_file} --action thread --pid {pid} --gpu {gpu} --node_number_per_process {node_number_per_process}
"""
    elif gpu == 'V100':
        script_template = f"""#!/bin/bash

#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --job-name={job_name}
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --output=slurm-{pid}.out
#SBATCH --gres=gpu:{gpu}:{gn}
#SBATCH --nodelist=gpu-1-{gid}


# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python {python_file_name} --input_file {input_file} --action thread --pid {pid} --gpu {gpu} --node_number_per_process {node_number_per_process}
    """
    elif gpu in ['TitanXP', 'RTX2']:
        script_template = f"""#!/bin/bash

#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --job-name={job_name}
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --output=slurm-{pid}.out
#SBATCH --gres=gpu:{gpu}:{gn}


# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python {python_file_name} --input_file {input_file} --action thread --pid {pid} --gpu {gpu} --node_number_per_process {node_number_per_process}
    """
    # Define the script path
    script_path = os.path.join(script_path, 'execute.sh')
    
    # Write the script to a file
    with open(script_path, 'w') as file:
        file.write(script_template)

    # Make the script executable
    os.chmod(script_path, 0o755)
    
    # Submit the script using sbatch
    result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.split()[-1]
        print(f"[Job ID: {job_id}] Job submitted successfully with pid {pid} at gpu {gpu}!")
        save_cache_name = '_'.join(input_file.split('_')[:-1] + [f'gpu_{gpu}_nodeNum_{node_number_per_process}_pid_{pid}.jsonl'])
        return job_name, save_cache_name

    else:
        print("Error submitting job:")
        print(result.stderr)
        i = 0
        # Is resubmitting successful?
        while result.returncode != 0 and i < 10:
            print(f'Resubmit job with pid {pid} at gpu {gpu}')
            result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
            i += 1
        
        # No
        if result.returncode != 0:
            print(f"Unable to submit job with pid {pid} at gpu {gpu}.")
            return None, None
        
        # Yes, return job id
        save_cache_name = '_'.join(input_file.split('_')[:-1] + [f'gpu_{gpu}_nodeNum_{node_number_per_process}_pid_{pid}.jsonl'])
        return job_name, save_cache_name

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

def send_email(sent_from, app_password, sent_to, subject="", email_body=""):

    smtpserver = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    smtpserver.ehlo()
    smtpserver.login(sent_from, app_password)

    # Test send mail
    smtpserver.sendmail(sent_from, sent_to, "Subject: {}\n\n{}".format(subject, email_body).encode('utf-8'))

    # Close the connection
    smtpserver.close()
    print(f"[Send email] sent {subject}")

def monitor_and_restart(jobs, script_path, check_interval=60, is_restart: bool=True):
    """Monitor the job and restart it if it fails."""
    while len(jobs) > 0:
        unfinished = []
        for job_name, save_cache_name in jobs:
            status = get_job_status(job_name)
            print(f"Job '{job_name}' status: {status}")
            save_cache_file_path = os.path.join(cache_path, save_cache_name)
            
            # Is job finished ?
            if os.path.exists(save_cache_file_path):
                continue
            # Is job failed ?
            elif status in ["FAILED", "NODE_FAIL", '']:
                time.sleep(30)
                if not os.path.exists(save_cache_file_path):
                    print(save_cache_file_path)
                    print(f"Job '{job_name}' failed. Restarting...")
                    job_dict = {attri.split('-')[0]:attri.split('-')[1] for attri in job_name.split('_')}
                    slurm_out_file = f'slurm-{job_dict["pid"]}.out'
                    if os.path.exists(slurm_out_file):
                        with open(slurm_out_file, 'r') as out_file:
                            send_email(
                                sent_from="zhengzheng@brandeis.edu",
                                app_password="iwim krcr teda zpuo",
                                sent_to="zhengzheng@brandeis.edu",
                                subject=f"Failed {job_name}",
                                email_body=out_file.read()
                            )
                        os.remove(slurm_out_file)
                    if is_restart:
                        job_name, save_cache_name = submit_job(
                            script_path=script_path,
                            gpu=job_dict['gpu'],
                            gn=int(job_dict['gn']),
                            pid=int(job_dict['pid'])
                        )
                        # Is restart successful?
                        if job_name == None:
                            continue
                    else:
                        continue
            unfinished.append((job_name, save_cache_name))
        print()
        jobs = unfinished
        time.sleep(check_interval)

def one_thread_parser(
        input_file_name: str,
        gpu: str,
        pid: int 
):
    # Load nodes
    nodes_cache_path = os.path.abspath(os.path.join(cache_path, input_file_name))
    nodes = load_nodes_jsonl(nodes_cache_path)[pid*node_number_per_process: (pid+1)*node_number_per_process]

    # Load parser
    parser = CustomHierarchicalNodeParser.from_defaults(
        llm_self=None,
        llm_config=llm_config,
        embedding_config=embedding_config,
        cache_dir_path='/home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/.cache',
        cache_file_name=f'pid-{pid}.jsonl'
    )
    nodes = parser.get_nodes_from_documents(nodes, show_progress=True)

    # Save nodes
    save_cache_name = '_'.join(input_file.split('_')[:-1] + [f'gpu_{gpu}_nodeNum_{node_number_per_process}_pid_{pid}.jsonl'])
    nodes_cache_path = os.path.abspath(os.path.join(cache_path, save_cache_name))
    print(nodes_cache_path)
    save_nodes_jsonl(nodes_cache_path, nodes)

if __name__ == '__main__':
    args = load_args()
    if args.action == 'main':
        jobs = []
        for (pid, gpu, gid) in gpus:
            job_name, save_cache_name = submit_job(
                script_path=os.getcwd(),
                gpu=gpu,
                gid=gid,
                gn=gn,
                pid=pid
            )
            if job_name != None:
                jobs.append((job_name, save_cache_name))

        monitor_and_restart(
            jobs=jobs,
            script_path=os.getcwd(),
            check_interval=check_interval,
            is_restart=False
        )

    elif args.action == 'thread':
        one_thread_parser(
            pid=args.pid,
            gpu=args.gpu,
            input_file_name=args.input_file
        )

    elif args.action == 'merge':
        save_files = ['_'.join(input_file.split('_')[:-1] + [f'gpu_{gpu}_nodeNum_{node_number_per_process}_pid_{pid}.jsonl']) for gpu, pid in zip(gpus, range(30))]
        all_nodes = []
        for save_cache_name in save_files:
            nodes_cache_path = os.path.abspath(os.path.join(cache_path, save_cache_name))
            all_nodes.extend(load_nodes_jsonl(nodes_cache_path))
        nodes_cache_path = os.path.abspath(os.path.join(cache_path, '_'.join(input_file.split('_')[:-2])))
        save_nodes_jsonl(all_nodes)