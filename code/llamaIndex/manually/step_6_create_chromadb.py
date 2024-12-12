import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import json
from llama_index.core.schema import TextNode
import argparse
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm
from configs.load_config import load_configs
import re
from utils.generate_and_execute_slurm_job import generate_and_execute_slurm_job

def load_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Create chromadbs.")
    parser.add_argument('--action', type=str, default='main', help='The action to create chromadb')
    parser.add_argument('--pid', type=str, default=None, help='The PID of the subtask to create chromadb')

    return parser.parse_args()

def submit_job(
    script_path: str,
    cpu_num: str,
    pid: int,
    python_file_name: str,
    action: str
):
    """Submit a job to Slurm and return the job ID."""
    job_name = f'gene_{pid}-chrdb'
    log_file_path = os.path.abspath(os.path.join(script_path, 'out/{job_name}.out'))
    script_path = os.path.abspath(os.path.join(script_path, 'execute/execute.sh'))
    job_name = generate_and_execute_slurm_job(
            python_start_script=f"{python_file_name} --action {action} --pid {pid}",
            account="guest",
            partition="guest-compute",
            job_name=job_name,
            qos='low',
            time="24:00:00",
            num=cpu_num,
            log_file_path=log_file_path,
            script_path=script_path
        )
    print(f"[PID: {pid}] is submitted!")
    return job_name

def iter_batch(iterable, batch_size):
    """Generate batches from an iterable with the given batch size."""
    length = len(iterable)
    for idx in range(0, length, batch_size):
        yield iterable[idx:idx + batch_size]

def add_nodes_to_chromadb(index_dir_path, index_id, break_num: int=None, db_name=None):
    # Generate index for nodes
    file_path = os.path.join(index_dir_path, index_id) + '.jsonl'
    db_name = db_name if db_name else index_id
    not_finish = False
    if os.path.exists(db_name):
        # Check if collection already exists
        db_path = os.path.join(index_dir_path, db_name+'_chroma')
        chroma_client = chromadb.PersistentClient(path=db_path)
        chroma_collection = chroma_client.get_collection(name='quickstart')
    else:
        # create one if it doesn't exist
        db_path = os.path.join(index_dir_path, db_name+'_not_finish_chroma')
        chroma_client = chromadb.PersistentClient(path=db_path)
        chroma_collection = chroma_client.create_collection(name='quickstart')
        not_finish = True
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Get the total file size
    file_size = os.path.getsize(file_path)
    
    # Read the file and track progress based on bytes read
    nodes = []
    with open(file_path, 'r') as file:
        with tqdm(total=file_size, desc=f'Loading {file_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for i, line in enumerate(file):
                if break_num is not None and i == break_num:
                    break
                node_data = json.loads(line)
                node = TextNode.from_dict(node_data)
                node.metadata = {k: str(v) for k, v in node.metadata.items()}
                nodes.append(node)
                # Update progress bar based on bytes read
                pbar.update(len(line))
                if len(nodes) == 2048:
                    vector_store.add(nodes)
    if len(nodes) > 0:
        vector_store.add(nodes)
    if not_finish:
        os.rename(db_path, os.path.join(index_dir_path, index_id+'_chroma'))

def add_nodes_to_levels_chromadb(index_dir_path, index_id, break_num: int=None, db_name=None):
    # Generate index for nodes
    levels = ['document', 'section', 'paragraph', 'multi-sentences']
    file_path = os.path.join(index_dir_path, index_id) + '.jsonl'
    db_name = db_name if db_name else index_id
    level_dict = {}
    for level in levels:
        not_finish = False
        if os.path.exists(db_name):
            # Check if collection already exists
            db_path = os.path.join(index_dir_path, f"{db_name}_{level}_chroma")
            chroma_client = chromadb.PersistentClient(path=db_path)
            chroma_collection = chroma_client.get_collection(name='quickstart')
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        else:
            not_finish = True
            db_path = os.path.join(index_dir_path, f"{db_name}_{level}_not_finish_chroma")
            chroma_client = chromadb.PersistentClient(path=db_path)
            chroma_collection = chroma_client.create_collection(name='quickstart')
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        level_dict[level] = {}
        level_dict[level]['vector_store'] = vector_store
        level_dict[level]['not_finish'] = not_finish
        level_dict[level]['nodes'] = []
    
    # Get the total file size
    file_size = os.path.getsize(file_path)
    
    # Read the file and track progress based on bytes read
    with open(file_path, 'r') as file:
        with tqdm(total=file_size, desc=f'Loading {file_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for i, line in enumerate(file):
                if break_num is not None and i == break_num:
                    break
                node_data = json.loads(line)
                node = TextNode.from_dict(node_data)
                node.metadata = {k: str(v) for k, v in node.metadata.items()}
                node_level = node.metadata['level']
                level_dict[node_level]['nodes'].append(node)
                pbar.update(len(line))
                if (len(level_dict[node_level]['nodes']) >= 2048):
                    level_dict[node_level]['vector_store'].add(level_dict[node_level]['nodes'])
                    print(f"Added {len(level_dict[node_level]['nodes'])} nodes to level {node_level}")
                    level_dict[node_level]['nodes'] = []
                    
    for level in level_dict.keys():
        if (len(level_dict[level]['nodes']) > 0):
            level_dict[level]['vector_store'].add(level_dict[level]['nodes'])
            print(f"Added {len(level_dict[level]['nodes'])} nodes to level {level}")
        if level_dict[level]['not_finish']:
            db_path = os.path.join(index_dir_path, f"{db_name}_{level}_not_finish_chroma")
            print(f'Renamed file {db_name}_{level}_not_finish_chroma')
            os.rename(db_path, os.path.join(index_dir_path, f"{db_name}_{level}_chroma"))
        print(level_dict[level]['vector_store']._collection.count())
    
if __name__ == "__main__":
    # Load config
    total_config, prefix_config = load_configs()
    root_path = "../../.."
    indexes_dir_path = os.path.abspath(os.path.join(root_path, total_config['indexes_dir_path']))
    print(f"indexes_dir_path: {indexes_dir_path}")
    
    # prefix parameters
    python_file_name = 'step_6_create_chromadb.py'
    index_name = 'gpt-4o-batch-all-target' # modify each time  ## gpt-4o-batch-all-target
    action = 'thread' # modify each time
    index_dir_path = os.path.join(indexes_dir_path, index_name)
    break_num = None
    notIncludeNotFinishChroma = True
    cpu_num = str(2) # cpu number
    
    leave_tasks = []
    for filename in os.listdir(index_dir_path):
        match = re.search(r'(\d+)\.jsonl', filename)
        if match:
            task_id = match.group(1)
            leave_tasks.append(task_id)
    if action == 'thread':
        for filename in os.listdir(index_dir_path):
            match = re.search(r'(\d+)\_chroma', filename)
            if match:
                task_id = match.group(1)
                leave_tasks.remove(task_id)
            else:
                match = re.search(r'(\d+)_not_finish_chroma', filename)
                if match and notIncludeNotFinishChroma:
                    task_id = match.group(1)
                    leave_tasks.remove(task_id)
    elif action == 'thread_levels':
        leave_tasks = {task_id:[] for task_id in leave_tasks}
        for filename in os.listdir(index_dir_path):
            levels = ['document', 'section', 'paragraph', 'multi-sentences']
            for level in levels:
                match = re.search(f'(\\d+)\\_{level}_chroma', filename)
                if match:
                    task_id = match.group(1)
                    leave_tasks[task_id].append(match)
                else:
                    match = re.search(f'(\\d+)\\_{level}_not_finish_chroma', filename)
                    for level in levels:
                        match = re.search(f'(\\d+)\\_{level}_chroma', filename)
                        if match and notIncludeNotFinishChroma:
                            task_id = match.group(1)
                            leave_tasks[task_id].append(match)
        leave_tasks = [task_id for task_id, v in leave_tasks.items() if len(v) < 4]
    print(f"leave task: {len(leave_tasks)}")
    leave_tasks.sort(key=lambda x: int(x))
    # leave_tasks = leave_tasks[:30]
    args = load_args()
    
    if args.action == 'main':
        for i, pid in enumerate(leave_tasks):
            job_name = submit_job(
                script_path=os.getcwd(),
                cpu_num=cpu_num,
                pid=pid,
                python_file_name=python_file_name,
                action=action
            )
                
    elif args.action == 'thread':
        add_nodes_to_chromadb(
            index_dir_path=index_dir_path, 
            index_id=args.pid, 
            break_num=break_num
        )
    elif args.action == 'thread_levels':
        add_nodes_to_levels_chromadb(
            index_dir_path=index_dir_path, 
            index_id=args.pid, 
            break_num=break_num
        )