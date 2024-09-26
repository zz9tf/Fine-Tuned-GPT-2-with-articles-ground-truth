import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
from custom.io import load_nodes_jsonl, save_nodes_jsonl


prefix_file_name = "gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_50"
cache_path = "/home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/.cache"
save_files = [file_name for file_name in os.listdir(cache_path) if prefix_file_name in file_name]
all_nodes = []
for save_cache_name in save_files:
    nodes_cache_path = os.path.abspath(os.path.join(cache_path, save_cache_name))
    all_nodes.extend(load_nodes_jsonl(nodes_cache_path))
nodes_cache_path = os.path.abspath(os.path.join(cache_path, f"gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_{len(all_nodes)}.jsonl"))
save_nodes_jsonl(nodes_cache_path, all_nodes)