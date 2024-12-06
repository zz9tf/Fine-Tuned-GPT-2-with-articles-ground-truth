import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
from custom.io import load_nodes_jsonl, save_nodes_jsonl

cache_path = "/home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/.cache"
new_file = ""
file_to_be_added = ""

all_nodes = load_nodes_jsonl(file_to_be_added)
all_nodes.extend(load_nodes_jsonl(new_file))
nodes_cache_path = os.path.abspath(os.path.join(cache_path, f"gpt-4o-batch-all-p_2_parser_ManuallyHierarchicalNodeParser_8165_gpu_V100_nodeNum_{len(all_nodes)}.jsonl"))
save_nodes_jsonl(nodes_cache_path, all_nodes)