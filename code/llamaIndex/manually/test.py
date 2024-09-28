# import os, sys
# sys.path.insert(0, os.path.abspath('..'))
# import torch
# import os
# import json
# from typing import List
# from tqdm import tqdm
# import numpy as np
# from llama_index.core.schema import BaseNode, TextNode
# from custom.embedding import get_embedding_model

# def save_nodes_jsonl(file_path: str, nodes: List[BaseNode]):
#     try:
#         with open(file_path, 'w') as file:
#             for node in tqdm(nodes, desc='Saving nodes...'):
#                 json.dump(node.to_dict(), file)
#                 file.write('\n')
#     except Exception as e:
#         print(f"An error occurred while saving nodes: {e}")

# def load_nodes_jsonl(file_path: str) -> List[TextNode]:
#     nodes = []
    
#     try:
#         # Get the total file size
#         file_size = os.path.getsize(file_path)
        
#         # Read the file and track progress based on bytes read
#         with open(file_path, 'r') as file:
#             with tqdm(total=file_size, desc='Loading nodes...', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
#                 for line in file:
#                     node_data = json.loads(line)
#                     nodes.append(TextNode.from_dict(node_data))
#                     # Update progress bar based on bytes read
#                     pbar.update(len(line))
#     except Exception as e:
#         print(f"An error occurred while loading nodes: {e}")
    
#     return nodes

# file_path = r'/home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/.cache/gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_50_pid_32.jsonl'
# # file_path = r'/home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/.cache/pid-20.jsonl'
# nodes = load_nodes_jsonl(file_path)

# print(nodes[1].text)

# print(nodes[2].text)

# embed_config = {
#     "based_on": "huggingface",
#     "name": "Linq-AI-Research/Linq-Embed-Mistral",
#     "cache_dir": "/work/zhengzheng/.hf_cache"
# }
# embed_model = get_embedding_model(
#     embedding_config=embed_config, device='cuda:1'
# )
# torch.cuda.empty_cache()
# embed1 = embed_model._get_text_embedding(nodes[1].text)
# torch.cuda.empty_cache()
# embed2 = embed_model._get_text_embedding(nodes[2].text)
# print(embed1 == embed2)

# np.save('embed1.npy', embed1)
# np.save('embed2.npy', embed2)

# # Print comparison
# print(embed1 == embed2)