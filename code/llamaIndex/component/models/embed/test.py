import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))
import torch
import os
import numpy as np
from component.models.embed.get_embedding_model import get_embedding_model
from component.io import load_nodes_jsonl
from configs.load_config import load_configs

cache_path = os.path.abspath('../../../.cache')
file_name = r'gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_50_pid_32.jsonl'
nodes = load_nodes_jsonl(os.path.join(cache_path, file_name))
print(nodes[1].text)
print(nodes[2].text)

_, prefix_config = load_configs()
embed_config = prefix_config['embedding_model']['Linq-AI-Research/Linq-Embed-Mistral']

embed_model = get_embedding_model(embedding_config=embed_config)

torch.cuda.empty_cache()
embed1 = embed_model._get_text_embedding(nodes[1].text)
torch.cuda.empty_cache()
embed2 = embed_model._get_text_embedding(nodes[2].text)
print(embed1 == embed2)

np.save('embed1.npy', embed1)
np.save('embed2.npy', embed2)