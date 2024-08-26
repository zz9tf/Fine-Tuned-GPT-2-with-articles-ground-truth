import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
import yaml
from dotenv import load_dotenv
from custom.io import save_nodes_jsonl, load_nodes_jsonl
from custom.extractor import PartalyOpenAIBasedQARExtractor

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
cache_path = r'D:\Projects(D)\Fine-Tuned-GPT-2-with-articles-ground-truth\code\llamaIndex\.save'
extractor_config = prefix_config['extractor']['manually_partaly_QAExtractor']

# load extractor
extractor = PartalyOpenAIBasedQARExtractor(
    model_name=extractor_config['llm'],
    cache_dir=os.path.abspath(os.path.join(root_path, config['cache'])),
    mode=extractor_config['mode'],
    embedding_only=extractor_config.get('embedding_only', True)
)

# Load data
input_file = 'gpt-4o-batch-all-p_2_parser_ManuallyHierarchicalNodeParser_8165_gpu_V100_nodeNum_200_pid_1.jsonl'
nodes = load_nodes_jsonl(os.path.abspath(os.path.join(cache_path, input_file)))
# Do extracting
output_file_base = 'gpt-4o-batch-all-p_2_parser_ManuallyHierarchicalNodeParser_8165_gpu_V100_nodeNum_200_pid_1_extract'
extractor.extract(
    nodes=nodes, 
    cache_path=os.path.abspath(os.path.join(root_path, config['cache'])),
    csv_file_name=output_file_base+'.csv')

# Save nodes
output_file = output_file_base + '.jsonl'
save_nodes_jsonl(file_path=os.path.abspath(os.path.join(cache_path, output_file)), nodes=nodes)