# import os
# import sys
# sys.path.insert(0, os.path.abspath('..'))
# from configs.load_config import load_configs
# from component.io import save_nodes_jsonl, load_nodes_jsonl
# from component.extractor.partaly_OpenAI_based_QAR_extractor import PartalyOpenAIBasedQARExtractor

# # Load config
# config, prefix_config = load_configs()
# root_path = os.path.abspath('..')
# cache_path = os.path.abspath(os.path.join(root_path, config['cache']))
# extractor_config = prefix_config['extractor']['manually_partaly_QAExtractor']

# # load extractor
# extractor = PartalyOpenAIBasedQARExtractor(
#     model_name=extractor_config['llm'],
#     cache_dir=cache_path,
#     mode=extractor_config['mode'],
#     embedding_only=extractor_config.get('embedding_only', True)
# )

# # Load data
# input_file = 'gpt-4o-batch-all-p_2_parser_ManuallyHierarchicalNodeParser_8165_gpu_V100_nodeNum_200_pid_1.jsonl'
# nodes = load_nodes_jsonl(os.path.abspath(os.path.join(cache_path, input_file)))

# # Do extracting
# output_file_base = 'gpt-4o-batch-all-p_2_parser_ManuallyHierarchicalNodeParser_8165_gpu_V100_nodeNum_200_pid_1_extract'
# # Extract response to improve the extract presentage
# extractor.extract(
#     nodes=nodes, 
#     cache_path=cache_path,
#     csv_file_name=output_file_base+'.csv')

# # Save nodes
# output_file = output_file_base + '.jsonl'
# save_nodes_jsonl(file_path=os.path.abspath(os.path.join(cache_path, output_file)), nodes=nodes)

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from configs.load_config import load_configs
from component.io import save_nodes_jsonl, load_nodes_jsonl
from component.extractor.partaly_OpenAI_based_QAR_extractor import PartalyOpenAIBasedQARExtractor

def extract_pipeline():
    # Initialize paths and configurations
    _, prefix_config = load_configs()
    cache_dir = os.path.abspath('../.cache')
    
    # Load extractor
    extractor_config = prefix_config['extractor']['manually_partaly_QAExtractor']
    extractor = PartalyOpenAIBasedQARExtractor(
        model_name=extractor_config['llm'],
        cache_dir=cache_dir,
        mode=extractor_config['mode'],
        embedding_only=extractor_config.get('embedding_only', True)
    )
    
    # Load input data
    input_file = 'gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_50_pid_0.jsonl'
    nodes = load_nodes_jsonl(os.path.join(cache_dir, input_file))
    
    # Define output file base name
    output_file_base = f"{input_file.split('_')[0]}_extract_gpt-4o-QAExtractor-batch_pid_{input_file.split('_')[-1]}"
    output_file_path = os.path.join(cache_dir, output_file_base+'.jsonl')
    
    # Run extraction
    extractor.extract(
        nodes=nodes,
        csv_file_name=output_file_base + '.csv'
    )
    
    # Save output data
    save_nodes_jsonl(nodes, output_file_path)

if __name__ == "__main__":
    extract_pipeline()
