import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from llama_index.core.node_parser import (
    SentenceSplitter, 
    SimpleFileNodeParser, 
    HierarchicalNodeParser
)
from component.parser.custom_hierarchical_node_parser import CustomHierarchicalNodeParser
from component.parser.manually_parser import ManuallyParser

def get_parser(config, llm_config=None, embedding_config=None, **kwargs):
    """get a parser"""
    if config['type'] == 'SentenceSplitter':
        return SentenceSplitter(
            chunk_size=config.get('chunk_size', 1024), 
            chunk_overlap=config.get('chunk_overlap', 200)
        )
    elif config['type'] == 'SimpleFileNodeParser':
        return SimpleFileNodeParser()
    elif config['type'] == 'HierarchicalNodeParser':
        return HierarchicalNodeParser.from_defaults(
            chunk_sizes=config.get('chunk_size', [2048, 512, 128])
        )
    elif config['type'] == 'CustomHierarchicalNodeParser':
        return CustomHierarchicalNodeParser.from_defaults(
            llm_config=llm_config,
            embedding_config=embedding_config
        )
    elif config['type'] == "ManuallyHierarchicalNodeParser":
        return ManuallyParser(
            cache_path=kwargs["cache_path"],
            cache_name=f'{kwargs["index_id"]}_{kwargs["step_id"]}_{kwargs["step_type"]}_{kwargs["action"]}',
            delete_cache=False
        )
    else:
        raise Exception(
            f"Invalid parser config with config {config}. Please provide parser types {self.prefix_config['parser'].keys()}"
        )
    