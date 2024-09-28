import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from component.models.llm.get_llm import get_llm
from component.extractor.openai_QAR_extractor import OpenAIBasedQARExtractor
from component.extractor.custom_LLM_based_QAR_extractor import CustomLLMBasedQARExtractor
from component.extractor.partaly_OpenAI_based_QAR_extractor import PartalyOpenAIBasedQARExtractor

def get_extractors(self, extractor_config):
    llm_config = self.prefix_config['llm'][extractor_config['llm']]
    
    if extractor_config['type'] == 'OpenAIBasedExtractor':
        return OpenAIBasedQARExtractor(
            model_name=extractor_config['llm'],
            cache_dir=os.path.abspath(os.path.join(self.root_path, self.config['cache'])),
            mode=extractor_config['mode'],
            embedding_only=extractor_config.get('embedding_only', True)
        )
    elif extractor_config['type'] in ['OllamaBasedExtractor', 'HuggingfaceBasedExtractor']:
        return CustomLLMBasedQARExtractor(
            llm_self=get_llm(self, llm_config),
            embedding_config=self.prefix_config['embedding_model'][extractor_config['embedding_model']],
            embedding_only=extractor_config.get('embedding_only', True)
        )
    elif extractor_config['type'] == PartalyOpenAIBasedQARExtractor:
        return PartalyOpenAIBasedQARExtractor(
            model_name=extractor_config['llm'],
            cache_dir=os.path.abspath(os.path.join(self.root_path, self.config['cache'])),
            mode=extractor_config['mode'],
            embedding_only=extractor_config.get('embedding_only', True)
        )