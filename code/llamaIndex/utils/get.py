##########################################################################
from utils.custom_parser import (
    CustomHierarchicalNodeParser
)
from llama_index.core.node_parser import (
    SentenceSplitter, 
    SimpleFileNodeParser, 
    HierarchicalNodeParser
)

def get_parser(self, parser_config):
    VALID_PARSER = self.prefix_config['parser'].keys()
    if parser_config['name'] == 'SentenceSplitter':
        return SentenceSplitter(
            chunk_size=parser_config.get('chunk_size', 1024), 
            chunk_overlap=parser_config.get('chunk_overlap', 200)
        )
    elif parser_config['name'] == 'SimpleFileNodeParser':
        return SimpleFileNodeParser()
    elif parser_config['name'] == 'HierarchicalNodeParser':
        return HierarchicalNodeParser.from_defaults(
            chunk_sizes=parser_config.get('chunk_size', [2048, 512, 128])
        )
    elif parser_config['name'] == 'CustomHierarchicalNodeParser':
        return CustomHierarchicalNodeParser.from_defaults()
    else:
        raise Exception("Invalid embedding model name. Please provide parser types {}".format(VALID_PARSER))
    
##########################################################################
# Extractor
import os
from utils.custom_extractor import (
    HuggingfaceBasedExtractor,
    OllamaBasedExtractor,
    OpenAIBasedExtractor
)

def get_extractors(self, extractor_config):
    llm_config = self.prefix_config['llm'][extractor_config['llm']]
    if extractor_config['extractor_type'] == 'QAExtractor':
        return HuggingfaceBasedExtractor(
            model_name=extractor_config['llm'],
            no_split_modules=llm_config['no_split_modules'],
            cache_dir=llm_config['cache'],
            num_questions=extractor_config['num_questions']
        )
    elif extractor_config['extractor_type'] == 'OllamaBasedExtractor':
        return OllamaBasedExtractor(
            model_name=extractor_config['llm'],
            embedding_only=extractor_config.get('embedding_only', True),
            only_meta=extractor_config.get('only_meta', None)
        )
    elif extractor_config['extractor_type'] == 'OpenAIBasedExtractor':
        return OpenAIBasedExtractor(
            model_name=extractor_config['llm'],
            cache_dir=os.path.abspath(os.path.join(self.root_path, extractor_config['cache'])),
            mode=extractor_config['mode'],
            embedding_only=extractor_config.get('embedding_only', True),
            only_meta=extractor_config.get('only_meta', None)
        )



##########################################################################
# Embedding model method
from utils.custom_embedding import (
    CustomHuggingfaceBasedEmbedding,
    CustomOllamaBasedEmbedding
)

def get_embedding_model(embedding_config):
    if embedding_config['basedOn'] == 'huggingface':
        return CustomHuggingfaceBasedEmbedding(
            model_name=embedding_config['name'],
            cache_dir=embedding_config['cache']
        )
    elif embedding_config["basedOn"] == 'ollama':
        return CustomOllamaBasedEmbedding(model_name=embedding_config['name'])

##########################################################################
# LLM method
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI as llama_index_openai
from utils.custom_llm import CustomHuggingFaceLLM

def get_llm(self, llm_name):
    if llm_name == "vicuna:13b":
        llm = Ollama(model=llm_name, request_timeout=240.0)
    elif llm_name == "lmsys/vicuna-13b-v1.3":
        # TODO Custom Huggingface model
        llm_config = self.prefix_config['llm'][llm_name]
        llm = CustomHuggingFaceLLM(model=llm_name)
    elif llm_name == 'gpt-4o':
        llm = llama_index_openai(model='gpt-4o', api_key=os.getenv('OPENAI_API_KEY'))
    
    return llm

##########################################################################
# Index
from llama_index.core import VectorStoreIndex

def get_an_index_generator(index_type):
        if index_type == 'VectorStoreIndex':
            return VectorStoreIndex
        else:
            raise Exception("Invalid embedding model name. Please provide embedding models {}".format())

##########################################################################
# Store
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.graph_stores.simple import SimpleGraphStore

def get_a_store(store_type):
    if store_type == 'SimpleDocumentStore':
        return SimpleDocumentStore()
    elif store_type == 'SimpleIndexStore':
        return SimpleIndexStore()
    elif store_type == 'SimpleVectorStore':
        return SimpleVectorStore()
    elif store_type == 'SimpleGraphStore':
        return SimpleGraphStore()