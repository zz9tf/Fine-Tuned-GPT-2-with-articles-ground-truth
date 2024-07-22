##########################################################################
# parser
from utils.custom_parser import (
    CustomHierarchicalNodeParser
)
from llama_index.core.node_parser import (
    SentenceSplitter, 
    SimpleFileNodeParser, 
    HierarchicalNodeParser
)

def get_parser(self, config):
    VALID_PARSER = self.prefix_config['parser'].keys()
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
            llm=get_llm(self, self.prefix_config['llm'][config['llm']])
        )
    else:
        raise Exception("Invalid parser config. Please provide parser types {}".format(VALID_PARSER))
    
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
    if extractor_config['type'] == 'HuggingfaceBasedExtractor':
        return HuggingfaceBasedExtractor(
            model_name=extractor_config['llm'],
            no_split_modules=llm_config['no_split_modules'],
            cache_dir=self.config['cache'],
            num_questions=extractor_config['num_questions']
        )
    elif extractor_config['type'] == 'OllamaBasedExtractor':
        return OllamaBasedExtractor(
            model_name=extractor_config['llm'],
            embedding_only=extractor_config.get('embedding_only', True),
            only_meta=extractor_config.get('only_meta', None)
        )
    elif extractor_config['type'] == 'OpenAIBasedExtractor':
        return OpenAIBasedExtractor(
            model_name=extractor_config['llm'],
            cache_dir=os.path.abspath(os.path.join(self.root_path, self.config['cache'])),
            mode=extractor_config['mode'],
            embedding_only=extractor_config.get('embedding_only', True),
            only_meta=extractor_config.get('only_meta', None)
        )

##########################################################################
# Embedding model method
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils.custom_embedding import CustomOllamaBasedEmbedding

def get_embedding_model(embedding_config):
    if embedding_config['based_on'] == 'huggingface':
        exit()
        return HuggingFaceEmbedding
    elif embedding_config["based_on"] == 'ollama':
        return CustomOllamaBasedEmbedding(
            model_name=embedding_config['name']
        )

##########################################################################
# LLM method
import torch
from transformers import BitsAndBytesConfig
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI as llama_index_openai
from llama_index.llms.huggingface import HuggingFaceLLM

def get_llm(self, llm_config):
    if llm_config['based_on'] == "ollama":
        llm = Ollama(model=llm_config['model_name'], request_timeout=240.0, temperature=0.3)
    elif llm_config['based_on'] == 'openai':
        llm = llama_index_openai(model=llm_config['model_name'], api_key=os.getenv('OPENAI_API_KEY'))
    elif llm_config['based_on'] == "huggingface":
        # TODO Custom Huggingface model

        # quantize to save memory
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            # bnb_4bit_use_double_quant=True,
        )

        llm = HuggingFaceLLM(
            model_name=llm_config['model_name'],
            tokenizer_name=llm_config['model_name'],
            context_window=3900,
            max_new_tokens=256,
            model_kwargs={
                "quantization_config": quantization_config,
                "cache_dir": llm_config['cache_dir']
            },
            generate_kwargs={"temperature": 0.3, "top_k": 50, "top_p": 0.95},
            token=os.getenv("HUGGING_FACE_TOKEN"),
            device_map="auto"
        )

    else:
        raise Exception(f"Invalid llm based {llm_config['based_on']}")
    
    return llm

##########################################################################
# Index
from llama_index.core import VectorStoreIndex, PropertyGraphIndex

def get_an_index_generator(index_type):
        if index_type == 'VectorStoreIndex':
            return VectorStoreIndex
        elif index_type == 'PropertyGraphIndex':
            return PropertyGraphIndex
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