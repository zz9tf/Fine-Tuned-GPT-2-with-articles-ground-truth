import os
import yaml
import openai
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex,
    PropertyGraphIndex,
    StorageContext,
    load_index_from_storage,
    load_indices_from_storage,
    load_graph_from_storage
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.node_parser import (
    SentenceSplitter, 
    SimpleFileNodeParser, 
    HierarchicalNodeParser
)
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core.ingestion import IngestionPipeline
from extractor.qa_extractor import QAExtractor
from llama_index.core.extractors import QuestionsAnsweredExtractor

class Database():
    def __init__(self, config_path):
        self.root_path = "../.."
        print("[init] Loading configuration ...", end=' ')
        config_path = os.path.abspath(os.path.join(self.root_path, config_path))
        with open(config_path, 'r') as config:
            self.config = yaml.safe_load(config)
        print("done")

    def _get_llm_model(self, repo_config):
        VALID_MODELS = ['lmsys/vicuna-13b-v1.3', 'lmsys/vicuna-13b-v1.5-16k', 'lmsys/vicuna-33b-v1.3', 'gpt-4o']
        if repo_config['name'] == 'gpt-4o':
            # os.environ["OPENAI_API_KEY"] = "sk-..."
            openai.api_key = os.environ["OPENAI_API_KEY"]
            return OpenAI(model="gpt-4o")
        elif repo_config['name'] in VALID_MODELS:
            tokenizer = AutoTokenizer.from_pretrained(repo_config['name'], cache_dir=repo_config['cache'], local_files_only=True)
            model = AutoModelForCausalLM .from_pretrained(repo_config['name'], cache_dir=repo_config['cache'], local_files_only=True)
            return HuggingFaceLLM(model_name=repo_config['name'], model=model, tokenizer=tokenizer)
        else:
            raise Exception("Invalid embedding model name. Please provide LLM model {}".format(VALID_MODELS))

    def _get_embedding_model(self, repo_config):
        VALID_EMBED_MODEL = ['Linq-AI-Research/Linq-Embed-Mistral']
        if repo_config['name'] in VALID_EMBED_MODEL:
            return HuggingFaceEmbedding(
                model_name=repo_config['name'],
                cache_folder=repo_config['cache']
            )
        elif repo_config['name'] == '[finetune]Linq-Embed-Mistral':
            return None
        else:
            raise Exception("Invalid embedding model name. Please provide embedding models {}".format(VALID_EMBED_MODEL))

    def _get_parser(self, parser_config):
        VALID_PARSER = ['SentenceSplitter', 'SimpleFileNodeParser', 'HierarchicalNodeParser']
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
        else:
            raise Exception("Invalid embedding model name. Please provide parser types {}".format(VALID_PARSER))

    def _get_extractors(self, extractor_config):
        if extractor_config['name'] == 'QAExtractor':
            return QAExtractor(
                llm=self._get_llm_model(repo_config=extractor_config['llm']),
                questions=extractor_config['questions']
            )
        elif extractor_config['name'] == 'QuestionsAnsweredExtractor':
            return QuestionsAnsweredExtractor(
                llm=self._get_llm_model(repo_config=extractor_config['llm']),
                questions=extractor_config['questions']
            )

    def _get_a_store(self, store_type):
        if store_type == 'SimpleDocumentStore':
            return SimpleDocumentStore()
        if store_type == 'SimpleIndexStore':
            return SimpleIndexStore()
        if store_type == 'SimpleVectorStore':
            return SimpleVectorStore()

    def get_an_index(self, index_config):
        if index_config['type'] == 'VectorStoreIndex':
            return VectorStoreIndex
        elif index_config['type'] == '':
            return None
        else:
            raise Exception("Invalid embedding model name. Please provide embedding models {}".format(VALID_EMBED_MODEL))

    def _init_nodes_generation_pipeline(self, index_config):
        transformations = []
        
        # Get parser
        if 'parser' in index_config:
            print("[update_database] Initializing parser...", end=' ')
            parser_config = self.config['prefix_config']['parser'][index_config['parser']]
            parser = self._get_parser(parser_config)
            transformations.append(parser)
            print("done")
        
        # Get extractors
        if 'extractors' in index_config:
            print("[update_database] Initializing extractors...", end=' ')
            for extractor_config_name in index_config['extractors']:
                extractor_config = self.config['prefix_config']['extractor'][extractor_config_name]
                transformations.append(self._get_extractors(extractor_config))
            print("done")

        # Get a embedding model
        if 'embedding_model' in index_config:
            print('[update_database] Initializing the embedding model...', end=' ')
            embedding_config = self.config['prefix_config']['embedding_model'][index_config['embedding_model']]
            embed_model = self._get_embedding_model(embedding_config)
            transformations.append(embed_model)
            print("done")
        
        # Initialize pipeline
        print("[update_database] Initializing pipeline...", end=' ')
        pipeline = IngestionPipeline(
            transformations=transformations,
        )
        print("done")

        return pipeline
        
    def _load_documents(self):
        print("[update_database] Loading documents ...", end=' ') 
        file_path = self.config['document_preprocessing']['data_dir_path']
        data_path = os.path.abspath(os.path.join(self.root_path, file_path))
        documents = SimpleDirectoryReader(
            input_dir=data_path, 
            exclude=[],
            file_metadata=lambda file_path : {"file_path": file_path},
            filename_as_id=True
        ).load_data()
        print("done")
        return documents
    
    def _generate_nodes_from_documents(self, index_config, documents, pipeline):
        print("[update_database] Executing pipeline ...")
        nodes = pipeline.run(
            show_progress=True,
            documents=documents,
            num_workers=index_config['pipeline'].get('num_workers', 1)
        )
        print('done')

        if index_config['pipeline']['is_cache']:
            cache_path = os.path.abspath(os.path.join(self.root_path, index_config['pipeline']['cache_path']))
            self.pipeline.persist(cache_path)

        return nodes

    def create_or_update_indexes(self):
        for index_id, index_config in self.config['document_preprocessing']['indexes'].items():
            print('[update_database] Updating index: {}'.format(index_id))
            pipeline = self._init_nodes_generation_pipeline(index_config)
            documents = self._load_documents()

            nodes = self._generate_nodes_from_documents(index_config, documents, pipeline)

            # Initial storage_context config
            storage_context_config = self.config['prefix_config']['storage_context'][index_config['storage_context']]
            store_path = os.path.abspath(os.path.join(storage_context_config['store_dir_path'], storage_context_config['name']))
            if os.path.exists(store_path):
                print("[update_database] Storage does not find")
                print("[update_database] Creating a new one...")

            # Generate index for nodes
            indexGenerator = self._get_an_index(index_config['index'])
            index = indexGenerator.from_documents(
                documents=nodes,
                storage_context=StorageContext.from_defaults(
                    docstore=self._get_a_store(storage_context_config['docstore']),
                    vector_store=self._get_a_store(storage_context_config['vector_store']),
                    index_store=self._get_a_store(storage_context_config['index_store']),
                    property_graph_store=self._get_a_store(storage_context_config['property_graph_store'])
                ),
                persist_dir=store_path if os.path.exists(store_path) else None,
                show_progress=True
            )

            if not os.path.exists(store_path):
                index.set_index_id(self.config['index']['index_id'])
                index.storage_context.persist(store_path)

        return index

if __name__ == '__main__':
    d = Database(config_path='./code/llamaIndex/config.yaml')
    index = d.create_or_update_indexes()

