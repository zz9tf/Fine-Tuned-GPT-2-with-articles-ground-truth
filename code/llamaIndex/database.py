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
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import QuestionsAnsweredExtractor
from utils.custom_extractor import QAExtractor
from utils.custom_embedding import CustomEmbeddings

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
            model = AutoModelForCausalLM .from_pretrained(repo_config['name'], cache_dir=repo_config['cache'], local_files_only=True, device_map='auto')
            return HuggingFaceLLM(model_name=repo_config['name'], model=model, tokenizer=tokenizer)
        else:
            raise Exception("Invalid embedding model name. Please provide LLM model {}".format(VALID_MODELS))

    def _get_embedding_model(self, repo_config):
        VALID_EMBED_MODEL = ['Linq-AI-Research/Linq-Embed-Mistral']
        if repo_config['name'] in VALID_EMBED_MODEL:
            return CustomEmbeddings(
                model_name=repo_config['name'],
                cache_dir=repo_config['cache'],
                embed_batch_size=4
            )
            # return HuggingFaceEmbedding(
            #     model_name=repo_config['name'],
            #     cache_folder=repo_config['cache']
            # )
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
        if extractor_config['extractor_type'] == 'QAExtractor':
            return QAExtractor(
                llm=self._get_llm_model(repo_config=extractor_config['llm']),
                questions=extractor_config['questions']
            )
        elif extractor_config['extractor_type'] == 'QuestionsAnsweredExtractor':
            return QuestionsAnsweredExtractor(
                llm=self._get_llm_model(repo_config=extractor_config['llm']),
                questions=extractor_config['questions']
            )

    def _get_a_store(self, store_type):
        if store_type == 'SimpleDocumentStore':
            return SimpleDocumentStore()
        elif store_type == 'SimpleIndexStore':
            return SimpleIndexStore()
        elif store_type == 'SimpleVectorStore':
            return SimpleVectorStore()
        
    def _get_an_indexGenerator(self, index_type):
        if index_type == 'VectorStoreIndex':
            return VectorStoreIndex
        elif index_type == '':
            return None
        else:
            raise Exception("Invalid embedding model name. Please provide embedding models {}".format())

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
    
    def _generate_nodes_from_documents(self, index_config, documents):
        print("[update_database] Generating nodes from documents...", end=' ')
        transformations = []
        # Get parser
        if 'parser' in index_config:
            parser_config = self.config['prefix_config']['parser'][index_config['parser']]
            transformations.append(self._get_parser(parser_config))

        # Initialize pipeline
        pipeline = IngestionPipeline(
            transformations=transformations,
        )

        nodes = pipeline.run(
            show_progress=True,
            documents=documents,
            num_workers=index_config['pipeline'].get('num_workers', 10)
        )
        print('done')

        if index_config['pipeline']['is_cache']:
            cache_path = os.path.abspath(os.path.join(self.root_path, index_config['pipeline']['cache_path']))
            self.pipeline.persist(cache_path)

        return nodes

    def _generate_metadata_to_nodes(self):
        pass
        # if 'extractors' in index_config:
        # # print('[update_database] Extracting metadata ...')
        # for extractor_name in index_config['extractors']:
        #     # print('[update_database] Doing {}...'.format(extractor_name))
        #     extractor = self._get_extractors(self.config['prefix_config']['extractor'][extractor_name])
        #     transformations.append(extractor)

    def create_or_update_indexes(self):
        for index_id in self.config['document_preprocessing']['indexes']:
            index_config = self.config['prefix_config']['indexes'][index_id]
            print('[update_database] Updating index: {}'.format(index_id))
            
            storage_context_config = self.config['prefix_config']['storage_context'][index_config['storage_context']]
            store_path = os.path.abspath(os.path.join(self.root_path, storage_context_config['store_dir_path'], storage_context_config['name']))
            
            if not os.path.exists(store_path):
                print("[update_database] Storage does not find with path: {}".format(store_path))
                print("[update_database] Creating a new one...")

            documents = self._load_documents()
            nodes = self._generate_nodes_from_documents(index_config, documents)

            # Generate index for nodes
            Settings.embed_model = self._get_embedding_model(self.config['prefix_config']['embedding_model'][index_config['embedding_model']])
            indexGenerator = self._get_an_indexGenerator(storage_context_config['index_generator'])
            index = indexGenerator(
                nodes=nodes,
                storage_context=StorageContext.from_defaults(
                    docstore=self._get_a_store(storage_context_config['docstore']),
                    vector_store=self._get_a_store(storage_context_config['vector_store']),
                    index_store=self._get_a_store(storage_context_config['index_store']),
                    # property_graph_store=self._get_a_store(storage_context_config['property_graph_store'])
                ),
                persist_dir=store_path if os.path.exists(store_path) else None,
                show_progress=True
            )

            if not os.path.exists(store_path):
                index.set_index_id(index_id)
                index.storage_context.persist(store_path)

        return index

    def show_indexes(self):
        pass


if __name__ == '__main__':
    d = Database(config_path='./code/llamaIndex/config.yaml')
    index = d.create_or_update_indexes()

