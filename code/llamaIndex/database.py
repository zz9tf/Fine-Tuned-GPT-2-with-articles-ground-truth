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
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import QuestionsAnsweredExtractor
from utils.custom_extractor import QAExtractor, OllamaBasedExtractor
from utils.custom_embedding import CustomEmbeddings

class Database():
    def __init__(self, config_path):
        self.root_path = "../.."
        print("[init] Loading configuration ...", end=' ')
        config_path = os.path.abspath(os.path.join(self.root_path, config_path))
        with open(config_path, 'r') as config:
            self.config = yaml.safe_load(config)
        print("done")

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

    def _get_extractors(self, extractor_config):
        if extractor_config['extractor_type'] == 'QAExtractor':
            return QAExtractor(
                model_name=extractor_config['model_name'],
                no_split_modules=extractor_config['no_split_modules'],
                cache_dir=extractor_config['cache'],
                num_questions=extractor_config['num_questions']
            )
        if extractor_config['extractor_type'] == 'OllamaBasedExtractor':
            return OllamaBasedExtractor(
                model_name=extractor_config['model_name'],
                prompt_template=extractor_config['prompt_template']
            )

    def _generate_metadata_to_nodes(self, index_config, nodes):
        print('[update_database] Extracting metadata ...')
        for extractor_name in index_config['extractors']:
            print('[update_database] Doing {}...'.format(extractor_name))
            extractor = self._get_extractors(self.config['prefix_config']['extractor'][extractor_name])
            extractor.extract(nodes)
        print("done")
        return nodes

    
    def _get_an_indexGenerator(self, index_type):
        if index_type == 'VectorStoreIndex':
            return VectorStoreIndex
        elif index_type == '':
            return None
        else:
            raise Exception("Invalid embedding model name. Please provide embedding models {}".format())

    def _get_a_store(self, store_type):
        if store_type == 'SimpleDocumentStore':
            return SimpleDocumentStore()
        elif store_type == 'SimpleIndexStore':
            return SimpleIndexStore()
        elif store_type == 'SimpleVectorStore':
            return SimpleVectorStore()

    def create_or_update_indexes(self):
        for index_id in self.config['document_preprocessing']['indexes']:
            index_config = self.config['prefix_indexes'][index_id]
            print('[update_database] Updating index: {}'.format(index_id))
            
            storage_context_config = self.config['prefix_config']['storage_context'][index_config['storage_context']]
            index_dir_path = os.path.abspath(os.path.join(self.root_path, self.config['index_dir_path']))
            
            if not os.path.exists(index_dir_path):
                print("[update_database] Storage does not find with path: {}".format(index_dir_path))
                print("[update_database] Creating a new one...")

            documents = self._load_documents()
            nodes = self._generate_nodes_from_documents(index_config, documents)
            nodes = self._generate_metadata_to_nodes(index_config, nodes)
            
            #Load embedding model
            embedding_config = self.config['prefix_config']['embedding_model'][index_config['embedding_model']]
            Settings.embed_model = CustomEmbeddings(
                model_name=embedding_config['name'],
                cache_dir=embedding_config['cache'],
                embed_batch_size=4
            )

            # Generate index for nodes
            indexGenerator = self._get_an_indexGenerator(storage_context_config['index_generator'])
            index = indexGenerator(
                nodes=nodes,
                storage_context=StorageContext.from_defaults(
                    docstore=self._get_a_store(storage_context_config['docstore']),
                    vector_store=self._get_a_store(storage_context_config['vector_store']),
                    index_store=self._get_a_store(storage_context_config['index_store']),
                    # property_graph_store=self._get_a_store(storage_context_config['property_graph_store'])
                ),
                persist_dir=index_dir_path if os.path.exists(index_dir_path) else None,
                show_progress=True
            )

            if not os.path.exists(index_dir_path):
                index.set_index_id(index_id)
                index.storage_context.persist(index_dir_path)

        return index

    def get_all_index_ids(self):
        index_dir_path = self.config['index_dir_path']
        return [d for d in os.listdir(index_dir_path) if os.path.isdir(os.path.join(index_dir_path, d))]


if __name__ == '__main__':
    d = Database(config_path='./code/llamaIndex/config.yaml')
    index = d.create_or_update_indexes()

