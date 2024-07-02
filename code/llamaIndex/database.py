import os
import yaml
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex,
    PropertyGraphIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.core.node_parser import (
    SentenceSplitter, 
    SimpleFileNodeParser, 
    HierarchicalNodeParser
)
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core import Settings
from utils.custom_extractor import HuggingfaceBasedExtractor, OllamaBasedExtractor, OpenAIBasedExtractor
from utils.custom_embedding import OllamaCustomEmbeddings
from datetime import datetime
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI as llama_index_openai
from utils.custom_llm import HuggingFace
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer
from dotenv import load_dotenv
from utils.gpt_4o_json_reader import easy_reader

class Database():
    def __init__(self, config_dir_path):
        self.root_path = "D:\Projects(D)\Fine-Tuned-GPT-2-with-articles-ground-truth" # "../.."
        self.config_dir_path = config_dir_path
        self._load_configs()

    def _load_configs(self):
        config_path = os.path.abspath(os.path.join(self.root_path, self.config_dir_path, 'config.yaml'))
        prefix_config_path = os.path.abspath(os.path.join(self.root_path, self.config_dir_path, 'prefix_config.yaml'))
        with open(config_path, 'r') as config:
            self.config = yaml.safe_load(config)
        with open(prefix_config_path, 'r') as prefix_config:
            self.prefix_config = yaml.safe_load(prefix_config)
        load_dotenv(dotenv_path=os.path.abspath(os.path.join(self.root_path, './code/llamaIndex/.env')))

    def _get_parser(self, parser_config):
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
        
        parser_config = self.prefix_config['parser'][index_config['parser']]
        parser = self._get_parser(parser_config)
        nodes = parser.get_nodes_from_documents(
            documents, show_progress=True
        )
        print('done')

        return nodes

    def _get_extractors(self, extractor_config):
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
                model_name=extractor_config['llm']
            )
        elif extractor_config['extractor_type'] == 'OpenAIBasedExtractor':
            return OpenAIBasedExtractor(
                model_name=extractor_config['llm'],
                cache_dir=os.path.abspath(os.path.join(self.root_path, extractor_config['cache'])),
                mode=extractor_config['mode']
            )

    def _generate_metadata_to_nodes(self, index_config, nodes):
        print('[update_database] Extracting metadata ...')
        for extractor_index_config in index_config['extractors']:
            extractor_name, extractor_index_config = next(iter(extractor_index_config.items()))
            print('[update_database] Doing {}...'.format(extractor_name))
            extractor_config = self.prefix_config['extractor'][extractor_name]
            if extractor_config['llm'] == 'gpt-4o':
                # Create a cache for gpt-4o results
                cache_path = os.path.abspath(os.path.join(self.root_path, extractor_config['cache']))
                os.makedirs(cache_path, exist_ok=True)

                # Check if the nodes cache exists
                nodes_cache_path = os.path.abspath(os.path.join(cache_path, extractor_name+'.json'))
                # If not, create nodes cache
                extractor = self._get_extractors(extractor_config)
                extractor.extract(nodes)
                # save nodes
                docstore = SimpleDocumentStore()
                docstore.add_documents(nodes)
                docstore.persist(persist_path=nodes_cache_path)
                print(f"[update_database] Nodes saved to {cache_path}")
                easy_reader(cache_path, extractor_name)
                if 'need_interrupt' in extractor_index_config and extractor_index_config['need_interrupt']:
                    # Break for the rest step
                    exit()
                    
            else:
                extractor = self._get_extractors(extractor_config)
                extractor.extract(nodes)

        print("done")
        return nodes

    def _get_an_index_generator(self, index_type):
        if index_type == 'VectorStoreIndex':
            return VectorStoreIndex
        else:
            raise Exception("Invalid embedding model name. Please provide embedding models {}".format())

    def _get_a_store(self, store_type):
        if store_type == 'SimpleDocumentStore':
            return SimpleDocumentStore()
        elif store_type == 'SimpleIndexStore':
            return SimpleIndexStore()
        elif store_type == 'SimpleVectorStore':
            return SimpleVectorStore()
        elif store_type == 'SimpleGraphStore':
            return SimpleGraphStore()

    def create_or_update_indexes(self):
        self._load_configs()
        for index_id in self.config['document_preprocessing']['indexes']:
            # Load index config
            index_config = self.prefix_config['indexes'][index_id]
            print('[update_database] Updating index: {}'.format(index_id))
            
            index_dir_path = os.path.abspath(os.path.join(self.root_path, self.config['indexes_dir_path'], index_id))
            
            if not os.path.exists(index_dir_path):
                print("[update_database] Storage does not find with path: {}".format(index_dir_path))
                print("[update_database] Creating a new one...")

            nodes = None
            extractors = index_config['extractors'].copy()
            # Try to find the latest available nodes
            for extractor_index_config in index_config['extractors']:
                extractor_name, extractor_index_config = next(iter(extractor_index_config.items()))
                extractor_config = self.prefix_config['extractor'][extractor_name]
                cache_path = os.path.abspath(os.path.join(self.root_path, extractor_config['cache']))
                nodes_cache_path = os.path.abspath(os.path.join(cache_path, extractor_name+'.json'))

                if os.path.exists(nodes_cache_path) and 'force_extract' in extractor_index_config and not extractor_index_config['force_extract']:
                    print(f"[update_database] Cache {extractor_name} is detected. Now at {extractor_name} ...")
                    extractors.pop()
                    # Directly use nodes have been generated
                    docstore = SimpleDocumentStore().from_persist_path(persist_path=nodes_cache_path)
                    nodes = [node for _, node in docstore.docs.items()]
                else:
                    index_config['extractors'] = extractors
                    break
                
            if nodes is None:
                documents = self._load_documents()
                nodes = self._generate_nodes_from_documents(index_config, documents)
                nodes = self._generate_metadata_to_nodes(index_config, nodes)
            elif len(index_config['extractors']) > 0:
                nodes = self._generate_metadata_to_nodes(index_config, nodes)

            print("I shouldn't be here")
            exit()

            # Load embedding model
            embedding_config = self.prefix_config['embedding_model'][index_config['embedding_model']]
            Settings.embed_model = OllamaCustomEmbeddings(
                model_name=embedding_config['name'],
            )

            # Generate index for nodes
            storage_context_config = self.prefix_config['storage_context'][index_config['storage_context']]
            index_generator = self._get_an_index_generator(storage_context_config['index_generator'])
            index = index_generator(
                nodes=nodes,
                storage_context=StorageContext.from_defaults(
                    docstore=self._get_a_store(storage_context_config['docstore']),
                    vector_store=self._get_a_store(storage_context_config['vector_store']),
                    index_store=self._get_a_store(storage_context_config['index_store']),
                    property_graph_store=self._get_a_store(storage_context_config['property_graph_store'])
                ),
                persist_dir=index_dir_path if os.path.exists(index_dir_path) else None,
                show_progress=True
            )

            index.set_index_id(index_id)
            index.storage_context.persist(index_dir_path)
            print("[update_database] Index: {} has been saved".format(index_id))

    def get_all_index_ids(self):
        self._load_configs()
        def get_directory_size(directory):
            total_size = 0
            for dirpath, _, filenames in os.walk(directory):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            return total_size / (1024 * 1024)
        
        indexes_dir_path = os.path.abspath(os.path.join(self.root_path, self.config['indexes_dir_path']))
        indexes = []
        for d in os.listdir(indexes_dir_path):
            folderpath = os.path.join(indexes_dir_path, d)
            if os.path.isdir(folderpath):
                size = get_directory_size(folderpath)
                modified_time = os.path.getmtime(folderpath)
                modified_date = datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
                indexes.append({'id': d, 'size': size, 'modified_date': modified_date})
        return indexes

    def set_llm(self, llm_name):
        if llm_name == "vicuna:13b":
            llm = Ollama(model=llm_name, request_timeout=240.0)
        elif llm_name == "lmsys/vicuna-13b-v1.3":
            # TODO Custom Huggingface model
            llm_config = self.prefix_config['llm'][llm_name]
            llm = HuggingFace(model=llm_name)
        elif llm_name == 'gpt-4o':
            llm = llama_index_openai(model='gpt-4o', api_key=os.getenv('OPENAI_API_KEY'))
        
        Settings.llm = llm

    def load_index(self, index_id, llm_name, is_rerank):
        self._load_configs()
        # Load index config
        index_config = self.prefix_config['indexes'][index_id]
        index_dir_path = os.path.abspath(os.path.join(self.root_path, self.config['indexes_dir_path'], index_id))

        if not os.path.exists(index_dir_path):
            print("Id: {} does not exist. Please make sure your database location do have the database".format(index_id))
            return

        # Load embedding model
        embedding_config = self.prefix_config['embedding_model'][index_config['embedding_model']]
        Settings.embed_model = OllamaCustomEmbeddings(
            model_name=embedding_config['name'],
        )

        storage_context = StorageContext.from_defaults(persist_dir=index_dir_path)
        index = load_index_from_storage(storage_context, index_id=index_id)
        
        self.set_llm(llm_name)
        
        if is_rerank:
            engine = RetrieverQueryEngine.from_args(
                retriever=index.as_retriever(),
                response_synthesizer=get_response_synthesizer(response_mode=ResponseMode.GENERATION, streaming=True),
                node_postprocessors=[LLMRerank(top_n=5)]
            )
        else:
            engine = index.as_query_engine(streaming=True)

        return engine


if __name__ == '__main__':
    d = Database(config_dir_path='./code/llamaIndex')
    index = d.create_or_update_indexes()