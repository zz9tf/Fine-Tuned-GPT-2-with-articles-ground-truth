import os
import yaml
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex,
    PropertyGraphIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core import Settings
from utils.custom_embedding import CustomHuggingfaceBasedEmbedding, CustomOllamaBasedEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer
from dotenv import load_dotenv
from utils.gpt_4o_json_reader import easy_reader
from utils.get import (
    load_documents,
    get_parser,
    get_extractors,
    get_embedding_model,
    get_llm,
    get_an_index_generator,
    get_a_store
)
from utils.evaluate_execution_time import evaluate_time

class Database():
    def __init__(self, config_dir_path):
        self.root_path = "../.."
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

    def _generate_nodes_from_documents(self, index_config, documents):
        print("[update_database] Generating nodes from documents...", end=' ')
        
        parser_config = self.prefix_config['parser'][index_config['parser']]
        parser = get_parser(self, parser_config)
        nodes = parser.get_nodes_from_documents(
            documents, show_progress=True
        )
        print('done')

        return nodes

    def _extract_and_add_metadata_to_nodes(self, index_config, nodes):
        print('[update_database] Extracting metadata ...')
        for extractor_index_config in index_config['extractors']:
            extractor_name = None

            if type(extractor_index_config) == dict:
                extractor_name, extractor_index_config = next(iter(extractor_index_config.items()))
                extractor_config = self.prefix_config['extractor'][extractor_name]
            else:
                extractor_name = extractor_index_config
                extractor_config = self.prefix_config['extractor'][extractor_name]
            
            print(f"Doing {extractor_name} ...")
            if extractor_config['llm'] == 'gpt-4o':
                # Create a cache for gpt-4o results
                cache_path = os.path.abspath(os.path.join(self.root_path, extractor_config['cache']))
                os.makedirs(cache_path, exist_ok=True)

                # Check if the nodes cache exists
                nodes_cache_path = os.path.abspath(os.path.join(cache_path, extractor_name+'.json'))
                # If not, create nodes cache
                extractor = get_extractors(self, extractor_config)
                extractor.extract(nodes)
                # save nodes
                docstore = get_a_store('SimpleDocumentStore')
                docstore.add_documents(nodes)
                docstore.persist(persist_path=nodes_cache_path)
                print(f"[update_database] Nodes saved to {cache_path}")
                easy_reader(cache_path, extractor_name)
                if 'need_interrupt' in extractor_index_config and extractor_index_config['need_interrupt']:
                    # Break for the rest step
                    exit()
                    
            else:
                extractor = get_extractors(self, extractor_config)
                extractor.extract(nodes)

        print("done")
        return nodes
        
    def _update_to_latest_extractors(self, index_config):
        nodes = None
        unfinished_extractors = []
        # Try to find the latest available nodes
        for extractor_index_config in index_config['extractors']:
            if type(extractor_index_config) == str:
                unfinished_extractors.append(extractor_index_config)
                continue

            extractor_name, extractor_index_config = next(iter(extractor_index_config.items()))
            extractor_config = self.prefix_config['extractor'][extractor_name]
            cache_path = os.path.abspath(os.path.join(self.root_path, extractor_config['cache']))
            nodes_cache_path = os.path.abspath(os.path.join(cache_path, extractor_name+'.json'))

            if os.path.exists(nodes_cache_path) and 'force_extract' in extractor_index_config and not extractor_index_config['force_extract']:
                print(f"[update_database] Cache {extractor_name} is detected. Now at {extractor_name} ...")
                unfinished_extractors = []
                # Directly use nodes have been generated
                docstore = get_a_store('SimpleDocumentStore').from_persist_path(persist_path=nodes_cache_path)
                nodes = [node for _, node in docstore.docs.items()]
            else:
                unfinished_extractors.append({extractor_name: extractor_index_config})

        return nodes, unfinished_extractors

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
            if 'extractors' in index_config:
                nodes, index_config['extractors'] = self._update_to_latest_extractors(index_config)

            if nodes is None:
                reader_config = self.prefix_config['reader'][index_config['reader']]
                documents = load_documents(self, reader_config)
                nodes = self._generate_nodes_from_documents(index_config, documents)
            
            if 'extractors' in index_config and len(index_config['extractors']) > 0:
                nodes = self._extract_and_add_metadata_to_nodes(index_config, nodes)

            # Load embedding model
            embedding_config = self.prefix_config['embedding_model'][index_config['embedding_model']]
            Settings.embed_model = get_embedding_model(embedding_config)

            # Generate index for nodes
            storage_context_config = self.prefix_config['storage_context'][index_config['storage_context']]
            index_generator = get_an_index_generator(storage_context_config['index_generator'])
            index = index_generator(
                nodes=nodes,
                storage_context=StorageContext.from_defaults(
                    docstore=get_a_store(storage_context_config['docstore']),
                    vector_store=get_a_store(storage_context_config['vector_store']),
                    index_store=get_a_store(storage_context_config['index_store']),
                    property_graph_store=get_a_store(storage_context_config['property_graph_store'])
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

    def load_index(self, index_id, llm, is_rerank):
        self._load_configs()
        # Load index config
        index_config = self.prefix_config['indexes'][index_id]
        index_dir_path = os.path.abspath(os.path.join(self.root_path, self.config['indexes_dir_path'], index_id))

        if not os.path.exists(index_dir_path):
            print("Id: {} does not exist. Please make sure your database location do have the database".format(index_id))
            return

        # Load embedding model
        embedding_config = self.prefix_config['embedding_model'][index_config['embedding_model']]
        Settings.embed_model = get_embedding_model(embedding_config)

        # Load storage_context
        storage_context = StorageContext.from_defaults(persist_dir=index_dir_path)
        
        # Load index and convert it to retriever
        index = load_index_from_storage(storage_context, index_id=index_id)
        retriever = None
        parser_config = self.prefix_config['parser'][index_config['parser']]
        if parser_config['retriever'] == 'BaseRetriever':
            retriever = index.as_retriever()
        elif parser_config['retriever'] == 'AutoMergingRetriever':
            nodes = [node for _, node in index._docstore.docs.items()]
            leaf_nodes = get_leaf_nodes(nodes)
            index = VectorStoreIndex(
                leaf_nodes,
                storage_context=storage_context
            )

            retriever = AutoMergingRetriever(index.as_retriever(), storage_context, verbose=True)

        # Set llm
        llm_config = self.prefix_config['llm'][llm]
        Settings.llm = get_llm(llm_config)
        
        # Set if it's ReRank
        if is_rerank:
            engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                response_synthesizer=get_response_synthesizer(response_mode=ResponseMode.GENERATION, streaming=True),
                node_postprocessors=[LLMRerank(top_n=5)]
            )
        else:
            engine = index.as_query_engine(streaming=True)

        return engine


if __name__ == '__main__':
    d = Database(config_dir_path='./code/llamaIndex')
    index = d.create_or_update_indexes()