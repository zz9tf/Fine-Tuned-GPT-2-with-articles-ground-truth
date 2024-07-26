import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
import yaml
from datetime import datetime
from llama_index.core import SimpleDirectoryReader
from custom.custom_document_reader import CustomDocumentReader
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer
from dotenv import load_dotenv
from utils.get import (
    get_parser,
    get_extractors,
    get_embedding_model,
    get_llm,
    get_an_index_generator,
    get_a_store
)
from basic.pipeline import CreateIndexPipeline

class Database():
    def __init__(self, root_path, config_dir_path):
        self.root_path = root_path
        self.config_dir_path = config_dir_path
        self._load_configs()
        database_path = os.path.abspath(os.path.join(self.root_path, self.config['indexes_dir_path']))
        os.makedirs(database_path, exist_ok=True)
        cache_path = os.path.abspath(os.path.join(self.root_path, self.config['cache']))
        os.makedirs(cache_path, exist_ok=True)

    def _load_configs(self):
        config_path = os.path.abspath(os.path.join(self.root_path, self.config_dir_path, 'config.yaml'))
        prefix_config_path = os.path.abspath(os.path.join(self.root_path, self.config_dir_path, 'prefix_config.yaml'))
        with open(config_path, 'r') as config:
            self.config = yaml.safe_load(config)
        with open(prefix_config_path, 'r') as prefix_config:
            self.prefix_config = yaml.safe_load(prefix_config)
        load_dotenv(dotenv_path=os.path.abspath(os.path.join(self.root_path, './code/llamaIndex/.env')))

    def _check_index_pipeline(self, index_pipeline):
        for step_id, step in enumerate(index_pipeline):
            assert len(step) == 1, \
                f"Invalid index pipeline. The length of index step {step_id} is {len(step)}. But the length each step should be 1."

        assert next(iter(index_pipeline[0])) == 'reader', \
            f"Invalid index pipeline. \'reader\' should be the first step of \'index pipeline\'"
        
        assert next(iter(index_pipeline[-1])) == 'storage', \
            f'Invalid index pipeline. \'storage\' should be the final step of \'index pipeline\''

    def _load_documents(self, config, **kwargs):
        print("[update_database] Loading documents ...", end=' ') 
        file_path = self.config['document_preprocessing']['data_dir_path']
        data_path = os.path.abspath(os.path.join(self.root_path, file_path))

        if config['type'] == 'SimpleDirectoryReader':
            nodes = SimpleDirectoryReader(
                input_dir=data_path,
                exclude=[],
                file_metadata=lambda file_path : {"file_path": file_path},
                filename_as_id=True
            ).load_data()
        elif config['type'] == 'CustomDocumentReader':
            cache_path = os.path.abspath(os.path.join(self.root_path, self.config['cache']))
            nodes = CustomDocumentReader(
                input_dir=data_path,
                cache_dir=cache_path,
            ).load_data()
        print("done")
        return nodes

    def _parser_documents(self, config, nodes, **kwargs):
        print("[update_database] Generating nodes from documents...", end=' ')
        parser = get_parser(self, config)
        nodes = parser.get_nodes_from_documents(
            nodes, show_progress=True
        )
        print('done')
        return nodes
    
    def _extract_metadata(self, config, nodes, action, index_id, cache_path, **kwargs):
        print(f'[update_database] Extracting metadata {config["name"]} ...')
        extractor = get_extractors(self, config)
        extractor.extract(nodes, index_id, action, cache_path)
        print("done")
        return nodes
        
    def _storage(self, index_id, index_dir_path, config, nodes, **kwargs):
        # Load embedding model
        embedding_config = self.prefix_config['embedding_model'][config['embedding_model']]
        Settings.embed_model = get_embedding_model(embedding_config)

        # Generate index for nodes
        index_generator = get_an_index_generator(config['index_generator'])
        index = index_generator(
            nodes=nodes,
            storage_context=StorageContext.from_defaults(
                docstore=get_a_store(config['docstore']),
                vector_store=get_a_store(config['vector_store']),
                index_store=get_a_store(config['index_store']),
                property_graph_store=get_a_store(config['property_graph_store'])
            ),
            persist_dir=index_dir_path if os.path.exists(index_dir_path) else None,
            show_progress=True
        )

        # Save index
        index.set_index_id(index_id)
        index.storage_context.persist(index_dir_path)
        print("[update_database] Index: {} has been saved".format(index_id))
        return index

    def _generate_pipeline(self, index_id, index_dir_path):
        index_pipeline = self.prefix_config['index_pipelines'][index_id]
        self._check_index_pipeline(index_pipeline=index_pipeline)
        pipeline = CreateIndexPipeline(
            index_id=index_id, 
            index_dir_path=index_dir_path, 
            database=self,
            index_pipeline = index_pipeline,
            delete_cache=False
        )

        return pipeline

    def create_or_update_indexes(self):
        self._load_configs()
        for index_id in self.config['document_preprocessing']['index_pipelines']:
            # Load index config
            print('[update_database] Updating index: {}'.format(index_id))
            index_dir_path = os.path.abspath(os.path.join(self.root_path, self.config['indexes_dir_path'], index_id))

            if not os.path.exists(index_dir_path):
                print("[update_database] Index does not find with path: {}".format(index_dir_path))
                print("[update_database] Creating a new one...")
            pipeline = self._generate_pipeline(index_id, index_dir_path)
            pipeline.run()

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

    def _get_index_config(self, index_pipeline):
        index_config = {}
        for step in index_pipeline:
            step_type, action  = next(iter(step))
            if step_type not in action:
                index_config[step_type] = action
            else:
                if isinstance(index_config[step_type], list):
                    index_config[step_type].append(action)
                else:    
                    index_config[step_type] = [index_config[step_type], action]

        return index_config

    def load_index(self, index_id, llm, is_rerank):
        self._load_configs()

        # if the index exist?
        index_dir_path = os.path.abspath(os.path.join(self.root_path, self.config['indexes_dir_path'], index_id))
        if not os.path.exists(index_dir_path):
            print("Id: {} does not exist. Please make sure your database location do have the database".format(index_id))
            return
        
        # Load index config and storage config
        index_pipeline = self.prefix_config['index_pipelines'][index_id]
        self._check_index_pipeline(index_pipeline)
        index_config = self._get_index_config(index_pipeline)
        storage_config = self.prefix_config['storage'][index_config['storage']]

        # Load embedding model
        embedding_config = self.prefix_config['embedding_model'][storage_config['embedding_model']]
        Settings.embed_model = get_embedding_model(embedding_config)
        
        # Load index
        storage_context = StorageContext.from_defaults(persist_dir=index_dir_path)
        index = load_index_from_storage(
            storage_context=storage_context, 
            index_id=index_id
        )

        # convert it to retriever
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
    d = Database(root_path='../../..', config_dir_path='./code/llamaIndex/configs')
    index = d.create_or_update_indexes()