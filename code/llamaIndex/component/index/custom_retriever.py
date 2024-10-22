import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import math
import json
from tqdm import tqdm
from typing import List, cast, Any
from llama_index.core.schema import QueryBundle
from llama_index.core.indices.query.embedding_utils import (
    get_top_k_embeddings,
    get_top_k_embeddings_learner,
    get_top_k_mmr_embeddings,
)
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.schema import TextNode
from concurrent.futures import ThreadPoolExecutor, as_completed

LEARNER_MODES = {
    VectorStoreQueryMode.SVM,
    VectorStoreQueryMode.LINEAR_REGRESSION,
    VectorStoreQueryMode.LOGISTIC_REGRESSION,
}

MMR_MODE = VectorStoreQueryMode.MMR

def readable_size(size_bytes):
    """Helper function to convert bytes into a human-readable format (KB, MB, GB)."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s:6.2f} {size_name[i]:2}"

class CustomRetriever():
    def __init__(
        self, 
        file_path: str, 
        embeddings: List[float],
        offsets: List[int],
        similarity_top_k: int, 
        kwargs: Any=None, 
        mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        worker: int = 10
    ):
        self.file_path = file_path
        self.embeddings = embeddings
        self.offsets = offsets
        self.similarity_top_k = similarity_top_k
        self.kwargs = kwargs
        self.mode = mode
        self.worker = worker
    
    @classmethod
    def from_nodes_file_with_all_levels(
        cls, 
        index_dir_path: str,
        index_id: str,
        retriever_kwargs: dict,
        break_num: int=None,
        worker: int=1
    ):
        
        file_path = os.path.join(index_dir_path, index_id) + '.jsonl'
        # Generate index for nodes
        level_to_embeddings = {}
        
        try:
            # Get the total file size
            file_size = os.path.getsize(file_path)
            
            current_offset = 0
            offsets = []
            
            total_embedding_size = 0
            total_offset_size = 0
            
            # Read the file and track progress based on bytes read
            with open(file_path, 'r', encoding='utf-8') as file:
                with tqdm(total=file_size, desc=f'Loading {file_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                    for i, line in enumerate(file):
                        if break_num is not None and i == break_num:
                            break
                        offsets.append(current_offset)
                        current_offset += len(line.encode('utf-8'))
                        node_data = json.loads(line)
                        if node_data['metadata']["level"] not in level_to_embeddings:
                            level_to_embeddings[node_data['metadata']["level"]] = []
                        level_to_embeddings[node_data['metadata']["level"]].append(node_data['embedding'])
                        
                        # Update the total size of embeddings and offsets
                        total_embedding_size += sys.getsizeof(node_data['embedding'])  # Add size of this embedding
                        total_offset_size += sys.getsizeof(offsets[-1])  # Add size of the last added offset

                        pbar.set_postfix_str(
                            f"Embeddings: {readable_size(total_embedding_size)}, Offsets: {readable_size(total_offset_size)}"
                        )
                        # Update progress bar based on bytes read
                        pbar.update(len(line.encode('utf-8')))
                        
        except Exception as e:
            print(f"An error occurred while loading nodes: {e}")
        
        return {level: cls(
            file_path=file_path,
            embeddings=embeddings,
            offsets=offsets,
            worker=worker,
            **retriever_kwargs
        ) for level, embeddings in level_to_embeddings.items()}

    @classmethod
    def from_nodes_file(
        cls, 
        index_dir_path: str,
        index_id: str,
        retriever_kwargs: dict,
        break_num: int=None,
        worker: int=1
    ):
        
        file_path = os.path.join(index_dir_path, index_id) + '.jsonl'
        # Generate index for nodes
        embeddings = []
        
        try:
            # Get the total file size
            file_size = os.path.getsize(file_path)
            
            current_offset = 0
            offsets = []
            
            total_embedding_size = 0
            total_offset_size = 0
            
            # Read the file and track progress based on bytes read
            with open(file_path, 'r', encoding='utf-8') as file:
                with tqdm(total=file_size, desc=f'Loading {file_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                    for i, line in enumerate(file):
                        if break_num is not None and i == break_num:
                            break
                        offsets.append(current_offset)
                        current_offset += len(line.encode('utf-8'))
                        node_data = json.loads(line)
                        embeddings.append(node_data['embedding'])
                        
                        # Update the total size of embeddings and offsets
                        total_embedding_size += sys.getsizeof(node_data['embedding'])  # Add size of this embedding
                        total_offset_size += sys.getsizeof(offsets[-1])  # Add size of the last added offset

                        pbar.set_postfix_str(
                            f"Embeddings: {readable_size(total_embedding_size)}, Offsets: {readable_size(total_offset_size)}"
                        )
                        # Update progress bar based on bytes read
                        pbar.update(len(line.encode('utf-8')))
                        
        except Exception as e:
            print(f"An error occurred while loading nodes: {e}")
        
        return cls(
            file_path=file_path,
            embeddings=embeddings,
            offsets=offsets,
            worker=worker,
            **retriever_kwargs
        )

    def get_node_by_offset(self, line_number):
        if line_number < 0 or line_number >= len(self.offsets):
            raise ValueError(f"Line number {line_number} is out of range.")
        target_offset = self.offsets[line_number]  # Line numbers are 0-based
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(target_offset)  # Jump to the byte offset
            line = f.readline().strip()
            node_data = json.loads(line)
            node = TextNode.from_dict(node_data)
            return node  # Read the target line

    def query(self, query_bundle_with_embeddings: QueryBundle, embeddings: List[float], embedding_ids) -> VectorStoreQueryResult:
        """Get nodes for response."""            
        # print(f"Job start at {embedding_ids[0]}")
        query = VectorStoreQuery(
            query_embedding=query_bundle_with_embeddings.embedding,
            similarity_top_k=self.similarity_top_k,
            query_str=query_bundle_with_embeddings.query_str,
            mode=self.mode
        )

        query_embedding = cast(List[float], query.query_embedding)

        if query.mode in LEARNER_MODES:
            top_similarities, top_ids = get_top_k_embeddings_learner(
                query_embedding,
                embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=embedding_ids
            )
        elif query.mode == MMR_MODE:
            mmr_threshold = self.kwargs.get("mmr_threshold", None)
            top_similarities, top_ids = get_top_k_mmr_embeddings(
                query_embedding,
                embeddings,
                similarity_top_k=query.similarity_top_k,
                mmr_threshold=mmr_threshold,
                embedding_ids=embedding_ids
            )
        elif query.mode == VectorStoreQueryMode.DEFAULT:
            top_similarities, top_ids = get_top_k_embeddings(
                query_embedding,
                embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=embedding_ids
            )
        else:
            raise ValueError(f"Invalid query mode: {query.mode}")

        # print(f"Job end at {embedding_ids[0]}")
        
        return VectorStoreQueryResult(similarities=top_similarities, ids=top_ids)
        
    def retrieve(self, query_bundle_with_embeddings):
        futures = []
        executor = ThreadPoolExecutor(max_workers=self.worker)
        
        embedding_size = math.ceil(len(self.embeddings)/self.worker)
        embedding_ids = list(range(len(self.embeddings)))
        
        for i in range(self.worker):
            sub_embeddings = self.embeddings[i*embedding_size : (i+1)*embedding_size]
            sub_embedding_ids = embedding_ids[i*embedding_size : (i+1)*embedding_size]
            
            future = executor.submit(
                self.query, 
                query_bundle_with_embeddings, 
                sub_embeddings, 
                sub_embedding_ids
            )
            
            futures.append(future)
            
        embeddings = []
        embedding_ids = []
        for future in as_completed(futures):
            result = future.result()
            embedding_ids.extend(result.ids)
            embeddings.extend([self.embeddings[em_id] for em_id in result.ids])
            
        result = self.query(query_bundle_with_embeddings, embeddings, embedding_ids)
            
        retrieved_nodes = [self.get_node_by_offset(top_id) for top_id in result.ids]
        return retrieved_nodes