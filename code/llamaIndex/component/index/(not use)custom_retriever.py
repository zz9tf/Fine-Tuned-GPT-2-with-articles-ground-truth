import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import json
import torch
from tqdm import tqdm
from typing import List, cast, Any, Optional, Sequence, Tuple
from llama_index.core.schema import QueryBundle
from llama_index.core.indices.query.embedding_utils import (
    get_top_k_embeddings,
    get_top_k_embeddings_learner,
    get_top_k_mmr_embeddings,
)
from enum import Enum
from dataclasses import dataclass
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode
)
from llama_index.core.schema import TextNode, BaseNode
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class LoadedEmbeddingResult:
    """Vector store query result."""
    embeddings: List[List[float]] = None
    ids: Optional[List[str]] = None
    offset: int=None

@dataclass
class QueryResult:
    """Vector store query result."""
    query_id: int = None
    nodes: Optional[Sequence[BaseNode]] = None
    similarities: Optional[List[float]] = None
    embeddings: List[List[float]] = None
    ids: Optional[List[str]] = None

LEARNER_MODES = {
    VectorStoreQueryMode.SVM,
    VectorStoreQueryMode.LINEAR_REGRESSION,
    VectorStoreQueryMode.LOGISTIC_REGRESSION,
}

MMR_MODE = VectorStoreQueryMode.MMR

class SimilarityMode(str, Enum):
    """Modes for similarity/distance."""

    DEFAULT = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"

def batch_similarity(query_embedding: torch.Tensor, embeddings: torch.Tensor, mode: str = SimilarityMode.DEFAULT) -> torch.Tensor:
    """Calculate similarity for a batch of embeddings."""
    if mode == SimilarityMode.EUCLIDEAN:
        # Compute the Euclidean distance
        similarities = -torch.norm(embeddings - query_embedding.unsqueeze(0), dim=1)
    elif mode == SimilarityMode.DOT_PRODUCT:
        # Compute the dot product
        similarities = torch.matmul(embeddings, query_embedding)
    else:
        # Cosine similarity
        normalized_embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        normalized_query = query_embedding / torch.norm(query_embedding)
        similarities = torch.matmul(normalized_embeddings, normalized_query)

    return similarities

def get_top_k_embeddings_gpu(
    query_embedding: List[float],
    embeddings: List[List[float]],
    similarity_top_k: Optional[int] = None,
    embedding_ids: Optional[List] = None,
    similarity_cutoff: Optional[float] = None,
    device: str = 'cuda'
) -> Tuple[List[float], List]:
    """Get top nodes by similarity to the query using PyTorch on GPU."""
    if embedding_ids is None:
        embedding_ids = list(range(len(embeddings)))
        
    # Convert input to PyTorch tensors and move them to GPU
    embeddings_tensor = torch.tensor(embeddings, device=device)
    query_embedding_tensor = torch.tensor(query_embedding, device=device)

    similarities = batch_similarity(query_embedding_tensor, embeddings_tensor)
    
    if similarity_cutoff is not None:
        mask = similarities > similarity_cutoff
        similarities = similarities[mask]
        embedding_ids = [embedding_ids[i] for i in range(len(embedding_ids)) if mask[i]]  # Retain corresponding ids
    top_similarities, top_indices = torch.topk(similarities, k=similarity_top_k)
    top_similarities = top_similarities.cpu().tolist()
    top_indices = top_indices.cpu().tolist()

    return top_similarities, top_indices

class CustomRetriever():
    def __init__(
        self, 
        file_path: str, 
        similarity_top_k: int, 
        kwargs: Any=None, 
        mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        retrieve_mode: str = 'one',
        break_num:int = None,
        batch_size: int=1000,
        worker: int = None
    ):
        self.file_path = file_path
        self.id_to_offsets = {}
        self.similarity_top_k = similarity_top_k
        self.kwargs = kwargs
        self.mode = mode
        self.retrieve_mode = retrieve_mode
        self.break_num = break_num
        self.batch_size = batch_size
        self.worker = worker if worker else torch.cuda.device_count()
    
    def load_batch_ids_and_embeddings(self, start_offset, pbar) -> LoadedEmbeddingResult:
        if self.break_num is not None and self.break_num <= 0:
            raise ValueError("break_num must be positive")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        ids = []
        embeddings = []
        current_offset = start_offset
        # Read the file and track progress based on bytes read
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(start_offset)
            for i, line in enumerate(f):
                node_data = json.loads(line)
                ids.append(len(self.id_to_offsets))
                embeddings.append(node_data['embedding'])
                
                line_length = len(line.encode('utf-8'))
                self.id_to_offsets[len(self.id_to_offsets)] = current_offset
                current_offset += line_length
                # Update progress bar
                pbar.update(line_length)
                # Check if either break condition is met
                if (self.break_num is not None and len(self.id_to_offsets) == self.break_num) \
                    or i+1 >= self.batch_size:
                    return LoadedEmbeddingResult(embeddings=embeddings, ids=ids, offset=current_offset)

        result = LoadedEmbeddingResult(
            embeddings=embeddings, ids=ids, offset=current_offset
        )
        return result
    
    # @classmethod
    # def from_nodes_file_with_all_levels(
    #     cls, 
    #     index_dir_path: str,
    #     index_id: str,
    #     retriever_kwargs: dict,
    #     break_num: int=None,
    #     worker: int=1
    # ):
        
    #     file_path = os.path.join(index_dir_path, index_id) + '.jsonl'
    #     # Generate index for nodes
    #     level_to_embeddings = {}
        
    #     try:
    #         # Get the total file size
    #         file_size = os.path.getsize(file_path)
            
    #         current_offset = 0
    #         offsets = []
            
    #         total_embedding_size = 0
    #         total_offset_size = 0
            
    #         # Read the file and track progress based on bytes read
    #         with open(file_path, 'r', encoding='utf-8') as file:
    #             with tqdm(total=file_size, desc=f'Loading {file_path.split(os.path.sep)[-1]}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
    #                 for i, line in enumerate(file):
    #                     if break_num is not None and i == break_num:
    #                         break
    #                     offsets.append(current_offset)
    #                     current_offset += len(line.encode('utf-8'))
    #                     node_data = json.loads(line)
    #                     if node_data['metadata']["level"] not in level_to_embeddings:
    #                         level_to_embeddings[node_data['metadata']["level"]] = []
    #                     level_to_embeddings[node_data['metadata']["level"]].append(node_data['embedding'])
                        
    #                     # Update the total size of embeddings and offsets
    #                     total_embedding_size += sys.getsizeof(node_data['embedding'])  # Add size of this embedding
    #                     total_offset_size += sys.getsizeof(offsets[-1])  # Add size of the last added offset

    #                     pbar.set_postfix_str(
    #                         f"Embeddings: {readable_size(total_embedding_size)}, Offsets: {readable_size(total_offset_size)}"
    #                     )
    #                     # Update progress bar based on bytes read
    #                     pbar.update(len(line.encode('utf-8')))
                        
    #     except Exception as e:
    #         print(f"An error occurred while loading nodes: {e}")
        
    #     return {level: cls(
    #         file_path=file_path,
    #         embeddings=embeddings,
    #         offsets=offsets,
    #         worker=worker,
    #         **retriever_kwargs
    #     ) for level, embeddings in level_to_embeddings.items()}

    def _query(self, query_id, query_bundle_with_embeddings: QueryBundle, embeddings: List[float], embedding_ids) -> QueryResult:
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
                similarity_top_k=query.similarity_top_k
            )
        elif query.mode == MMR_MODE:
            mmr_threshold = self.kwargs.get("mmr_threshold", None)
            top_similarities, top_ids = get_top_k_mmr_embeddings(
                query_embedding,
                embeddings,
                similarity_top_k=query.similarity_top_k,
                mmr_threshold=mmr_threshold
            )
        elif query.mode == VectorStoreQueryMode.DEFAULT:
            top_similarities, top_ids = get_top_k_embeddings(
                query_embedding,
                embeddings,
                similarity_top_k=query.similarity_top_k
            )
        else:
            raise ValueError(f"Invalid query mode: {query.mode}")

        # print(f"Job end at {embedding_ids[0]}")
        
        top_embeddings = [emb for i, emb in enumerate(embeddings) if i in top_ids]
        new_top_ids = [embedding_ids[top_id] for top_id in top_ids]
        
        return QueryResult(
            query_id=query_id,
            similarities=top_similarities, 
            ids=new_top_ids, 
            embeddings=top_embeddings
        )
    
    def _query_gpu(
        self, 
        query_id, 
        query_bundle_with_embeddings: QueryBundle, 
        embeddings: List[float], 
        embedding_ids,
        device: str = 'cuda'
    ) -> QueryResult:
        # print(f"at {device}")
        query = VectorStoreQuery(
            query_embedding=query_bundle_with_embeddings.embedding,
            similarity_top_k=self.similarity_top_k,
            query_str=query_bundle_with_embeddings.query_str,
            mode=self.mode
        )
        top_similarities, top_ids = get_top_k_embeddings_gpu(
            query.query_embedding,
            embeddings,
            similarity_top_k=query.similarity_top_k,
            device=device
        )
        
        top_embeddings = [emb for i, emb in enumerate(embeddings) if i in top_ids]
        new_top_ids = [embedding_ids[top_id] for top_id in top_ids]
        # print(f"end {device}")
        return QueryResult(
            query_id=query_id,
            similarities=top_similarities, 
            ids=new_top_ids, 
            embeddings=top_embeddings
        )
        
    def query(
        self, 
        query_id, 
        query_bundle_with_embeddings: QueryBundle, 
        embeddings: List[float], 
        embedding_ids,
        device: str = None
    ) -> QueryResult:
        if device:
            return self._query_gpu(
                query_id=query_id,
                query_bundle_with_embeddings=query_bundle_with_embeddings,
                embeddings=embeddings,
                embedding_ids=embedding_ids,
                device=device
            )
        else:
            return self._query(
                query_id=query_id,
                query_bundle_with_embeddings=query_bundle_with_embeddings,
                embeddings=embeddings,
                embedding_ids=embedding_ids
            )
    
    
    def get_node_by_offset(self, node_id):
        if node_id not in self.id_to_offsets:
            raise ValueError(f"Node id: {node_id} doesn't exist in id_to_offsets.")
        target_offset = self.id_to_offsets[node_id]  # Line numbers are 0-based
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(target_offset)  # Jump to the byte offset
            line = f.readline().strip()
            node_data = json.loads(line)
            node = TextNode.from_dict(node_data)
            return node  # Read the target line
    
    def retrieve(self, queries):
        current_offset = 0
        executor = ThreadPoolExecutor(max_workers=self.worker)
        
        # Get the total file size
        file_size = os.path.getsize(self.file_path)
        
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024)
        while current_offset < file_size:
            if self.break_num is not None and len(self.id_to_offsets) >= self.break_num:
                break
            pbar.set_description(f'Loading {self.file_path.split(os.path.sep)[-1]}')
            loaded_result = self.load_batch_ids_and_embeddings(current_offset, pbar)
            current_offset = loaded_result.offset
            
            query_id_to_result = {}
            futures = []
            for query_id, query in enumerate(queries):
                pbar.set_description(f'Processing queries {query_id+1}/{len(queries)}')
                # load ids and embeddings
                ids = loaded_result.ids
                embeddings = loaded_result.embeddings
                if query_id in query_id_to_result:
                    ids.extend(query_id_to_result[query_id].ids)
                    embeddings.extend(query_id_to_result[query_id].embeddings)
                
                future = executor.submit(
                    self.query, 
                    query_id,
                    query, 
                    embeddings, 
                    ids,
                    f'cuda:{len(futures)}'
                )
                
                futures.append(future)
                
                if len(futures) == self.worker or query_id == len(queries)-1:
                    for future in as_completed(futures):
                        result = future.result()
                        query_id_to_result[result.query_id] = result
                    futures = []
        pbar.close()
        for query_id in tqdm(query_id_to_result.keys(), desc="Retrieving nodes ..."):
            result = query_id_to_result[query_id]
            nodes = [self.get_node_by_offset(node_id) for node_id in result.ids]
            result.nodes = nodes
            query_id_to_result[query_id] = result          
        
        return query_id_to_result