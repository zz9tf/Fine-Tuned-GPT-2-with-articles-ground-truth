from typing import Any, List, Optional, TypedDict, Dict
import gc
import numpy as np
import torch
import time
from enum import Enum
from custom.embedding import get_embedding_model
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.schema import BaseNode

class SimilarityMode(str, Enum):
    """Modes for similarity/distance."""

    DEFAULT = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"

def similarity(
    embedding1,
    embedding2,
    mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> float:
    """Get embedding similarity."""
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    if mode == SimilarityMode.EUCLIDEAN:
        # Using -euclidean distance as similarity to achieve same ranking order
        return -float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
    elif mode == SimilarityMode.DOT_PRODUCT:
        if len(np.array(embedding1).shape) == 0:
            return embedding1*embedding2
        if len(np.array(embedding1).shape) == 1:
            return np.dot(embedding1, embedding2)
        if len(np.array(embedding1).shape) == 2:
            return embedding1 @ embedding2
        
    else:
        if len(np.array(embedding1).shape) == 0:
            product = embedding1*embedding2
        if len(np.array(embedding1).shape) == 1:
            product = np.dot(embedding1, embedding2)
        if len(np.array(embedding1).shape) == 2:
            product = embedding1 @ embedding2
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return product / norm

class SentenceCombination(TypedDict):
    sentence: str
    index: int
    combined_sentence: str
    combined_sentence_embedding: List[float]

class SemanticSplitter():
    """Semantic node parser.

    Splits a document into Nodes, with each node being a group of semantically related sentences.

    Args:
        buffer_size (int): number of sentences to group together when evaluating semantic similarity
        embed_model: (BaseEmbedding): embedding model to use
        sentence_splitter (Optional[Callable]): splits text into sentences
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships
    """
    @classmethod
    def class_name(cls) -> str:
        return "SemanticSplitterNodeParser"

    def __init__(
        self,
        embed_model_config: Dict,
        breakpoint_percentile_threshold: Optional[int] = 95,
        buffer_size: Optional[int] = 1,
    ) -> "SemanticSplitter":
        sentence_splitter = split_by_sentence_tokenizer()
        self.embed_model_config = embed_model_config
        self.breakpoint_percentile_threshold=breakpoint_percentile_threshold
        self.buffer_size=buffer_size
        self.sentence_splitter=sentence_splitter

    def load_embed(self):
        self.embed_model = get_embedding_model(self.embed_model_config)

    def del_embed(self):
        del self.embed_model
        self.embed_model = None
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        
        free_memory, total_memory = torch.cuda.mem_get_info()
    
        # Convert the values from bytes to megabytes (MB)
        free_memory_MB = free_memory / (1024 ** 3)
        total_memory_MB = total_memory / (1024 ** 3)
        
        print(f"Free memory: {free_memory_MB:.2f} GB")
        print(f"Used memory: {total_memory_MB - free_memory_MB:.2f} GB")
        print(f"Total memory: {total_memory_MB:.2f} GB")

    def parse_text(
        self,
        text: List[str],
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse text"""
        
        text_splits = self.sentence_splitter(text)
        sentences = self._build_sentence_groups(text_splits)
        combined_sentence_embeddings = []
        for s in sentences:
            with torch.no_grad():
                embedding = self.embed_model._get_text_embedding(s["combined_sentence"])
            torch.cuda.empty_cache()
            gc.collect()
            combined_sentence_embeddings.append(embedding)
        for i, embedding in enumerate(combined_sentence_embeddings):
            sentences[i]["combined_sentence_embedding"] = embedding

        distances = self._calculate_distances_between_sentence_groups(sentences)

        chunks = self._build_node_chunks(sentences, distances)

        return chunks

    def _build_sentence_groups(
        self, text_splits: List[str]
    ) -> List[SentenceCombination]:
        sentences: List[SentenceCombination] = [
            {
                "sentence": x,
                "index": i,
                "combined_sentence": "",
                "combined_sentence_embedding": [],
            }
            for i, x in enumerate(text_splits)
        ]

        # Group sentences and calculate embeddings for sentence groups
        for i in range(len(sentences)):
            combined_sentence = ""

            for j in range(i - self.buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]["sentence"]

            combined_sentence += sentences[i]["sentence"]

            for j in range(i + 1, i + 1 + self.buffer_size):
                if j < len(sentences):
                    combined_sentence += sentences[j]["sentence"]

            sentences[i]["combined_sentence"] = combined_sentence

        return sentences

    def _calculate_distances_between_sentence_groups(
        self, sentences: List[SentenceCombination]
    ) -> List[float]:
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]["combined_sentence_embedding"]
            embedding_next = sentences[i + 1]["combined_sentence_embedding"]

            sim = similarity(embedding_current, embedding_next)

            distance = 1 - sim

            distances.append(distance)

        return distances

    def _build_node_chunks(
        self, sentences: List[SentenceCombination], distances: List[float]
    ) -> List[str]:
        chunks = []
        if len(distances) > 0:
            breakpoint_distance_threshold = np.percentile(
                distances, self.breakpoint_percentile_threshold
            )

            indices_above_threshold = [
                i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
            ]

            # Chunk sentences into semantic groups based on percentile breakpoints
            start_index = 0

            for index in indices_above_threshold:
                group = sentences[start_index : index + 1]
                combined_text = "".join([d["sentence"] for d in group])
                chunks.append(combined_text)

                start_index = index + 1

            if start_index < len(sentences):
                combined_text = "".join(
                    [d["sentence"] for d in sentences[start_index:]]
                )
                chunks.append(combined_text)
        else:
            # If, for some reason we didn't get any distances (i.e. very, very small documents) just
            # treat the whole document as a single node
            chunks = [" ".join([s["sentence"] for s in sentences])]

        return chunks
