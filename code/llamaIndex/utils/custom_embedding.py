import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Any, List
from transformers import AutoTokenizer, AutoModel
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from typing import List, Optional
from llama_index.core.bridge.pydantic import Field, PrivateAttr
DEFAULT_HUGGINGFACE_LENGTH = 4096

class CustomEmbeddings(BaseEmbedding):
    max_length: int = Field(
        default=DEFAULT_HUGGINGFACE_LENGTH, description="Maximum length of input.", gt=0
    )
    normalize: bool = Field(default=True, description="Normalize embeddings or not.")
    query_instruction: Optional[str] = Field(
        description="Instruction to prepend to query text."
    )
    text_instruction: Optional[str] = Field(
        description="Instruction to prepend to text."
    )
    cache_dir: Optional[str] = Field(
        description="Cache folder for Hugging Face files."
    )

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _device: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = "hkunlp/instructor-large",
        cache_dir: str = None,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        max_length: int = 4096,
        embed_batch_size: int = 10,
        normalize: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, device_map='auto')
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=model_name,
            max_length=max_length,
            normalize=normalize,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
        )

    def _last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _embed(self, texts: List[str]) -> Tensor:
    # Tokenize input texts
        batch_dict = self._tokenizer(texts, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
        
        # Forward pass through model
        with torch.no_grad():
            outputs = self._model(**batch_dict)
            embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        detailed_instruct = f'Instruct: {self.query_instruction}\nQuery: {query}'
        embeddings = self._embed([detailed_instruct])
        print(embeddings)
        print(embeddings.shape)
        return embeddings[0]
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        detailed_instruct = f'Instruct: {self.text_instruction}\nQuery: {text}'
        embeddings = self._embed([detailed_instruct])
        return embeddings[0]
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._embed(
            [f'Instruct: {self.text_instruction}\nQuery: {text}' for text in texts]
        )
        return embeddings