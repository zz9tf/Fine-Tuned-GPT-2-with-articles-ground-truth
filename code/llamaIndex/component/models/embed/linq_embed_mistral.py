import gc
from typing import Any, List
from transformers import AutoTokenizer, AutoModel
from transformers.models.mistral.modeling_mistral import MistralModel
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.core.embeddings import BaseEmbedding


class LinqEmbedMistral(BaseEmbedding):
    max_length: int = Field(
        default=4096, description="Maximum length of input.", gt=0
    )
    _model: MistralModel = PrivateAttr()
    _tokenizer: LlamaTokenizerFast = PrivateAttr()
    _device: torch.device = PrivateAttr()

    def __init__(
        self,
        embedding_config,
        device,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        # print(self._device)
            
        self._model = AutoModel.from_pretrained(
            "Linq-AI-Research/Linq-Embed-Mistral", 
            cache_dir=embedding_config['cache_dir']
        ).to(self._device)
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            "Linq-AI-Research/Linq-Embed-Mistral", 
            cache_dir=embedding_config['cache_dir']
        )

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def generate_embeddings(self, input_texts):
        # Tokenize the input texts
        with torch.no_grad():
            batch_dict = self._tokenizer(input_texts, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
            batch_dict = {k: v.to(self._device) for k, v in batch_dict.items()}
            outputs = self._model(**batch_dict)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            embeddings_list = embeddings.tolist()
        del embeddings
        torch.cuda.empty_cache()
        gc.collect()
        free_memory, total_memory = torch.cuda.mem_get_info()
    
        # Convert the values from bytes to megabytes (MB)
        free_memory_MB = free_memory / (1024 ** 3)
        total_memory_MB = total_memory / (1024 ** 3)
        
        print(f"Free memory: {free_memory_MB:.2f} GB")
        print(f"Used memory: {total_memory_MB - free_memory_MB:.2f} GB")
        print(f"Total memory: {total_memory_MB:.2f} GB")
        return embeddings_list
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    async def _aget_query_embedding(self, query: str, task_description: str=None) -> List[float]:
        return self._get_query_embedding(query, task_description)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str, task_description: str=None) -> List[float]:
        if task_description != None:
            query = self.get_detailed_instruct(task_description, query)
        embeddings = self.generate_embeddings([query])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self.generate_embeddings([text])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.generate_embeddings(texts)
        return embeddings