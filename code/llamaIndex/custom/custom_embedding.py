from typing import Any, List
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from typing import List, Optional
from llama_index.core.bridge.pydantic import Field, PrivateAttr

from llama_index.embeddings.ollama import OllamaEmbedding
from custom.custom_huggingface_embedding import CustomHuggingFaceEmbedding

class CustomEmbedding(BaseEmbedding):
    query_instruction: Optional[str] = Field(
        description="Instruction to prepend to query text."
    )
    text_instruction: Optional[str] = Field(
        description="Instruction to prepend to text."
    )
    _model: Any = PrivateAttr()

    def __init__(
        self,
        embedding_config: dict,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        max_length: int = 4096,
        embed_batch_size: int = 10,
        normalize: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        if embedding_config["based_on"] == 'huggingface':
            self._model = CustomHuggingFaceEmbedding(
                model_name=embedding_config['name'],
                cache_folder=embedding_config['cache_dir']
            )
        elif embedding_config["based_on"] == 'ollama':
            self._model = OllamaEmbedding(
                model_name=embedding_config['name'],
                base_url="http://localhost:11434",
                ollama_additional_kwargs={"mirostat": 0},
            )
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=embedding_config['name'],
            max_length=max_length,
            normalize=normalize,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
        )

    def _get_query_embedding(self, query: str) -> List[float]:
        detailed_instruct = f'Query Instruct: {self.query_instruction}\nQuery: {query}'
        return self._model.get_query_embedding(detailed_instruct)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        detailed_instruct = f'Text Instruct: {self.text_instruction}\nText: {text}'
        return self._model.get_text_embedding_batch(
            [detailed_instruct], show_progress=True
        )
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.get_text_embedding_batch(
            [f'Text Instruct: {self.text_instruction}\nText: {text}' for text in texts]
        )
        return embeddings