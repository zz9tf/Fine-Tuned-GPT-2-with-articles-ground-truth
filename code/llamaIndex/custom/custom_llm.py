import os
from typing import Optional, List, Mapping, Any
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms.callbacks import llm_completion_callback
from transformers import BitsAndBytesConfig

class CustomHuggingFaceLLM(CustomLLM):
    model_name: str = Field(
        description="The huggingface model to use."
    )

    context_window: int = Field(
        default=3900,
        description="The maximum number of context tokens for the model.",
        gt=0
    )

    max_new_tokens: int = Field(
        default=256,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    
    cache_dir: str = Field(
        default=None,
        description="The dir path of model cache"
    )

    stopping_ids: List[int] = Field(
        default_factory=list,
        description=(
            "The stopping ids to use. "
            "Generation stops when these token IDs are predicted."
        ),
    )

    model_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during initialization.",
    )

    generate_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during generation.",
    )

    num_output: int = 8192
    
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _stopping_criteria: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        context_window: int,
        max_new_tokens: int,
        is_chat_model: bool,
        cache_dir: str
    ):
        pass

    @classmethod
    def class_name(cls) -> str:
        return "custom_huggingface_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # TODO: generate response here
        return CompletionResponse(text=self.dummy_response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)