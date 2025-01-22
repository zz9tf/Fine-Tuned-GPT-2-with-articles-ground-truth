from typing import Optional, List, Mapping, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
# from transformers import LlamaForCausalLM, LlamaTokenizer

class SampleLLM:
    def __init__(
        self,
        model_name: str,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        tokenizer_name: Optional[str] = None,
        tokenizer_kwargs: Optional[Mapping[str, Any]] = None,
        query_wrapper_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        context_window: int = 4096,
        generate_kwargs: Optional[Mapping[str, Any]] = None,
        device_map: str = "auto",
    ):
        # Set up model and tokenizer
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.query_wrapper_prompt = query_wrapper_prompt
        self.max_new_tokens = max_new_tokens
        self.generate_kwargs = generate_kwargs or {}
        # Initialize tokenizer
        tokenizer_kwargs = tokenizer_kwargs or {}
        if "max_length" not in tokenizer_kwargs:
            tokenizer_kwargs["max_length"] = context_window
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, **(tokenizer_kwargs or {})
        )
        # self.tokenizer = LlamaTokenizer.from_pretrained(
        #     self.tokenizer_name, **(tokenizer_kwargs or {})
        # )

        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map=device_map, **(model_kwargs or {})
        )
        self.device = next(self.model.parameters()).device
    
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt (str): Input prompt for the model.

        Returns:
            str: Generated completion.
        """
        # Apply query wrapper prompt if provided
        if self.query_wrapper_prompt:
            prompt = self.query_wrapper_prompt.format(query_str=prompt)

        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate output
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            **self.generate_kwargs
        )

        # Decode and return output
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text
    