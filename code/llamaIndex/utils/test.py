import os
import torch
from transformers import BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM

# from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print('configging ...')

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    # bnb_4bit_use_double_quant=True,
)

print('LLM')

llm = HuggingFaceLLM(
    model_name='lmsys/vicuna-13b-v1.3',
    tokenizer_name='lmsys/vicuna-13b-v1.3',
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={
        "quantization_config": quantization_config,
        "cache_dir": '/work/zhengzheng/.hf_cache',
        "token": "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",
    },
    generate_kwargs={"temperature": 0.3, "top_k": 50, "top_p": 0.95},
    device_map="auto"
)

print('Completing ...')

print(llm.complete("Hello, here is a test if llm is working."))

