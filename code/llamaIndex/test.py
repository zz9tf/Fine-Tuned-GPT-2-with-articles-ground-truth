import gc
import torch
from custom.llm import get_llm
from custom.embedding import get_embedding_model

def show_space():
    torch.cuda.empty_cache()
    gc.collect()
    # Get the total memory allocated on the GPU
    allocated_memory = torch.cuda.memory_allocated()
    # Get the total memory cached on the GPU
    cached_memory = torch.cuda.memory_reserved()

    # Convert bytes to megabytes (MB)
    allocated_memory_mb = allocated_memory / (1024 ** 3)
    cached_memory_mb = cached_memory / (1024 ** 3)

    print(f"Allocated Memory: {allocated_memory_mb:.2f} GB")
    print(f"Cached Memory: {cached_memory_mb:.2f} GB")

llm_config = {
    "based_on": "huggingface",
    "model_name": "lmsys/vicuna-13b-v1.5",
    "cache_dir": "/work/zhengzheng/.hf_cache"
}

embedding_config = {
    'based_on': 'huggingface',
    'name': 'dunzhang/stella_en_1.5B_v5',
    'cache_dir': '/work/zhengzheng/.hf_cache'
}

llm = get_llm(None, llm_config)
del llm
show_space()
embedding = get_embedding_model(embedding_config)
del embedding
show_space()
llm = get_llm(None, llm_config)
show_space()
