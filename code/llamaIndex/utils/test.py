import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
import torch
from transformers import BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM
from custom.schema import LLMTemplate

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

# llm = HuggingFaceLLM(
#     model_name='lmsys/vicuna-13b-v1.5',
#     model_kwargs={
#         # "quantization_config": quantization_config,
#         "cache_dir": '/work/zhengzheng/.hf_cache',
#         "local_files_only": True
#     },
#     tokenizer_name='lmsys/vicuna-13b-v1.5',
#     tokenizer_kwargs={
#         "cache_dir": '/work/zhengzheng/.hf_cache',
#         "local_files_only": True
#     },
#     # context_window=3900,
#     query_wrapper_prompt=LLMTemplate.tmpl,
#     max_new_tokens=4096,
#     generate_kwargs={'do_sample': True, "temperature": 0.3, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.2}, # , 
#     device_map="auto"
# )

# user_input = "test"
# print("User:", user_input)
# print("LLM response:", llm.complete(user_input))

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="Linq-AI-Research/Linq-Embed-Mistral",
    cache_folder="/work/zhengzheng/.hf_cache"
)

embeddings = embed_model.get_text_embedding("Hello World!")
print(len(embeddings))
print(embeddings[:5])

