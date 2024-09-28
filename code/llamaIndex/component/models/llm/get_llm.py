import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))
import torch
from transformers import BitsAndBytesConfig
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI as llama_index_openai
from component.schema import LLMTemplate
from component.models.llm.custom_huggingface_LLM import CustomHuggingFaceLLM

def get_llm(llm_config, device=None):
    if llm_config['based_on'] == "ollama":
        llm = Ollama(model=llm_config['model_name'], request_timeout=240.0, temperature=0.3)
    elif llm_config['based_on'] == 'openai':
        llm = llama_index_openai(model=llm_config['model_name'], api_key=os.getenv('OPENAI_API_KEY'))
    elif llm_config['based_on'] == "huggingface":
        # quantize to save memory
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            # bnb_4bit_use_double_quant=True,
        )

        llm = CustomHuggingFaceLLM(
            model_name=llm_config['model_name'],
            model_kwargs={
                "quantization_config": quantization_config,
                "cache_dir": llm_config['cache_dir'],
                "local_files_only": True
            },
            tokenizer_name=llm_config['model_name'],
            tokenizer_kwargs={
                "cache_dir": llm_config['cache_dir'],
                "local_files_only": True
            },
            query_wrapper_prompt=LLMTemplate.tmpl,
            max_new_tokens=4096,
            generate_kwargs={'do_sample': True, "temperature": 0.3, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.2},
            device_map=device if device is not None else 'auto'
        )
    else:
        raise Exception(f"Invalid llm based {llm_config['based_on']}")
    
    return llm