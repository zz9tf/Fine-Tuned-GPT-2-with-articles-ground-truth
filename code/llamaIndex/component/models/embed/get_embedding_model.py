import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))
from typing import Any, List
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from typing import List, Optional
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from component.models.embed.Ollama_Embed import OllamaEmbed
from component.models.embed.Linq_Embed_Mistral import LinqEmbedMistral

def get_embedding_model(embedding_config, device=None):
    if embedding_config["based_on"] == 'huggingface':
        if embedding_config['name'] == 'Linq-AI-Research/Linq-Embed-Mistral':
            return LinqEmbedMistral(embedding_config=embedding_config, device=device)
    elif embedding_config["based_on"] == 'ollama':
        return OllamaEmbed(embedding_config=embedding_config, device=device)

