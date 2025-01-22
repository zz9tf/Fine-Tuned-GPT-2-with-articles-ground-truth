import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))
from component.models.embed.ollama_embed import OllamaEmbed
from component.models.embed.linq_embed_mistral import LinqEmbedMistral
from component.models.embed.stella_en_400M_v5 import StellaEn400MV5

def get_embedding_model(embedding_config, device=None):
    if embedding_config["based_on"] == 'huggingface':
        if embedding_config['name'] == 'Linq-AI-Research/Linq-Embed-Mistral':
            return LinqEmbedMistral(embedding_config=embedding_config, device=device)
        elif embedding_config["name"] == 'dunzhang/stella_en_400M_v5':
            return StellaEn400MV5(embedding_config=embedding_config, device=device)
    elif embedding_config["based_on"] == 'ollama':
        return OllamaEmbed(embedding_config=embedding_config, device=device)
    

