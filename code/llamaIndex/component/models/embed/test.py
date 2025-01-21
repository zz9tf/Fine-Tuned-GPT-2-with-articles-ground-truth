import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))
import torch
import os
import numpy as np
# from component.models.embed.get_embedding_model import get_embedding_model
from component.io import load_nodes_jsonl
from configs.load_config import load_configs
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from component.models.embed.stella_en_400M_v5 import StellaEn400MV5

# cache_path = os.path.abspath('../../../.cache')
# file_name = r'gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_50_pid_32.jsonl'
# nodes = load_nodes_jsonl(os.path.join(cache_path, file_name))
# print(nodes[1].text)
# print(nodes[2].text)

_, prefix_config = load_configs()
# embed_config = prefix_config['embedding_model']['Linq-AI-Research/Linq-Embed-Mistral']

# embed_model = get_embedding_model(embedding_config=embed_config)

# torch.cuda.empty_cache()
# embed1 = embed_model._get_text_embedding(nodes[1].text)
# torch.cuda.empty_cache()
# embed2 = embed_model._get_text_embedding(nodes[2].text)
# print(embed1 == embed2)

# np.save('embed1.npy', embed1)
# np.save('embed2.npy', embed2)

cache_dir = "/home/zhengzheng/work/.hf_cache"
model = StellaEn400MV5(prefix_config['embedding_model']['dunzhang/stella_en_1.5B_v5'], device="cuda:0")

# vector_dim = 1024
# vector_linear_directory = f"2_Dense_{vector_dim}"
# model = AutoModel.from_pretrained('dunzhang/stella_en_400M_v5', trust_remote_code=True, cache_dir=cache_dir).cuda().eval()
# tokenizer = AutoTokenizer.from_pretrained('dunzhang/stella_en_400M_v5', trust_remote_code=True, cache_dir=cache_dir)
# vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=vector_dim)
# linear_vector_dir = os.path.join(cache_dir, f"stella_en_400M_v5/{vector_linear_directory}")

# vector_linear_dict = {
#     k.replace("linear.", ""): v for k, v in
#     torch.load(os.path.join(
#         linear_vector_dir, 'pytorch_model.bin'
#     )).items()
# }
# vector_linear.load_state_dict(vector_linear_dict)
# vector_linear.cuda()

query_prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
queries = [
    "What are some ways to reduce stress?",
    "What are the benefits of drinking green tea?",
]
queries = [query_prompt + query for query in queries]
# docs do not need any prompts
docs = [
    "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
    "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
]

# # Embed the queries
# with torch.no_grad():
#     input_data = tokenizer(queries, padding="longest", truncation=True, max_length=512, return_tensors="pt")
#     input_data = {k: v.cuda() for k, v in input_data.items()}
#     attention_mask = input_data["attention_mask"]
#     last_hidden_state = model(**input_data)[0]
#     last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     query_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
#     query_vectors = normalize(vector_linear(query_vectors).cpu().numpy())

# # Embed the documents
# with torch.no_grad():
#     input_data = tokenizer(docs, padding="longest", truncation=True, max_length=512, return_tensors="pt")
#     input_data = {k: v.cuda() for k, v in input_data.items()}
#     attention_mask = input_data["attention_mask"]
#     last_hidden_state = model(**input_data)[0]
#     last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     docs_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
#     docs_vectors = normalize(vector_linear(docs_vectors).cpu().numpy())
query_vectors = np.array(model._get_text_embeddings(queries))
docs_vectors = np.array(model._get_text_embeddings(docs))
print(query_vectors.shape, docs_vectors.shape)
# (2, 1024) (2, 1024)

similarities = query_vectors @ docs_vectors.T
print(similarities)