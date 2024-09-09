import os
# Load embedding model
print("loading embedding model")
import yaml
from custom.embedding import get_embedding_model
prefix_config_path = "/home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/configs/prefix_config.yaml"
with open(prefix_config_path, 'r') as prefix_config:
    prefix_config = yaml.safe_load(prefix_config)
embed_config = prefix_config['embedding_model']['Linq-AI-Research/Linq-Embed-Mistral']
embedding_model = get_embedding_model(embed_config)

# Load questions
print("loading nodes")
import pandas as pd
cache_path = "/home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/.save"
file_name = "gpt-4o-batch-all-p_2_parser_ManuallyHierarchicalNodeParser_8165_gpu_V100_nodeNum_200_pid_1_extract.csv"
df = pd.read_csv(os.path.join(cache_path, file_name))
texts = df['Question']

# Get embedding for nodes
print("get embedding from texts")
import torch
import numpy as np
from tqdm import tqdm
file_path = "embeddings.npy"
if os.path.exists(file_path):
    # Load the existing NumPy array
    print("Loading embeddings from file...")
    embeddings = np.load(file_path)
else:
    # Initialize an empty list to collect embeddings
    embeddings = []
    for text in tqdm(texts, desc="get embeddings ..."):
        with torch.no_grad():
            embedding = embedding_model._get_text_embedding(text)
        torch.cuda.empty_cache()
        embeddings.append(embedding)
    embeddings = np.array(embeddings)
    np.save(file_path, embeddings)

# KMeans
print("executing kmeans")
import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
from custom.kmeans import KMeans

# Apply GPU-accelerated KMeans clustering
kmeans = KMeans(n_clusters=4, device='cuda')
kmeans.fit(embeddings)
labels = kmeans.predict(embeddings)
kmeans.centroids.cpu().numpy()

# Save results to pd
import pandas as pd
df = pd.DataFrame(embeddings)
df['kmean_labels'] = labels

# # DBSCAN
# print("executing DBSCAN")
# from custom.dbscan import DBSCAN
# dbscan = DBSCAN(eps=0.1, min_samples=5, device='cuda')
# dbscan.fit(embeddings)
# labels = dbscan.predict(embeddings)

# # Save results to pd
# df['dbscan_labels'] = labels

# PCA
print("executing pca")
from custom.pca import PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)
# Add PCA components to the DataFrame
df['PCA1'] = embeddings_2d[:, 0]
df['PCA2'] = embeddings_2d[:, 1]
df.to_hdf("cluster_result.h5", key='df', mode='w')

# TSNE
print("executing tsne")
from custom.tsne import TSNE
tsne = TSNE(n_components=2, n_iter=1000, learning_rate=200.0, device='cuda')
embeddings_2d = tsne.fit_transform(embeddings)
# Add TSNE components to the DataFrame
df['TSNE1'] = embeddings_2d[:, 0]
df['TSNE2'] = embeddings_2d[:, 1]

df.to_hdf("cluster_result.h5", key='df', mode='w')
