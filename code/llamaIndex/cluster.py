import numpy as np
import os, sys
import pandas as pd

file_path = "utils/embeddings_pid_1.npy"
if os.path.exists(file_path):
    # Load the existing NumPy array
    print("Loading embeddings from file...")
    # for node in nodes[:10]:
    #     print(f"node level: {node.metadata['level']}")
    embeddings = np.load(file_path)
else:
    print(f"File {file_path} not found.")
    exit()

# KMeans
from custom.kmeans import KMeans
print("executing kmeans")
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

# Apply GPU-accelerated KMeans clustering on embeddings
kmeans = KMeans(n_clusters=4, device='cuda')
kmeans.fit(embeddings)
labels = kmeans.predict(embeddings)
kmeans.centroids.cpu().numpy()

# Save results to pd
df = pd.DataFrame(embeddings)
df['kmean_labels'] = labels
df.to_hdf("cluster_result_kmeans.h5", key='df', mode='w')

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
# Apply Kmeans
kmeans_pca = KMeans(n_clusters=4, device='cuda')
kmeans_pca.fit(embeddings_2d)
labels_pca = kmeans_pca.predict(embeddings_2d)
kmeans_pca.centroids.cpu().numpy()

# Add PCA components to the DataFrame
df['PCA1'] = embeddings_2d[:, 0]
df['PCA2'] = embeddings_2d[:, 1]
df['kmean_labels'] = labels_pca
df.to_hdf("cluster_result_pca.h5", key='df', mode='w')

# # TSNE
# print("executing tsne")
# from custom.tsne import TSNE
# tsne = TSNE(n_components=2, n_iter=1000, learning_rate=200.0, device='cuda')
# embeddings_2d = tsne.fit_transform(embeddings)
# # Add TSNE components to the DataFrame
# df['TSNE1'] = embeddings_2d[:, 0]
# df['TSNE2'] = embeddings_2d[:, 1]

# df.to_hdf("cluster_result_tsne.h5", key='df', mode='w')

# # Autoencoder
# print("executing Autoencoder")
# from custom.autoencoder import AutoEncoder, train_autoencoder
# ae = AutoEncoder(input_dim=4096, encoding_dim=2) # 2D encoding
# embeddings_2d= train_autoencoder(ae, embeddings, epochs=50, lr=0.001, device='cuda')
# print("Encoded data shape:", embeddings_2d.shape)

# Apply Kmeans
# kmeans_ae = KMeans(n_clusters=4, device='cuda')
# kmeans_ae.fit(embeddings_2d)
# labels_ae = kmeans_ae.predict(embeddings_2d)
# kmeans_ae.centroids.cpu().numpy()

# # Add Autoencoder components to the DataFrame
# df['AE1'] = embeddings_2d[:, 0]
# df['AE2'] = embeddings_2d[:, 1]
# df['kmean_labels'] = labels_ae

# df.to_hdf("cluster_result_ae.h5", key='df', mode='w')
