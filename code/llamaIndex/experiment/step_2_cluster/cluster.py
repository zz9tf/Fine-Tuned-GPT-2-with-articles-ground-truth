import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath('../..'))
import pandas as pd
from custom.kmeans import KMeans
from custom.dbscan import DBSCAN
from custom.pca import PCA
from custom.tsne import TSNE
from custom.autoencoder import AutoEncoder, train_autoencoder

def load_embeddings(file_path):
    """Load embeddings from a file if it exists."""
    if os.path.exists(file_path):
        print("Loading embeddings from file...")
        return np.load(file_path)
    else:
        print(f"File {file_path} not found.")
        sys.exit()

def apply_kmeans(embeddings, n_clusters=4, device='cuda'):
    """Apply KMeans clustering on embeddings."""
    print("Executing KMeans")
    kmeans = KMeans(n_clusters=n_clusters, device=device)
    kmeans.fit(embeddings)
    labels = kmeans.predict(embeddings)
    centroids = kmeans.centroids.cpu().numpy()
    return labels, centroids

def apply_dbscan(embeddings, eps=0.1, min_samples=5, device='cuda'):
    """Apply DBSCAN clustering on embeddings."""
    print("Executing DBSCAN")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, device=device)
    dbscan.fit(embeddings)
    labels = dbscan.predict(embeddings)
    return labels

def apply_pca(embeddings, n_components=2):
    """Apply PCA to reduce the dimensionality of embeddings."""
    print("Executing PCA")
    pca = PCA(n_components=n_components)
    embeddings_2d = pca.fit_transform(embeddings)
    return embeddings_2d

def apply_tsne(embeddings, n_components=2, n_iter=1000, learning_rate=200.0, device='cuda'):
    """Apply t-SNE on embeddings."""
    print("Executing t-SNE")
    tsne = TSNE(n_components=n_components, n_iter=n_iter, learning_rate=learning_rate, device=device)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d

def apply_autoencoder(embeddings, input_dim=4096, encoding_dim=2, epochs=50, lr=0.001, device='cuda'):
    """Apply an Autoencoder on embeddings."""
    print("Executing Autoencoder")
    ae = AutoEncoder(input_dim=input_dim, encoding_dim=encoding_dim)
    embeddings_2d = train_autoencoder(ae, embeddings, epochs=epochs, lr=lr, device=device)
    print(f"Encoded data shape: {embeddings_2d.shape}")
    return embeddings_2d

def save_to_hdf(df, file_name):
    """Save DataFrame to HDF5 format."""
    df.to_hdf(file_name, key='df', mode='w')

def main():
    # Load embeddings
    file_path = "../1_get_embedding_value/embeddings_pid_1.npy"
    embeddings = load_embeddings(file_path)
    df = pd.DataFrame(embeddings)

    # KMeans clustering on raw embeddings
    # labels_kmeans, _ = apply_kmeans(embeddings)
    # df['kmean_labels'] = labels_kmeans
    # save_to_hdf(df, "cluster_result_kmeans.h5")

    # DBSCAN clustering
    # labels_dbscan = apply_dbscan(embeddings)
    # df['dbscan_labels'] = labels_dbscan

    # PCA + KMeans clustering
    # embeddings_2d_pca = apply_pca(embeddings)
    # labels_pca, _ = apply_kmeans(embeddings_2d_pca)
    # df['PCA1'] = embeddings_2d_pca[:, 0]
    # df['PCA2'] = embeddings_2d_pca[:, 1]
    # df['kmean_labels'] = labels_pca
    # save_to_hdf(df, "cluster_result_pca.h5")

    # t-SNE + Save results
    embeddings_2d_tsne = apply_tsne(embeddings)
    exit()
    df['TSNE1'] = embeddings_2d_tsne[:, 0]
    df['TSNE2'] = embeddings_2d_tsne[:, 1]
    save_to_hdf(df, "cluster_result_tsne.h5")

    # Autoencoder + KMeans clustering
    # embeddings_2d_ae = apply_autoencoder(embeddings)
    # labels_ae, _ = apply_kmeans(embeddings_2d_ae)
    # df['AE1'] = embeddings_2d_ae[:, 0]
    # df['AE2'] = embeddings_2d_ae[:, 1]
    # df['kmean_labels'] = labels_ae
    # save_to_hdf(df, "cluster_result_ae.h5")

if __name__ == "__main__":
    main()
