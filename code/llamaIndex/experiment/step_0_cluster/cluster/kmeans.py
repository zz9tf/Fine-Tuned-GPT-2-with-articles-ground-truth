import torch

class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, device='cuda'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = device

    def fit(self, X):
        # Ensure X is a PyTorch tensor and move to device
        X = torch.tensor(X, device=self.device, dtype=torch.float32)

        n_samples, n_features = X.shape
        # Randomly initialize centroids
        indices = torch.randint(0, n_samples, (self.n_clusters,), device=self.device)
        centroids = X[indices]

        for _ in range(self.max_iter):
            # Compute distances from points to centroids
            distances = torch.cdist(X, centroids)

            # Assign clusters
            labels = torch.argmin(distances, dim=1)

            # Compute new centroids
            new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(self.n_clusters)])

            # Check for convergence
            if torch.all(torch.abs(new_centroids - centroids) < self.tol):
                break

            centroids = new_centroids

        self.centroids = centroids
        self.labels = labels

    def predict(self, X):
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        distances = torch.cdist(X, self.centroids)
        return torch.argmin(distances, dim=1).cpu().numpy()

# Example usage
if __name__ == "__main__":
    # Sample data (replace with your actual data)
    import numpy as np
    data = np.random.rand(100, 2)

    kmeans = KMeans(n_clusters=3, device='cuda')
    kmeans.fit(data)
    labels = kmeans.predict(data)

    print("Cluster centroids:\n", kmeans.centroids.cpu().numpy())
    print("Labels:\n", labels)
