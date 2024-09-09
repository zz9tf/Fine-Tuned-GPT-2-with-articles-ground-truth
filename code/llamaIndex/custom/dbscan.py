import torch

class DBSCAN:
    def __init__(self, eps, min_samples, device='cuda'):
        self.eps = eps
        self.min_samples = min_samples
        self.device = device

    def fit(self, X):
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        n_samples = X.size(0)
        labels = -torch.ones(n_samples, dtype=torch.int32, device=self.device)
        cluster_id = 0

        # Compute distance matrix
        distances = torch.cdist(X, X)

        for i in range(n_samples):
            if labels[i] != -1:
                continue

            # Find neighbors
            neighbors = (distances[i] < self.eps).nonzero(as_tuple=True)[0]

            if neighbors.size(0) < self.min_samples:
                labels[i] = -1
                continue

            # Start a new cluster
            self._expand_cluster(X, labels, neighbors, cluster_id, distances)
            cluster_id += 1

        self.labels = labels.cpu().numpy()

    def _expand_cluster(self, X, labels, neighbors, cluster_id, distances):
        queue = neighbors.tolist()
        labels[neighbors] = cluster_id

        while queue:
            current = queue.pop(0)
            if distances[current].size(0) < self.min_samples:
                continue

            current_neighbors = (distances[current] < self.eps).nonzero(as_tuple=True)[0]
            for neighbor in current_neighbors:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                    queue.append(neighbor)
                elif labels[neighbor] == -1:
                    labels[neighbor] = cluster_id

    def predict(self, X):
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        distances = torch.cdist(X, X)
        n_samples = X.size(0)
        labels = -torch.ones(n_samples, dtype=torch.int32, device=self.device)

        for i in range(n_samples):
            if labels[i] != -1:
                continue

            neighbors = (distances[i] < self.eps).nonzero(as_tuple=True)[0]

            if neighbors.size(0) < self.min_samples:
                labels[i] = -1
                continue

            self._expand_cluster(X, labels, neighbors, 0, distances)

        return labels.cpu().numpy()

# Example usage
if __name__ == "__main__":
    # Sample data (replace with your actual data)
    import numpy as np
    data = np.random.rand(100, 2)

    dbscan = DBSCAN(eps=0.1, min_samples=5, device='cuda')
    dbscan.fit(data)
    labels = dbscan.predict(data)

    print("Labels:\n", labels)
