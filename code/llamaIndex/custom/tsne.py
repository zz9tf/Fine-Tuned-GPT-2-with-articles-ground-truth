import torch
import numpy as np
from scipy.special import expit  # Sigmoid function

class TSNE:
    def __init__(self, n_components=2, n_iter=1000, learning_rate=200.0, device='cuda'):
        self.n_components = n_components
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.device = device

    def fit_transform(self, X):
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        n_samples = X.size(0)
        
        # Compute pairwise affinities
        distances = torch.cdist(X, X)
        pairwise_affinities = torch.exp(-distances**2 / 2)
        pairwise_affinities /= pairwise_affinities.sum(dim=1, keepdim=True)
        
        # Initialize low-dimensional embeddings
        Y = torch.randn(n_samples, self.n_components, device=self.device, dtype=torch.float32)
        
        # Optimization
        for _ in range(self.n_iter):
            # Compute pairwise affinities in low-dimensional space
            distances_Y = torch.cdist(Y, Y)
            print(f"Y share {Y.shape}")
            print(f"distances_Y {distances_Y.shape}")
            pairwise_affinities_Y = 1 / (1 + distances_Y**2)
            pairwise_affinities_Y /= pairwise_affinities_Y.sum(dim=1, keepdim=True)
            
            print(f"pairwise_affinities shape {pairwise_affinities}")
            print(f"pairwise_affinities_Y shape {pairwise_affinities_Y}")
            # Compute gradient
            P = pairwise_affinities - pairwise_affinities_Y
            print(f'P share {P.shape}')
            gradient = torch.matmul(P, (Y - Y.mean(dim=0))) / (1 + distances_Y**2)
            Y -= self.learning_rate * gradient
            
            # Optionally, add momentum or other enhancements here

        return Y.cpu().numpy()

# Example usage
if __name__ == "__main__":
    import numpy as np
    data = np.random.rand(100, 5)

    tsne = TSNE(n_components=2, n_iter=1000, learning_rate=200.0, device='cuda')
    transformed_data = tsne.fit_transform(data)
    print("Transformed data:\n", transformed_data)
