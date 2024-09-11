import torch
import numpy as np

class TSNE:
    def __init__(self, n_components=2, n_iter=1000, learning_rate=200.0, device='cuda'):
        self.n_components = n_components
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.device = device

    def fit_transform(self, X):
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        n_samples = X.size(0)
        
        # Compute pairwise distances in high-dimensional space
        distances = torch.cdist(X, X)
        
        # Compute pairwise affinities in high-dimensional space (P matrix)
        sigmas = torch.std(X, dim=1, keepdim=True)
        pairwise_affinities = torch.exp(-distances**2 / (2 * sigmas**2))  # Gaussian kernel
        pairwise_affinities /= pairwise_affinities.sum()  # Normalize to sum to 1
        
        # Symmetrize the affinities (P matrix should be symmetric in t-SNE)
        pairwise_affinities = (pairwise_affinities + pairwise_affinities.T) / (2 * n_samples)

        # Initialize low-dimensional embeddings (Y) randomly
        Y = torch.randn(n_samples, self.n_components, device=self.device, dtype=torch.float32)

        # Optimization loop
        for i in range(self.n_iter):
            # Compute pairwise distances in low-dimensional space (Q matrix)
            distances_Y = torch.cdist(Y, Y)
            pairwise_affinities_Y = 1 / (1 + distances_Y**2)  # Student-t distribution
            pairwise_affinities_Y /= pairwise_affinities_Y.sum()  # Normalize to sum to 1
            
            # Compute the gradient (based on the attractive/repulsive forces)
            P_minus_Q = pairwise_affinities - pairwise_affinities_Y
            weight = P_minus_Q / (1 + distances_Y**2)
            gradient = torch.matmul(weight, (Y - Y.mean(dim=0)))

            # Update Y with gradient descent
            Y -= self.learning_rate * gradient

            # Optional: Add momentum or learning rate decay for stabilization
            if (i + 1) % 100 == 0:
                print(f'Iteration {i+1}/{self.n_iter}, Loss: {torch.norm(P_minus_Q).item():.4f}')

        return Y.cpu().numpy()

# Example usage
if __name__ == "__main__":
    data = np.random.rand(100, 5)  # Example data: 100 samples, 5 dimensions

    tsne = TSNE(n_components=2, n_iter=1000, learning_rate=200.0, device='cuda')
    transformed_data = tsne.fit_transform(data)
    print("Transformed data:\n", transformed_data)