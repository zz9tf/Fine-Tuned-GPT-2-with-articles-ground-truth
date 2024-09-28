import torch

class PCA:
    def __init__(self, n_components, device='cuda'):
        self.n_components = n_components
        self.device = device

    def fit_transform(self, X):
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        # Center the data
        mean = X.mean(dim=0)
        X_centered = X - mean
        
        # Compute covariance matrix
        cov_matrix = torch.matmul(X_centered.T, X_centered) / (X_centered.size(0) - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, sorted_indices]
        eigenvalues = eigenvalues[sorted_indices]
        
        # Select top components
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # Project data
        X_pca = torch.matmul(X_centered, self.components_)
        return X_pca.cpu().numpy()

# Example usage
if __name__ == "__main__":
    import numpy as np
    data = np.random.rand(100, 5)

    pca = PCA(n_components=2, device='cuda')
    transformed_data = pca.fit_transform(data)
    print("Transformed data:\n", transformed_data)
