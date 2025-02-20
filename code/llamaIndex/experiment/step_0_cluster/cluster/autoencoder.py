import torch
import torch.nn as nn
import torch.optim as optim

class AutoEncoder(nn.Module):
    # optional: encoding_dim=128
    def __init__(self, input_dim=4096, encoding_dim=2):
        super(AutoEncoder, self).__init__()
        
        # Encoder with more layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim)  # Final output is 2D
        )
        
        # Decoder with reversed structure
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),  # Reconstructed output dimension
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(model, data, epochs=50, lr=0.001, device='cuda'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    data = torch.tensor(data, device=device, dtype=torch.float32)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, data)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Return the encoded data after training
    with torch.no_grad():
        encoded_data = model.encoder(data).cpu().numpy()
    
    return encoded_data

# Example usage
if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)
    
    # Example data with shape (n, 4096)
    data = np.random.rand(100, 4096)

    autoencoder = AutoEncoder(input_dim=4096, encoding_dim=2)  # 2D encoding
    encoded_data = train_autoencoder(autoencoder, data, epochs=50, lr=0.001, device='cuda')
    print("Encoded data shape:", encoded_data.shape)