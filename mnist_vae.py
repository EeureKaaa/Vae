import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Parameters
batch_size = 128
epochs = 20
learning_rate = 1e-3
latent_dim = 20  # Size of latent space
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size
        self.flatten_size = 64 * 7 * 7
        
        # Mean and variance for latent space
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        z = self.decoder_input(z)
        z = self.decoder(z)
        return z
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

# Loss function
def loss_function(recon_x, x, mu, log_var):
    # Reconstruction loss (binary cross entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + KLD

# Initialize model
model = VAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(epoch):
    model.train()
    train_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# Testing function
def test(epoch):
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, log_var = model(data)
            test_loss += loss_function(recon, data, mu, log_var).item()
    
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    return test_loss

# Generate images
def generate_images(num_images=10):
    model.eval()
    with torch.no_grad():
        # Sample from latent space
        z = torch.randn(num_images, latent_dim).to(device)
        sample = model.decode(z).cpu()
        
        # Display images
        fig, axes = plt.subplots(1, num_images, figsize=(12, 2))
        for i, ax in enumerate(axes):
            ax.imshow(sample[i].squeeze().numpy(), cmap='gray')
            ax.axis('off')
        
        plt.savefig('generated_mnist.png')
        plt.close()
        print("Generated images saved as 'generated_mnist.png'")

# Reconstruct images
def reconstruct_images():
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        data, _ = next(iter(test_loader))
        data = data[:10].to(device)
        
        # Reconstruct
        recon, _, _ = model(data)
        
        # Display original and reconstructed images
        n = 10
        plt.figure(figsize=(12, 4))
        for i in range(n):
            # Original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(data[i].cpu().squeeze().numpy(), cmap='gray')
            plt.axis('off')
            
            # Reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(recon[i].cpu().squeeze().numpy(), cmap='gray')
            plt.axis('off')
        
        plt.savefig('reconstructed_mnist.png')
        plt.close()
        print("Reconstructed images saved as 'reconstructed_mnist.png'")

# Main training loop
if __name__ == "__main__":
    best_loss = float('inf')
    
    # Train the model
    for epoch in range(1, epochs + 1):
        train(epoch)
        loss = test(epoch)
        
        # Save model if it has the best loss so far
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 'mnist_vae_model.pt')
            print(f"Model saved with loss: {loss:.4f}")
    
    # Load the best model
    model.load_state_dict(torch.load('mnist_vae_model.pt'))
    
    # Generate and reconstruct images
    generate_images(10)
    reconstruct_images()
    
    print("Training complete!")
