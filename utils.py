import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

# Set random seed for reproducibility
torch.manual_seed(42)

# Create directories if they don't exist
def ensure_directory_exists(file_path):
    """Create the directory for a file path if it doesn't exist"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Configuration parameters
config = {
    'dataset': 'MNIST',
    'batch_size': 16,
    'epochs': 20,
    'learning_rate': 1e-3,
    'latent_dim': 20,  # Size of latent space
}
config_update = {
    'model_save_path': f'./checkpoints/{config["dataset"]}_vae_model_{config["latent_dim"]}_dim.pt',
    'generated_images_path': f'./outputs/{config["dataset"]}_dim{config["latent_dim"]}/generated.png',
    'reconstructed_images_path': f'./outputs/{config["dataset"]}_dim{config["latent_dim"]}/reconstructed.png',
    'interpolation_images_path': f'./outputs/{config["dataset"]}_dim{config["latent_dim"]}/interpolation.png',
    'latent_space_2d_path': f'./outputs/{config["dataset"]}_dim{config["latent_dim"]}/latent_space_2d.png'
}
config.update(config_update)

# Create necessary directories
for path_key in ['model_save_path', 'generated_images_path', 'reconstructed_images_path', 'interpolation_images_path', 'latent_space_2d_path']:
    ensure_directory_exists(config[path_key])

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders(batch_size=None):
    """
    Create and return train and test data loaders for MNIST dataset
    """
    if batch_size is None:
        batch_size = config['batch_size']
        
    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST(root=f'./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = datasets.MNIST(root=f'./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
