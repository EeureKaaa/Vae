import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set random seed for reproducibility
torch.manual_seed(42)

# Configuration parameters
config = {
    'batch_size': 128,
    'epochs': 20,
    'learning_rate': 1e-3,
    'latent_dim': 20,  # Size of latent space
    'model_save_path': 'mnist_vae_model.pt',
    'generated_images_path': 'generated_mnist.png',
    'reconstructed_images_path': 'reconstructed_mnist.png'
}

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
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
