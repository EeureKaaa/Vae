import torch
import matplotlib.pyplot as plt
import numpy as np
from model import VAE
from utils import config, device, get_data_loaders

def load_model(model_path=None, latent_dim=None):
    """
    Load a trained VAE model
    """
    if model_path is None:
        model_path = config['model_save_path']
    if latent_dim is None:
        latent_dim = config['latent_dim']
    
    model = VAE(latent_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_images(model=None, num_images=10, save_path=None):
    """
    Generate images by sampling from the latent space
    """
    if model is None:
        model = load_model()
    if save_path is None:
        save_path = config['generated_images_path']
    
    with torch.no_grad():
        # Sample from latent space
        z = torch.randn(num_images, config['latent_dim']).to(device)
        sample = model.decode(z).cpu()
        
        # Display images
        fig, axes = plt.subplots(1, num_images, figsize=(12, 2))
        for i, ax in enumerate(axes):
            ax.imshow(sample[i].squeeze().numpy(), cmap='gray')
            ax.axis('off')
        
        plt.savefig(save_path)
        plt.close()
        print(f"Generated images saved as '{save_path}'")
        
        return sample

def reconstruct_images(model=None, save_path=None):
    """
    Reconstruct images from the test set
    """
    if model is None:
        model = load_model()
    if save_path is None:
        save_path = config['reconstructed_images_path']
    
    # Get test data loader
    _, test_loader = get_data_loaders()
    
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
        
        plt.savefig(save_path)
        plt.close()
        print(f"Reconstructed images saved as '{save_path}'")
        
        return data.cpu(), recon.cpu()

def interpolate_digits(model=None, start_idx=0, end_idx=1, steps=10):
    """
    Interpolate between two digits in the latent space
    """
    if model is None:
        model = load_model()
    
    # Get test data loader
    _, test_loader = get_data_loaders()
    
    with torch.no_grad():
        # Get test data
        data, _ = next(iter(test_loader))
        
        # Get latent representations
        start_img = data[start_idx:start_idx+1].to(device)
        end_img = data[end_idx:end_idx+1].to(device)
        
        # Encode images to get latent vectors
        start_mu, _ = model.encode(start_img)
        end_mu, _ = model.encode(end_img)
        
        # Interpolate in latent space
        interpolation = torch.zeros(steps, config['latent_dim']).to(device)
        for i in range(steps):
            alpha = i / (steps - 1)
            interpolation[i] = start_mu * (1 - alpha) + end_mu * alpha
        
        # Decode the interpolated points
        decoded = model.decode(interpolation)
        
        # Display interpolation
        plt.figure(figsize=(12, 2))
        for i in range(steps):
            plt.subplot(1, steps, i + 1)
            plt.imshow(decoded[i].cpu().squeeze().numpy(), cmap='gray')
            plt.axis('off')
        
        plt.savefig('interpolated_digits.png')
        plt.close()
        print("Interpolated digits saved as 'interpolated_digits.png'")
        
        return decoded.cpu()

if __name__ == "__main__":
    model = load_model()
    generate_images(model)
    reconstruct_images(model)
    interpolate_digits(model)
