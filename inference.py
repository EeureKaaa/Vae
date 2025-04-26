import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from model import VAE
from utils import config, device, get_data_loaders, ensure_directory_exists,save_image
from torch_fidelity import calculate_metrics
import logging
import time
import json

def load_model(model_path=None) -> VAE:
    """
    Load a trained VAE model
    """
    if model_path is None:
        model_path = config['model_save_path']
    logging.info(f"Loading model from {model_path}")
    
    model = VAE(config['latent_dim']).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_images(model=None, num_images=10, save_path=None) -> None:
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

def reconstruct_images(model=None, save_path=None) -> None:
    """
    Reconstruct images from the test set
    """
    if model is None:
        model = load_model()
    if save_path is None:
        save_path = config['reconstructed_images_path']
    
    # Get test data loader
    _, test_loader = get_data_loaders(batch_size=1000)  # Larger batch to ensure we get all digits
    
    with torch.no_grad():
        # Get a batch of test data
        data, labels = next(iter(test_loader))
        
        # Create a list to store one example of each digit
        selected_digits = []
        selected_data_list = []
        
        # Find one example of each digit (0-9)
        for digit in range(10):
            indices = (labels == digit).nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                print(f"No examples of digit {digit} found in the batch")
                continue
            
            # Take the first example of this digit
            idx = indices[0].item()
            selected_digits.append(digit)
            selected_data_list.append(data[idx].unsqueeze(0))  # Add batch dimension back
        
        # Stack the selected examples
        selected_data = torch.cat(selected_data_list, dim=0).to(device)
        
        # Reconstruct
        recon, _, _ = model(selected_data)
        
        # Display original and reconstructed images in 2 rows
        n = len(selected_digits)
        plt.figure(figsize=(12, 4))
        
        for i in range(n):
            # Original (top row)
            plt.subplot(2, n, i + 1)
            plt.imshow(selected_data[i].cpu().squeeze().numpy(), cmap='gray')
            plt.title(f"{selected_digits[i]}")
            plt.axis('off')
            
            # Reconstruction (bottom row)
            plt.subplot(2, n, i + 1 + n)
            plt.imshow(recon[i].cpu().squeeze().numpy(), cmap='gray')
            plt.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Make room for the suptitle
        plt.savefig(save_path)
        plt.close()
        print(f"Reconstructed all digits saved as '{save_path}'")
        return True

def interpolate_digits(model=None, start_idx=0, end_idx=1, steps=10, save_path=None):
    """
    Interpolate between two digits in the latent space
    """
    if model is None:
        model = load_model()
    if save_path is None:
        save_path = config['interpolation_images_path']
    
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
        
        plt.savefig(save_path if save_path else 'interpolated_digits.png', bbox_inches='tight')
        plt.close()
        print("Interpolated digits saved as 'interpolated_digits.png'")
        
        return decoded.cpu()

def visualize_2d_latent_space(model=None, n_points=20, save_path=None) -> None:
    """
    Visualize the 2D latent space by creating a grid of generated images.
    This only works when latent_dim is set to 2.
    
    Args:
        model: The VAE model (will be loaded if None)
        n_points: Number of points in each dimension of the grid
        save_path: Path to save the visualization
    """
    if config['latent_dim'] != 2:
        print("This function only works with 2D latent space (latent_dim=2)")
        return
    if save_path is None:
        save_path = config['latent_space_2d_path']
    
    if model is None:
        model = load_model()
    
    model.eval()
    
    # Create a grid of latent points
    x = np.linspace(-3, 3, n_points)
    y = np.linspace(-3, 3, n_points)
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Generate images for each point in the grid
    with torch.no_grad():
        for i, yi in enumerate(y):
            for j, xi in enumerate(x):
                # Create latent vector
                z = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
                
                # Generate image
                img = model.decode(z).cpu()
                
                # Add subplot
                plt.subplot(n_points, n_points, i * n_points + j + 1)
                plt.imshow(img.squeeze(0).squeeze(0), cmap='gray')
                plt.axis('off')
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(save_path if save_path else 'latent_space_2d.png', bbox_inches='tight')
    plt.close()
    print(f"2D latent space visualization saved as '{save_path}'")
def calculate_fid(model=None, args=None, real_images_dir=None, generated_images_dir=None, force_cpu=False) -> float:
    """
    Calculate the Fréchet Inception Distance (FID) between generated images and real images.
    
    Args:
        model: The VAE model (will be loaded if None)
        num_images: Number of images to generate for FID calculation
        real_images_dir: Directory to save real images (will use config if None)
        generated_images_dir: Directory to save generated images (will use config if None)
        
    Returns:
        FID score (lower is better)
    """
    if model is None:
        model = load_model()
    if real_images_dir is None:
        real_images_dir = config['fid_real_images_dir']
    if generated_images_dir is None:
        generated_images_dir = config['fid_generated_images_dir']
    
    # Create directories if they don't exist
    ensure_directory_exists(real_images_dir)
    ensure_directory_exists(generated_images_dir)
    
    # Get test data loader for real images
    num_images = args.fid_images
    _, test_loader = get_data_loaders(batch_size=num_images)
    
    # Save real images
    print(f"Saving {num_images} real images to {real_images_dir}...")
    real_data, _ = next(iter(test_loader))
    # Limit to num_images
    real_data = real_data[:num_images]
    
    # Save each real image separately
    for i in range(len(real_data)):
        save_image(real_data[i], os.path.join(real_images_dir, f"real_{i}.png"))
    
    # Generate and save fake images
    print(f"Generating and saving {num_images} fake images to {generated_images_dir}...")
    with torch.no_grad():
        # Sample from latent space
        z = torch.randn(num_images, config['latent_dim']).to(device)
        fake_data = model.decode(z).cpu()
        
        # Save each generated image separately
        for i in range(len(fake_data)):
            save_image(fake_data[i], os.path.join(generated_images_dir, f"fake_{i}.png"))
    
    # Calculate FID
    use_cuda = torch.cuda.is_available() and not force_cpu
    if use_cuda:
        print("Calculating FID score using GPU...")
    else:
        print("Calculating FID score using CPU...")
        
    metrics = calculate_metrics(
        input1=real_images_dir,
        input2=generated_images_dir,
        cuda=use_cuda,
        fid=True,
        verbose=True
    )
    
    # The key is 'frechet_inception_distance' not 'fid'
    fid_score = metrics['frechet_inception_distance']
    print(f"FID Score: {fid_score:.4f}")

    # store json for fid score
    if fid_score:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        fid_score = {
            'model_path': args.model_path,
            'num_images': num_images,
            'fid_score': fid_score,
            'timestamp': timestamp
        }
        with open(config['fid_score_path'] + f'/fid_score_{timestamp}.json', 'w') as f:
            json.dump(fid_score, f, indent=4)
    
    return fid_score

if __name__ == "__main__":
    model = load_model()
    generate_images(model)
    reconstruct_images(model)
    interpolate_digits(model)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=config['model_save_path'], help='Path to model file')
    parser.add_argument('--fid-images', type=int, default=1000, help='Number of images to use for FID calculation')
    args = parser.parse_args()
    
    # Calculate FID score
    calculate_fid(model, args)
    
    # Only run these if latent_dim is 2
    if config['latent_dim'] == 2:
        visualize_2d_latent_space(model)
