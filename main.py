import argparse
from train import train_model
from inference import load_model, generate_images, reconstruct_images, interpolate_digits, visualize_2d_latent_space, calculate_fid
from utils import config

def main():
    parser = argparse.ArgumentParser(description='MNIST VAE')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate', 'reconstruct', 'interpolate', 'latent-space-2d', 'fid', 'all'],
                        help='Mode to run (train, generate, reconstruct, interpolate, latent-space-2d, fid, all)')
    parser.add_argument('--epochs', type=int, default=config['epochs'], help='Number of epochs for training')
    parser.add_argument('--learning-rate', type=float, default=config['learning_rate'], help='Learning rate')
    parser.add_argument('--num-images', type=int, default=16, help='Number of images to generate')
    parser.add_argument('--fid-images', type=int, default=500, help='Number of images to use for FID calculation')
    # parser.add_argument('--model-path', type=str, default=config['model_save_path'], help='Path to model file')
    # parser.add_argument('--latent-dim', type=int, default=config['latent_dim'], help='Dimension of latent space')
    # parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases for tracking training')
    
    args = parser.parse_args()
    
    if args.mode == 'train' or args.mode == 'all':
        print("Training VAE model...")
        model = train_model(
            epochs=args.epochs, 
            learning_rate=args.learning_rate, 
            latent_dim=config['latent_dim'],
            use_wandb=True
        )
    else:
        model = load_model()
    
    if args.mode == 'generate' or args.mode == 'all':
        print("Generating images from latent space...")
        generate_images(model, num_images=args.num_images)
    
    if args.mode == 'reconstruct' or args.mode == 'all':
        print("Reconstructing test images...")
        reconstruct_images(model)
    
    if args.mode == 'interpolate' or args.mode == 'all':
        print("Interpolating between digits...")
        interpolate_digits(model)
    
    # Calculate FID score
    if args.mode == 'fid' or args.mode == 'all':
        print("Calculating FID score...")
        calculate_fid(model, args)
    
    # Only run these if latent_dim is 2
    if config['latent_dim'] == 2 and (args.mode == 'latent-space-2d' or args.mode == 'all'):
        print("Visualizing 2D latent space...")
        visualize_2d_latent_space(model)


if __name__ == "__main__":
    main()
