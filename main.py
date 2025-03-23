import argparse
from train import train_model
from inference import load_model, generate_images, reconstruct_images, interpolate_digits
from utils import config

def main():
    parser = argparse.ArgumentParser(description='MNIST VAE')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate', 'reconstruct', 'interpolate', 'all'],
                        help='Mode to run (train, generate, reconstruct, interpolate, all)')
    parser.add_argument('--epochs', type=int, default=config['epochs'], help='Number of epochs for training')
    parser.add_argument('--latent-dim', type=int, default=config['latent_dim'], help='Dimension of latent space')
    parser.add_argument('--learning-rate', type=float, default=config['learning_rate'], help='Learning rate')
    parser.add_argument('--num-images', type=int, default=10, help='Number of images to generate')
    parser.add_argument('--model-path', type=str, default=config['model_save_path'], help='Path to model file')
    
    args = parser.parse_args()
    
    if args.mode == 'train' or args.mode == 'all':
        print("Training VAE model...")
        model = train_model(epochs=args.epochs, learning_rate=args.learning_rate, latent_dim=args.latent_dim)
    else:
        model = load_model(model_path=args.model_path, latent_dim=args.latent_dim)
    
    if args.mode == 'generate' or args.mode == 'all':
        print("Generating images from latent space...")
        generate_images(model, num_images=args.num_images)
    
    if args.mode == 'reconstruct' or args.mode == 'all':
        print("Reconstructing test images...")
        reconstruct_images(model)
    
    if args.mode == 'interpolate' or args.mode == 'all':
        print("Interpolating between digits...")
        interpolate_digits(model)

if __name__ == "__main__":
    main()
