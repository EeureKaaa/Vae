# MNIST VAE Generator

This project implements a Variational Autoencoder (VAE) to generate MNIST handwritten digits.

## Overview

A Variational Autoencoder (VAE) is a type of generative model that learns to encode data into a latent space and then decode it back.

## Requirements

- PyTorch
- torchvision
- matplotlib
- numpy

## Usage

Simply run the script:

```bash
python main.py --mode all
```

Use `--help` to get information about command-line options.
```bash
python main.py --help
```

The script will:
1. Download the MNIST dataset (if not already present)
2. Train the VAE model for 20 epochs
3. Save the best model in `checkpoints` directory
4. Inference with 4 modes:
    - generate_images: Generate 10 random digit images
    - reconstruct_images: Reconstruct 10 test digits
    - interpolate: Interpolate between two random points in the latent space
    - visualize_2d_latent_space: Visualize the 2D latent space


## Model Architecture

### Encoder
- Convolutional layers to extract features
- Two fully connected layers to produce mean (μ) and log-variance (log σ²) of the latent distribution

### Decoder
- Fully connected layer to map from latent space to convolutional features
- Transposed convolutional layers to reconstruct the image
- Sigmoid activation for final output (pixel values between 0 and 1)

## Loss Function

The loss function combines:
- Reconstruction loss (binary cross-entropy)
- KL divergence to ensure the latent space follows a standard normal distribution


