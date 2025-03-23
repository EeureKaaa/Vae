# MNIST VAE Generator

This project implements a Variational Autoencoder (VAE) to generate MNIST handwritten digits.

## Overview

A Variational Autoencoder (VAE) is a type of generative model that learns to encode data into a latent space and then decode it back. The key features of this implementation:

- Convolutional neural network architecture for both encoder and decoder
- Latent space dimension of 20
- Training on the MNIST dataset
- Generation of new digit images from random latent vectors
- Reconstruction of existing images

## Requirements

- PyTorch
- torchvision
- matplotlib
- numpy

## Usage

Simply run the script:

```bash
python mnist_vae.py
```

The script will:
1. Download the MNIST dataset (if not already present)
2. Train the VAE model for 20 epochs
3. Save the best model as `mnist_vae_model.pt`
4. Generate 10 random digit images and save them as `generated_mnist.png`
5. Reconstruct 10 test images and save the comparison as `reconstructed_mnist.png`

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

## Results

After training, you can examine:
- `generated_mnist.png`: New digits generated from random points in the latent space
- `reconstructed_mnist.png`: Original test images (top row) and their reconstructions (bottom row)
