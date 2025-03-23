import torch
import torch.optim as optim
from model import VAE, loss_function
from utils import config, device, get_data_loaders

def train_epoch(model, optimizer, train_loader, epoch):
    """
    Train the model for one epoch
    """
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
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

def test_epoch(model, test_loader, epoch):
    """
    Test the model on the test dataset
    """
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

def train_model(epochs=None, learning_rate=None, latent_dim=None):
    """
    Train the VAE model
    """
    # Use config values if parameters not provided
    if epochs is None:
        epochs = config['epochs']
    if learning_rate is None:
        learning_rate = config['learning_rate']
    if latent_dim is None:
        latent_dim = config['latent_dim']
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders()
    
    # Initialize model and optimizer
    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        train_epoch(model, optimizer, train_loader, epoch)
        loss = test_epoch(model, test_loader, epoch)
        
        # Save model if it has the best loss so far
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"Model saved with loss: {loss:.4f}")
    
    print("Training complete!")
    return model

if __name__ == "__main__":
    train_model()
