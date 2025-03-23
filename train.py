import torch
import torch.nn.functional as F
import torch.optim as optim
from model import VAE, loss_function
from utils import config, device, get_data_loaders, is_wandb_ready, WANDB_AVAILABLE

if WANDB_AVAILABLE:
    import wandb

# Initialize wandb only when needed
def init_wandb():
    """Initialize wandb for training"""
    return wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="eureka_experiment",
        # Set the wandb project where this run will be logged.
        project=f"Vae_{config['latent_dim']}_dim_{config['epochs']}_epochs",
        # Track hyperparameters and run metadata.
        name=f"Vae_{config['latent_dim']}_dim_{config['epochs']}_epochs",
        config={
            "learning_rate": config['learning_rate'],
            "architecture": "CNN",
            "dataset": config['dataset'],
            "epochs": config['epochs'],
            "latent_dim": config['latent_dim'],
            "batch_size": config['batch_size'],
        },
    )

def train_epoch(model, optimizer, train_loader, epoch, use_wandb=True):
    """
    Train the model for one epoch
    """
    model.train()
    train_loss = 0
    bce = 0
    kld = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        BCE, KLD, loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        bce += BCE.item()
        kld += KLD.item()
        
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    
    # Log epoch metrics to wandb if available
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "epoch": epoch,
            "epoch_loss": avg_loss,
            "epoch_bce": bce / len(train_loader.dataset),
            "epoch_kld": kld / len(train_loader.dataset)
        })
    
    return avg_loss

def test_epoch(model, test_loader, epoch, use_wandb=True):
    """
    Test the model on the test dataset
    """
    model.eval()
    test_loss = 0
    bce = 0
    kld = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, log_var = model(data)
            BCE, KLD, loss = loss_function(recon, data, mu, log_var)
            test_loss += loss.item()
            bce += BCE.item()
            kld += KLD.item()
    
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    
    # Log epoch metrics to wandb if available
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "epoch": epoch,
            "test_loss": test_loss,
            "test_bce": bce / len(test_loader.dataset),
            "test_kld": kld / len(test_loader.dataset)
        })
    
    return test_loss

def train_model(epochs=None, learning_rate=None, latent_dim=None, use_wandb=True):
    """
    Train the VAE model
    """
    # Initialize wandb if requested and available
    run = None
    if use_wandb and is_wandb_ready():
        run = init_wandb()
    elif use_wandb and not is_wandb_ready():
        print("Warning: Wandb is not available or not authenticated. Training will continue without wandb logging.")
        use_wandb = False
    
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
    
    # Log model architecture to wandb if available
    if use_wandb and WANDB_AVAILABLE:
        wandb.watch(model, log="all", log_freq=100)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        avg_train_loss = train_epoch(model, optimizer, train_loader, epoch, use_wandb)
        test_loss = test_epoch(model, test_loader, epoch, use_wandb)
        
        # Save model if it has the best loss so far
        if test_loss < best_loss:
            best_loss = test_loss
        torch.save(model.state_dict(), config['model_save_path'])
        print(f"Model saved with loss: {test_loss:.4f}")
    
    # Close wandb run if it was initialized
    if use_wandb and run is not None:
        run.finish()
    
    print("Training complete!")
    return model

if __name__ == "__main__":
    train_model()
