# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.convhawkes import ConvHawkes

class EventDataset(torch.utils.data.Dataset):
    def __init__(self, images, events):
        """
        Custom dataset for event data
        
        Parameters:
        - images: Image sequences [N, T, C, H, W]
        - events: List of (t, s) tuples
        """
        self.images = images
        self.events = events
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.events[idx]

def train_convhawkes(
    train_images, 
    train_events,
    val_images=None,
    val_events=None,
<<<<<<< HEAD
    num_epochs=100,
=======
    num_epochs=3,
>>>>>>> f516a0a (Final clean commit)
    batch_size=16,
    learning_rate=1e-3,
    N_l=3,
    beta=1.0,
    mu=0.1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Training function for ConvHawkes model
    
    Parameters:
    - train_images: Training image sequences [N, T, C, H, W]
    - train_events: Training event sequences (list of (t, s) tuples)
    - val_images: Validation image sequences
    - val_events: Validation event sequences
    - num_epochs: Number of training epochs
    - batch_size: Batch size
    - learning_rate: Learning rate
    - N_l: Number of CNN layers
    - beta: Temporal decay parameter
    - mu: Background rate
    """
    # Initialize spatial covariance matrices
    Sigma_k = torch.eye(2, device=device)
    Sigma_zeta = torch.eye(2, device=device)
    
    # Create dataset and dataloader
    train_dataset = EventDataset(train_images, train_events)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_images is not None and val_events is not None:
        val_dataset = EventDataset(val_images, val_events)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = ConvHawkes(
        N_l=N_l,
        beta=beta,
        Sigma_k=Sigma_k,
        Sigma_zeta=Sigma_zeta,
        mu=mu
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, events) in enumerate(train_loader):
            images = images.to(device)
            
            # Forward pass
            loss = model(
                image_sequence=images,
                events=events,
                T=torch.max(torch.tensor([e[0] for e in events])),
                S=[(0, 1), (0, 1)]  # Normalized spatial bounds
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch}, Average Training Loss: {avg_train_loss:.4f}')
        
        # Validation
        if val_images is not None and val_events is not None:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, events in val_loader:
                    images = images.to(device)
                    loss = model(
                        image_sequence=images,
                        events=events,
                        T=torch.max(torch.tensor([e[0] for e in events])),
                        S=[(0, 1), (0, 1)]
                    )
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f'Epoch {epoch}, Validation Loss: {avg_val_loss:.4f}')

if __name__ == "__main__":
    # Example usage
    
    train_images = torch.randn(1000, 10, 3, 128, 128)  # 1000 sequences, 10 timesteps each
    train_events = [[(t, (x, y)) for t, x, y in zip(
        torch.sort(torch.rand(5))[0],
        torch.rand(5),
        torch.rand(5)
    )] for _ in range(1000)]
    
    train_convhawkes(
        train_images=train_images,
        train_events=train_events,
<<<<<<< HEAD
        num_epochs=100,
=======
        num_epochs=3,
>>>>>>> f516a0a (Final clean commit)
        batch_size=16
    )