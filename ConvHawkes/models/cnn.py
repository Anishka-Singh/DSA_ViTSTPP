# models/cnn.py
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, N_l=3):
        """
        A simple CNN that transforms input images into latent feature maps.
        
        Parameters:
        - N_l: Number of CNN layers
        """
        super(SimpleCNN, self).__init__()
        
        # Define the CNN layers
        self.cnn = nn.Sequential(
            # Layer 1
<<<<<<< HEAD
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
=======
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
>>>>>>> f516a0a (Final clean commit)
            nn.ReLU(),
            
            # Layer 2
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Layer 3
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Forward pass for the CNN.
        
        Parameters:
        - x: Input image sequence [B, T, C, H, W]
        
        Returns:
        - latent_features: Latent feature maps [B, T, 1, H, W]
        """
        B, T, C, H, W = x.size()
        
        # Reshape to process each image
        x = x.view(B * T, C, H, W)
        
        # Pass through CNN
        latent_features = self.cnn(x)
        
        # Reshape back to sequence format
        latent_features = latent_features.view(B, T, 1, H, W)
        
        return latent_features