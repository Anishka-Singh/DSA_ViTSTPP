# models/convhawkes.py
import torch
import torch.nn as nn
from .cnn import SimpleCNN
from utils.kernels import compute_external_effect, compute_spatiotemporal_decay
from loss.nll_loss import negative_log_likelihood_loss

class ConvHawkes(nn.Module):
    def __init__(self, N_l, beta, Sigma_k, Sigma_zeta, mu):
        """
        ConvHawkes model implementation.
        
        Parameters:
        - N_l: Number of CNN layers
        - beta: Temporal decay parameter
        - Sigma_k: Spatial kernel covariance matrix
        - Sigma_zeta: Spatial decay covariance matrix
        - mu: Background rate
        """
        super().__init__()
        self.cnn = SimpleCNN(N_l)
        self.beta = beta
        self.Sigma_k = Sigma_k
        self.Sigma_zeta = Sigma_zeta
        self.mu = mu
        
        # Initialize time parameters (tau_l)
        self.tau_l = nn.Parameter(torch.randn(N_l))

    def forward(self, image_sequence, events, T, S):
        """
        Forward pass of the ConvHawkes model.
        
        Parameters:
        - image_sequence: Input image sequence [B, T, C, H, W]
        - events: List of (t, s) tuples for event times and locations
        - T: End time of observation window
        - S: Spatial region bounds
        
        Returns:
        - loss: Negative log-likelihood loss
        """
        # Get latent features from CNN
        h_l = self.cnn(image_sequence)
        
        # Compute negative log-likelihood loss
        loss = negative_log_likelihood_loss(
            events=events,
            T=T,
            S=S,
            mu=self.mu,
            h_l=h_l,
            tau_l=self.tau_l,
            beta=self.beta,
            Sigma_k=self.Sigma_k,
            Sigma_zeta=self.Sigma_zeta
        )
        
        return loss