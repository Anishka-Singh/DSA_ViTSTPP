# models/convhawkes.py
import torch
import torch.nn as nn
from .cnn import SimpleCNN
<<<<<<< HEAD
from utils.kernels import compute_external_effect, compute_spatiotemporal_decay
from loss.nll_loss import negative_log_likelihood_loss
=======
from ConvHawkes.utils.kernels import compute_external_effect_vectorized, compute_spatiotemporal_decay_vectorized
from ConvHawkes.loss.nll_loss import negative_log_likelihood_loss
>>>>>>> f516a0a (Final clean commit)

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
<<<<<<< HEAD
        self.beta = beta
        self.Sigma_k = Sigma_k
        self.Sigma_zeta = Sigma_zeta
        self.mu = mu
=======
        self.mu = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.Sigma_k = nn.Parameter(torch.eye(2) * 0.1)
        self.Sigma_zeta = nn.Parameter(torch.eye(2) * 0.1)
>>>>>>> f516a0a (Final clean commit)
        
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