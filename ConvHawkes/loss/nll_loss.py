# loss/nll_loss.py
import torch
from ..utils.kernels import compute_external_effect, compute_spatiotemporal_decay
from ..utils.integrals import compute_Lambda_n

def negative_log_likelihood_loss(events, T, S, mu, h_l, tau_l, beta, Sigma_k, Sigma_zeta):
    """
    Compute negative log-likelihood loss based on Equation (11)
    
    Parameters:
    - events: List of (t_n, s_n) tuples for event times and locations
    - T: End time of observation window
    - S: Spatial region bounds [(x_min, x_max), (y_min, y_max)]
    - mu: Background rate
    - h_l: Latent feature maps from CNN [L, H, W, 1]
    - tau_l: Time parameters [L]
    - beta: Temporal decay parameter
    - Sigma_k: Spatial kernel covariance matrix [2,2]
    - Sigma_zeta: Spatial decay covariance matrix [2,2]
    
    Returns:
    - nll: Negative log-likelihood value
    """
    device = h_l.device
    
    # Convert events to tensors
    times = torch.tensor([e[0] for e in events], device=device)
    locations = torch.tensor([e[1] for e in events], device=device)
    N = len(events)
    
    # First term: sum of log intensities
    log_intensities = torch.zeros(N, device=device)
    
    for n in range(N):
        t_n, s_n = times[n], locations[n]
        
        # Calculate conditional intensity λ(t_n, s_n|H(t_n))
        intensity = mu
        
        # Sum over previous events (j: t_j < t_n)
        for j in range(n):
            if times[j] < t_n:
                t_j, s_j = times[j], locations[j]
                
                # Calculate external effect α(t_n, s_n|I)
                alpha = compute_external_effect(t_n, s_n, h_l, tau_l, Sigma_k)
                
                # Calculate spatio-temporal decay γ(t_n-t_j, s_n-s_j)
                gamma = compute_spatiotemporal_decay(
                    t_n - t_j, 
                    s_n - s_j, 
                    beta, 
                    Sigma_zeta
                )
                
                intensity += alpha * gamma
        
        log_intensities[n] = torch.log(intensity)
    
    # Second term: integral term
    # μT|S| term
    S_area = (S[0][1] - S[0][0]) * (S[1][1] - S[1][0])
    integral_term = mu * T * S_area
    
    # Add integral of α(t,s)γ(t-t_n, s-s_n)
    Lambda_n = compute_Lambda_n(
        times, locations, T, S, h_l, tau_l, beta, Sigma_k, Sigma_zeta
    )
    
    # Combine terms for final NLL
    nll = -(torch.sum(log_intensities) - integral_term - Lambda_n)
    
    return nll