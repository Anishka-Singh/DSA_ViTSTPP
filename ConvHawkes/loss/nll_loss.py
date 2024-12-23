# loss/nll_loss.py
<<<<<<< HEAD
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
=======
# import torch
# from ..utils.kernels import compute_external_effect, compute_spatiotemporal_decay
# from ..utils.integrals import compute_Lambda_n

# def negative_log_likelihood_loss(events, T, S, mu, h_l, tau_l, beta, Sigma_k, Sigma_zeta):
#     """
#     Compute negative log-likelihood loss based on Equation (11)
    
#     Parameters:
#     - events: List of (t_n, s_n) tuples for event times and locations
#     - T: End time of observation window
#     - S: Spatial region bounds [(x_min, x_max), (y_min, y_max)]
#     - mu: Background rate
#     - h_l: Latent feature maps from CNN [L, H, W, 1]
#     - tau_l: Time parameters [L]
#     - beta: Temporal decay parameter
#     - Sigma_k: Spatial kernel covariance matrix [2,2]
#     - Sigma_zeta: Spatial decay covariance matrix [2,2]
    
#     Returns:
#     - nll: Negative log-likelihood value
#     """
#     device = h_l.device
    
#     # Convert events to tensors
#     times = torch.tensor([e[0] for e in events], device=device)
#     locations = torch.tensor([e[1] for e in events], device=device)
#     N = len(events)


#     # First term: sum of log intensities
#     log_intensities = torch.zeros(N, device=device)
    
#     for n in range(N):
#         t_n, s_n = times[n], locations[n]
        
#         # Calculate conditional intensity λ(t_n, s_n|H(t_n))
#         # Ensure `intensity` is on the same device as `alpha` and `gamma`
#         intensity = torch.tensor(mu, device=h_l.device)  # Move `mu` to the device of `h_l`

        
#         # Sum over previous events (j: t_j < t_n)
#         for j in range(n):
#             if times[j] < t_n:
#                 t_j, s_j = times[j], locations[j]
                
#                 # Calculate external effect α(t_n, s_n|I)
#                 alpha = compute_external_effect(t_n, s_n, h_l, tau_l, Sigma_k).to(device)
#                 alpha = alpha.squeeze()  # Ensure alpha is a scalar
            
#                 # Calculate spatio-temporal decay γ(t_n-t_j, s_n-s_j)
#                 gamma = compute_spatiotemporal_decay(
#                     t_n - t_j, 
#                     s_n - s_j, 
#                     beta, 
#                     Sigma_zeta
#                 ).to(device)
#                 gamma = gamma.squeeze()  # Ensure gamma is a scalar

#                 # print(f"alpha shape: {alpha.shape}, gamma shape: {gamma.shape}, intensity shape: {intensity.shape}")
#                 # print(f"alpha device: {alpha.device}, gamma device: {gamma.device}, intensity device: {intensity.device}")
#                 intensity += alpha * gamma
#                 intensity = intensity.to(device)
                

        
#         log_intensities[n] = torch.log(intensity)
    
#     # Second term: integral term
#     # μT|S| term
#     S_area = (S[0][1] - S[0][0]) * (S[1][1] - S[1][0])
#     integral_term = mu * T * S_area
    
#     # Add integral of α(t,s)γ(t-t_n, s-s_n)
#     Lambda_n = compute_Lambda_n(
#         times, locations, T, S, h_l, tau_l, beta, Sigma_k, Sigma_zeta
#     )
    
#     # Combine terms for final NLL
#     nll = -(torch.sum(log_intensities) - integral_term - Lambda_n)
    
#     return nll

import torch
from ..utils.kernels import compute_external_effect_vectorized, compute_spatiotemporal_decay_vectorized
from ..utils.integrals import compute_Lambda_n_vectorized

def negative_log_likelihood_loss(events, T, S, mu, h_l, tau_l, beta, Sigma_k, Sigma_zeta):
    """Vectorized implementation of negative log-likelihood loss with memory-efficient operations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert and normalize events
    times = torch.tensor([e[0] for e in events], device=device)
    locations = torch.tensor([[loc[0], loc[1]] for loc in [e[1] for e in events]], device=device)
    N = len(events)
    
    # Normalize times to [0,1] range
    times = (times - times.min()) / (times.max() - times.min() + 1e-10)

    # Print debug information
    print(f"Total events: {N}")
    print(f"GPU Memory before processing: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Move tensors to device
    h_l = h_l.to(device)
    tau_l = tau_l.to(device)
    Sigma_k = Sigma_k.to(device)
    Sigma_zeta = Sigma_zeta.to(device)
    
    # Normalize locations to [-1,1] range for each dimension
    for dim in range(2):
        loc_min = locations[:, dim].min()
        loc_max = locations[:, dim].max()
        locations[:, dim] = 2 * (locations[:, dim] - loc_min) / (loc_max - loc_min + 1e-10) - 1
    
    # Normalize parameters
    mu = torch.clamp(mu, min=1e-10)  # Ensure positive background rate
    beta = torch.abs(beta)  # Ensure positive decay
    
    # Normalize matrices
    Sigma_k = Sigma_k / (torch.norm(Sigma_k) + 1e-10)
    Sigma_zeta = Sigma_zeta / (torch.norm(Sigma_zeta) + 1e-10)

    # Process in smaller chunks
    chunk_size = 100  # Further reduced chunk size
    log_intensities = torch.zeros(N, device=device)
    
    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        print(f"Processing chunk {i}/{N}")
        
        # Clear cache at start of chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create time and spatial differences for current chunk only
        t_diffs_chunk = times[i:i_end].unsqueeze(1) - times.unsqueeze(0)  # [chunk_size, N]
        s_diffs_chunk = (
            locations[i:i_end].unsqueeze(1) - locations.unsqueeze(0)
        )  # [chunk_size, N, 2]
        
        # Compute external effects for chunk
        alphas_chunk = compute_external_effect_vectorized(
            times[i:i_end],
            locations[i:i_end],
            h_l, tau_l, Sigma_k
        )
        
        # Compute spatiotemporal decay for chunk
        gammas_chunk = compute_spatiotemporal_decay_vectorized(
            t_diffs_chunk,
            s_diffs_chunk,
            beta,
            Sigma_zeta
        )
        
        # Compute intensities for chunk
        intensities_chunk = mu + alphas_chunk + torch.sum(gammas_chunk, dim=1)
        log_intensities[i:i_end] = torch.log(torch.clamp(intensities_chunk, min=1e-10))
        
        # Clear intermediate results
        del t_diffs_chunk, s_diffs_chunk, alphas_chunk, gammas_chunk, intensities_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU Memory after chunk: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Compute integral terms
    log_intensities_term = torch.sum(log_intensities)
    S_area = (S[0][1] - S[0][0]) * (S[1][1] - S[1][0])
    integral_term = mu * T * S_area
    Lambda_n_term = compute_Lambda_n_vectorized(times, locations, T, S, h_l, tau_l, beta, Sigma_k, Sigma_zeta)
    
    # Combine terms with scaling and regularization
    nll = -(log_intensities_term - integral_term - Lambda_n_term) / N
    l2_reg = 1e-5 * (torch.norm(h_l) + torch.norm(tau_l) + torch.norm(Sigma_k) + torch.norm(Sigma_zeta))
    
    return nll + l2_reg
>>>>>>> f516a0a (Final clean commit)
