# utils/kernels.py
import torch
<<<<<<< HEAD
=======
import torch.nn.functional as F
>>>>>>> f516a0a (Final clean commit)

def temporal_kernel(t, tau, delta):
    """
    Uniform temporal kernel (Equation 7)
    h(t - τ) = 𝟙[τ - Δ < t < τ + Δ]
    """
    return torch.where((tau - delta < t) & (t < tau + delta),
                      torch.ones_like(t),
                      torch.zeros_like(t))

def spatial_kernel(s, x, Sigma_k):
    """
    Gaussian spatial kernel (Equation 8)
    k(s - x) = exp(-(s - x)ᵀΣₖ⁻¹(s - x))
    """
<<<<<<< HEAD
    diff = s - x
    return torch.exp(-torch.matmul(torch.matmul(diff.unsqueeze(1), 
                                               torch.inverse(Sigma_k)), 
                                 diff.unsqueeze(-1)).squeeze())
=======
    device = Sigma_k.device
    s = s.to(device)
    x = x.to(device)
    
    # Ensure s and x are 2D vectors
    if len(s.shape) == 1:
        s = s.view(-1)
    if len(x.shape) == 1:
        x = x.view(-1)
        
    if s.shape[0] != 2 or x.shape[0] != 2:
        raise ValueError(f"Expected 2D vectors, got shapes {s.shape} and {x.shape}")
    
    # Compute difference vector
    diff = (s - x).view(2, 1)  # Shape: [2, 1]
    
    # Compute inverse of Sigma_k
    Sigma_k_inv = torch.inverse(Sigma_k)  # Shape: [2, 2]
    
    # Compute quadratic form
    quad_form = torch.matmul(torch.matmul(diff.T, Sigma_k_inv), diff)  # Shape: [1, 1]
    
    return torch.exp(-quad_form.squeeze())  # Return scalar
>>>>>>> f516a0a (Final clean commit)

def temporal_decay(t_diff, beta):
    """
    Exponential temporal decay (Equation 9)
    κ(t - tⱼ) = exp(-β(t - tⱼ))
    """
    return torch.exp(-beta * t_diff)

def spatial_decay(s_diff, Sigma_zeta):
    """
    Gaussian spatial decay (Equation 10)
    ζ(s - sⱼ) = exp(-(s - sⱼ)ᵀΣ_ζ⁻¹(s - sⱼ))
    """
<<<<<<< HEAD
    return torch.exp(-torch.matmul(torch.matmul(s_diff.unsqueeze(1), 
                                               torch.inverse(Sigma_zeta)), 
                                 s_diff.unsqueeze(-1)).squeeze())
=======
    # Ensure all tensors are on the same device
    device = Sigma_zeta.device
    s_diff = s_diff.to(device)
    
    # Ensure s_diff is 2D vector [2,]
    if len(s_diff.shape) == 1:
        s_diff = s_diff.view(-1)  # Flatten to 1D
    elif len(s_diff.shape) == 2:
        s_diff = s_diff.view(-1)  # Flatten to 1D
    
    if s_diff.shape[0] != 2:
        raise ValueError(f"Expected s_diff to have 2 elements, got {s_diff.shape[0]}")
    
    # Reshape for matrix multiplication
    s_diff = s_diff.view(2, 1)  # Shape: [2, 1]
    
    # Compute inverse of Sigma_zeta
    Sigma_zeta_inv = torch.inverse(Sigma_zeta)  # Shape: [2, 2]
    
    # Compute quadratic form: (s_diff)ᵀ Σ⁻¹ (s_diff)
    quad_form = torch.matmul(torch.matmul(s_diff.T, Sigma_zeta_inv), s_diff)  # Shape: [1, 1]
    
    return torch.exp(-quad_form.squeeze())  # Return scalar

>>>>>>> f516a0a (Final clean commit)

def compute_external_effect(t, s, h_l, tau_l, Sigma_k):
    """
    Compute external effect α(t,s|I) based on Equation (6)
    """
<<<<<<< HEAD
    L, H, W, _ = h_l.shape
    device = h_l.device
    alpha = torch.zeros(1, device=device)
    
    dx = 1.0 / W
    dy = 1.0 / H
    
    for l in range(L):
        for h in range(H):
            for w in range(W):
                h_lhw = h_l[l, h, w]
                x_hw = torch.tensor([w * dx, h * dy], device=device)
                f_temporal = temporal_kernel(t, tau_l[l], delta=1.0)
                f_spatial = spatial_kernel(s, x_hw, Sigma_k)
                alpha += h_lhw * f_temporal * f_spatial
=======

    # print("Shape of h_l:", h_l.shape)
    B, T, C, H, W = h_l.shape
    device = h_l.device
    t = t.to(device)
    s = s.to(device)
    tau_l = tau_l.to(device)
    Sigma_k = Sigma_k.to(device)
    
    
    # Initialize output tensor
    alpha = torch.zeros(B, device=device)
    
    # Grid spacing
    dx = 1.0 / W
    dy = 1.0 / H
    
    # Move inputs to correct device
    t = t.to(device)
    s = s.to(device)
    tau_l = tau_l.to(device)
    
    for b in range(B):
        for t_idx in range(T):
            for h in range(H):
                for w in range(W):
                    h_lbt = h_l[b, t_idx, 0, h, w]
                    x_hw = torch.tensor([w * dx, h * dy], device=device)
                    
                    f_temporal = temporal_kernel(t, tau_l[t_idx], delta=1.0)
                    f_spatial = spatial_kernel(s, x_hw, Sigma_k)
                    
                    alpha[b] += h_lbt * f_temporal * f_spatial
>>>>>>> f516a0a (Final clean commit)
    
    return alpha

def compute_spatiotemporal_decay(t_diff, s_diff, beta, Sigma_zeta):
    """
    Compute spatio-temporal decay γ(t-t_j, s-s_j)
    """
<<<<<<< HEAD
    kappa = temporal_decay(t_diff, beta)
    zeta = spatial_decay(s_diff, Sigma_zeta)
=======
    device = Sigma_zeta.device
    t_diff = t_diff.to(device)
    s_diff = s_diff.to(device)
    beta = torch.tensor(beta).to(device)
    # Temporal decay (vectorized)
    kappa = temporal_decay(t_diff, beta).to(device)  # Shape: [N]
    
    # Ensure s_diff is properly shaped for batch processing
    if len(s_diff.shape) == 2:  # If shape is [N, 2]
        zeta = torch.stack([spatial_decay(s_diff[i], Sigma_zeta).to(device) for i in range(s_diff.shape[0])])
    else:  # If shape is [2]
        zeta = spatial_decay(s_diff, Sigma_zeta).to(device)
    
    # Ensure kappa and zeta are properly broadcast
    if len(kappa.shape) != len(zeta.shape):
        if len(kappa.shape) > len(zeta.shape):
            zeta = zeta.expand_as(kappa)
        else:
            kappa = kappa.expand_as(zeta)
    
>>>>>>> f516a0a (Final clean commit)
    return kappa * zeta

def continuous_conv_kernel(t_diff, s_diff, Sigma_k):
    """
    Compute continuous convolution kernel f(t-τ, s-x)
    """
    delta = 1.0
    f_temporal = torch.where(torch.abs(t_diff) < delta,
                           torch.ones_like(t_diff),
                           torch.zeros_like(t_diff))
    f_spatial = spatial_kernel(s_diff, torch.zeros_like(s_diff), Sigma_k)
<<<<<<< HEAD
    return f_temporal * f_spatial
=======
    return f_temporal * f_spatial

def compute_external_effect_vectorized(times, locations, h_l, tau_l, Sigma_k):
    """Vectorized implementation of external effect computation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move all inputs to device and ensure proper shapes
    h_l = h_l.to(device)  # [B, T, C, H, W]
    times = times.to(device)  # [N]
    locations = locations.to(device)  # [N, 2]
    tau_l = tau_l.to(device)  # [L]
    Sigma_k = Sigma_k.to(device)  # [2, 2]
    
    # Get dimensions
    B, T, C, H, W = h_l.shape
    N = len(times)
    L = len(tau_l)
    
    # Create grid coordinates [H, W, 2]
    x_coords = torch.linspace(0, 1, W, device=device)
    y_coords = torch.linspace(0, 1, H, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid_coords = torch.stack([grid_x, grid_y], dim=-1)
    
    # Reshape locations for broadcasting [N, 1, 1, 2]
    locations = locations.view(N, 1, 1, 2)
    
    # Compute spatial kernels directly without sparse operations
    spatial_diffs = locations - grid_coords.unsqueeze(0)  # [N, H, W, 2]
    Sigma_k_inv = torch.inverse(Sigma_k)
    
    # Compute spatial kernels efficiently
    spatial_kernels = torch.exp(-torch.sum(
        torch.matmul(spatial_diffs.view(-1, 2), Sigma_k_inv) * 
        spatial_diffs.view(-1, 2),
        dim=-1
    )).view(N, H, W)  # [N, H, W]
    
    # Compute temporal kernels [N, T]
    times = times.view(N, 1)  # [N, 1]
    tau_grid = tau_l.unsqueeze(0)  # [1, L]
    temporal_kernels = (torch.abs(times - tau_grid) < 1.0).float()  # [N, L]
    
    # Adjust temporal kernels to match feature map time dimension
    temporal_kernels_adjusted = F.interpolate(
        temporal_kernels.unsqueeze(1),  # [N, 1, L]
        size=T,
        mode='linear',
        align_corners=False
    ).squeeze(1)  # [N, T]
    
    # Compute final external effects [N]
    alphas = torch.zeros(N, device=device)
    for b in range(B):
        feature_map = h_l[b]  # [T, C, H, W]
        
        # Compute weighted sum using einsum for efficiency
        weighted_sum = torch.einsum(
            'nhw,tchw,nt->n',
            spatial_kernels,
            feature_map,
            temporal_kernels_adjusted
        )
        alphas += weighted_sum
    
    return alphas

# def compute_spatiotemporal_decay_vectorized(t_diffs, s_diffs, beta, Sigma_zeta):
#     """Vectorized implementation of spatiotemporal decay with memory-efficient chunking"""
#     device = t_diffs.device
#     chunk_N, total_N = t_diffs.shape  # Get actual dimensions
#     chunk_size = 100  # Further reduced chunk size
    
#     # Print shape information for debugging
#     print(f"Input shapes - t_diffs: {t_diffs.shape}, s_diffs: {s_diffs.shape}")
    
#     # Initialize output tensor
#     decay_matrix = torch.zeros((chunk_N, total_N), device=device)
    
#     # Process in sub-chunks to save memory
#     for i in range(0, chunk_N, chunk_size):
#         i_end = min(i + chunk_size, chunk_N)
#         current_chunk_size = i_end - i
        
#         # Get current chunk
#         t_chunk = t_diffs[i:i_end]  # [current_chunk_size, total_N]
#         s_chunk = s_diffs[i:i_end]  # [current_chunk_size, total_N, 2]
        
#         # Compute temporal decay for chunk
#         temporal_decay = torch.exp(-beta * t_chunk) * (t_chunk > 0).float()
        
#         # Compute spatial decay for chunk
#         Sigma_zeta_inv = torch.inverse(Sigma_zeta)
        
#         # Process spatial decay in sub-chunks
#         for j in range(0, total_N, chunk_size):
#             j_end = min(j + chunk_size, total_N)
            
#             # Get current sub-chunk of spatial differences
#             s_sub_chunk = s_chunk[:, j:j_end]  # [current_chunk_size, sub_chunk_size, 2]
            
#             # Compute quadratic form for spatial decay
#             quad_form = torch.sum(
#                 torch.matmul(
#                     s_sub_chunk.reshape(-1, 2),
#                     Sigma_zeta_inv
#                 ) * s_sub_chunk.reshape(-1, 2),
#                 dim=-1
#             ).reshape(current_chunk_size, j_end - j)
            
#             # Compute spatial decay for sub-chunk
#             spatial_decay_sub = torch.exp(-quad_form)
            
#             # Combine with temporal decay for this sub-chunk
#             decay_matrix[i:i_end, j:j_end] = (
#                 temporal_decay[:, j:j_end] * spatial_decay_sub
#             )
            
#             # Clear intermediate results
#             del quad_form, spatial_decay_sub
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
        
#         # Clear chunk intermediates
#         del temporal_decay
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
    
#     return decay_matrix

def compute_spatiotemporal_decay_vectorized(t_diffs, s_diffs, beta, Sigma_zeta):
    device = t_diffs.device
    chunk_N, total_N = t_diffs.shape
    chunk_size = 100  # Reduced chunk size
    
    # Initialize output tensor
    decay_matrix = torch.zeros((chunk_N, total_N), device=device)
    
    # Process in chunks
    for i in range(0, chunk_N, chunk_size):
        i_end = min(i + chunk_size, chunk_N)
        
        # Get current chunk
        t_chunk = t_diffs[i:i_end]
        s_chunk = s_diffs[i:i_end]
        
        # Compute temporal decay with numerical stability
        temporal_decay = torch.exp(-torch.clamp(beta * t_chunk, max=100)) * (t_chunk > 0).float()
        
        # Process spatial decay in sub-chunks
        for j in range(0, total_N, chunk_size):
            j_end = min(j + chunk_size, total_N)
            
            # Compute quadratic form with numerical stability
            quad_form = torch.clamp(
                torch.sum(
                    torch.matmul(s_chunk[:, j:j_end].reshape(-1, 2), Sigma_zeta) * 
                    s_chunk[:, j:j_end].reshape(-1, 2),
                    dim=-1
                ).reshape(i_end - i, j_end - j),
                max=50
            )
            
            # Compute spatial decay
            spatial_decay = torch.exp(-quad_form)
            
            # Combine decays
            decay_matrix[i:i_end, j:j_end] = temporal_decay[:, j:j_end] * spatial_decay
            
            # Clear memory
            del quad_form, spatial_decay
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        del temporal_decay
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return decay_matrix
>>>>>>> f516a0a (Final clean commit)
