# utils/kernels.py
import torch

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
    diff = s - x
    return torch.exp(-torch.matmul(torch.matmul(diff.unsqueeze(1), 
                                               torch.inverse(Sigma_k)), 
                                 diff.unsqueeze(-1)).squeeze())

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
    return torch.exp(-torch.matmul(torch.matmul(s_diff.unsqueeze(1), 
                                               torch.inverse(Sigma_zeta)), 
                                 s_diff.unsqueeze(-1)).squeeze())

def compute_external_effect(t, s, h_l, tau_l, Sigma_k):
    """
    Compute external effect α(t,s|I) based on Equation (6)
    """
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
    
    return alpha

def compute_spatiotemporal_decay(t_diff, s_diff, beta, Sigma_zeta):
    """
    Compute spatio-temporal decay γ(t-t_j, s-s_j)
    """
    kappa = temporal_decay(t_diff, beta)
    zeta = spatial_decay(s_diff, Sigma_zeta)
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
    return f_temporal * f_spatial