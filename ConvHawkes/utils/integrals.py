# utils/integrals.py
import torch
<<<<<<< HEAD

=======
from ConvHawkes.utils.kernels import compute_external_effect_vectorized
from time import time
>>>>>>> f516a0a (Final clean commit)
def compute_temporal_integral(t_n, T, tau_l, beta, delta_tau):
    """
    Compute temporal integral based on Equation (17)
    """
    lower_bound = max(t_n, tau_l - delta_tau)
    upper_bound = min(T, tau_l + delta_tau)
    
    if lower_bound >= upper_bound:
        return torch.tensor(0.0, device=t_n.device)
    
    G_upper = -torch.exp(-beta * (upper_bound - t_n))
    G_lower = -torch.exp(-beta * (lower_bound - t_n))
    
    return G_upper - G_lower

def compute_spatial_integral(s_n, x_hw, Sigma_k, Sigma_zeta, S):
    """
    Compute spatial integral based on Equation (18)
    """
    # Combined covariance
    Sigma_c = torch.inverse(
        torch.inverse(Sigma_k) + torch.inverse(Sigma_zeta)
    )
    
    # Combined mean
    x_c = Sigma_c @ (
        torch.inverse(Sigma_k) @ s_n + 
        torch.inverse(Sigma_zeta) @ x_hw
    )
    
    # Compute normalization factor
    det_sum = torch.det(2 * torch.pi * (Sigma_k + Sigma_zeta))
    diff = s_n - x_hw #1D tensor of shape 2 (for x and y coordinates)
    exp_term = torch.exp(
        -0.5 * torch.matmul(
            torch.matmul(diff.unsqueeze(0), # transforms shape from 2 to 1,2 (added row dimension)
                        torch.inverse(Sigma_k + Sigma_zeta)),
            diff.unsqueeze(-1) # transforms shape from 2 to 2,1 (added column dimension)
        ).squeeze()
    )
     
    # Compute error function term for spatial bounds
    def error_function_term(x, mu, sigma, a, b):

        # # not normalised Gaussian kernel
        # sqrt_pi_sigma = torch.sqrt(torch.tensor(torch.pi)) * sigma
        # return sqrt_pi_sigma * 0.5 * (torch.erf((b - mu) / (sigma * torch.sqrt(torch.tensor(2.0)))) - 
        #              torch.erf((a - mu) / (sigma * torch.sqrt(torch.tensor(2.0)))))

        # normalised Gaussian kernel
        return 0.5 * (torch.erf((b - mu) / (sigma * torch.sqrt(torch.tensor(2.0)))) - 
                     torch.erf((a - mu) / (sigma * torch.sqrt(torch.tensor(2.0)))))
      
    # Compute integral over spatial bounds
    x_integral = error_function_term(x_c[0], x_c[0], torch.sqrt(Sigma_c[0,0]), S[0][0], S[0][1])
    y_integral = error_function_term(x_c[1], x_c[1], torch.sqrt(Sigma_c[1,1]), S[1][0], S[1][1])
    
    return (1.0 / torch.sqrt(det_sum)) * exp_term * x_integral * y_integral

def compute_Lambda_n(times, locations, T, S, h_l, tau_l, beta, Sigma_k, Sigma_zeta):
    """
    Equation (12)
    """
    N = len(times)
    L, H, W, _ = h_l.shape
    device = h_l.device
    Lambda_n = torch.zeros(1, device=device) 
<<<<<<< HEAD
    
=======
    t2=time()
>>>>>>> f516a0a (Final clean commit)
    for n in range(N):
        t_n, s_n = times[n], locations[n]
        
        for l in range(L):
            for h in range(H):
                for w in range(W):
                    h_lhw = h_l[l, h, w] #CNN feature value at position (l,h,w)
                    x_hw = torch.tensor([w/W, h/H], device=device) # Grid position to normalized coordinates (0-1 range)
                    
                    temporal_int = compute_temporal_integral(
                        t_n, T, tau_l[l], beta, delta_tau=1.0
                    )
                    
                    spatial_int = compute_spatial_integral(
                        s_n, x_hw, Sigma_k, Sigma_zeta, S
                    )
                    
                    Lambda_n += h_lhw * temporal_int * spatial_int
<<<<<<< HEAD
=======
                print(time()-t2,"second innermost loop point")
    print(time()-t2)
    return Lambda_n

def compute_Lambda_n_vectorized(times, locations, T, S, h_l, tau_l, beta, Sigma_k, Sigma_zeta):
    """Vectorized implementation of Lambda_n computation"""
    device = h_l.device
    B, T_len, C, H, W = h_l.shape
    N = len(times)
    
    # Create integration grid
    n_grid = 20  # Adjust based on memory constraints
    t_grid = torch.linspace(0, T, n_grid, device=device)
    x_grid = torch.linspace(S[0][0], S[0][1], n_grid, device=device)
    y_grid = torch.linspace(S[1][0], S[1][1], n_grid, device=device)
    
    # Create meshgrid for integration
    t_mesh, x_mesh, y_mesh = torch.meshgrid(t_grid, x_grid, y_grid, indexing='ij')
    grid_points = torch.stack([t_mesh, x_mesh, y_mesh], dim=-1)
    
    # Compute external effects for grid points using sparse operations
    grid_locations = grid_points[..., 1:]
    grid_times = grid_points[..., 0]
    
    # Reshape for sparse operations
    flat_times = grid_times.reshape(-1)
    flat_locations = grid_locations.reshape(-1, 2)
    
    # Compute external effects
    external_effects = compute_external_effect_vectorized(
        flat_times,
        flat_locations,
        h_l, tau_l, Sigma_k
    ).reshape(grid_points.shape[:-1])
    
    # Compute integral using trapezoidal rule
    dt = t_grid[1] - t_grid[0]
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    
    # Use sparse operations for large tensors
    Lambda_n = torch.sparse.sum(
        torch.sparse_coo_tensor(
            indices=torch.nonzero(external_effects).t(),
            values=external_effects[external_effects != 0],
            size=external_effects.shape
        )
    ) * dt * dx * dy
>>>>>>> f516a0a (Final clean commit)
    
    return Lambda_n