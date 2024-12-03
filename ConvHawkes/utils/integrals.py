# utils/integrals.py
import torch

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
    
    return Lambda_n