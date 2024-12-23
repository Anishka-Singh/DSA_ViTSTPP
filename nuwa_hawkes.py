import torch
import torch.nn as nn
from nvwa_downstream_pred import Nvwa_enchane_SimVP
from ConvHawkes.models.convhawkes import ConvHawkes

class NuwaHawkesModel(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, args=None):
        super().__init__()
        # Initialize NuwaDynamics model
        self.nuwa_model = Nvwa_enchane_SimVP(
            shape_in=shape_in,
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            args=args
        )
        
        # Initialize ConvHawkes
        self.convhawkes = ConvHawkes(
            N_l=N_S,  # Match encoder depth
            beta=torch.tensor(1.0),
            Sigma_k=torch.eye(2),
            Sigma_zeta=torch.eye(2),
            mu=torch.tensor(0.1)
        )
    
    def _convert_to_events(self, predictions, threshold=0.5):
        """Convert predictions to event sequences"""
        B, T, C, H, W = predictions.shape
        events_list = []
        
        for b in range(B):
            batch_events = []
            for t in range(T):
                # Find significant activations
                activations = predictions[b, t, 0] > threshold
                y_coords, x_coords = torch.where(activations)
                
                # Convert to normalized coordinates
                for y, x in zip(y_coords, x_coords):
                    t_norm = t / T
                    s = (x.item() / W, y.item() / H)
                    batch_events.append((t_norm, s))
            
            events_list.append(batch_events)
        
        return events_list
    
    def forward(self, x):
        # Get NuwaDynamics predictions
        predictions = self.nuwa_model(x)
        
        # Convert predictions to events
        events_list = self._convert_to_events(predictions)
        
        # Process through ConvHawkes
        total_loss = 0
        for b, events in enumerate(events_list):
            if events:  # Only process if events exist
                hawkes_loss = self.convhawkes(
                    image_sequence=predictions[b:b+1],  # Add batch dimension
                    events=events,
                    T=1.0,  # Normalized time window
                    S=[(0, 1), (0, 1)]  # Normalized spatial bounds
                )
                total_loss += hawkes_loss
        
        return predictions, total_loss / len(events_list) if events_list else total_loss