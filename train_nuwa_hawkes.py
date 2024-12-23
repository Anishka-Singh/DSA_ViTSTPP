import os
import torch
import logging
import datetime 
from tqdm import tqdm
import time
import numpy as np
import os.path as osp
from ConvHawkes.models.convhawkes import ConvHawkes
from nvwa_downstream_pred import Nvwa_enchane_SimVP
from utils import *
from API import *

class HawkesTrainer:
    def __init__(self, args):
        self.args = args
        self.log_file = self.setup_hawkes_logging()  
        self.device = self._acquire_device()  
        
        self._preparation()
        
        # Load pre-trained NuwaDynamics model
        self.nuwa_model = self._load_nuwa_model()
        self.nuwa_model.eval()  # Set to evaluation mode
        
        # Initialize ConvHawkes
        self.hawkes_model = ConvHawkes(
            N_l=args.N_S,
            beta=torch.tensor(args.beta),
            Sigma_k=torch.eye(2) * args.sigma_k_scale,
            Sigma_zeta=torch.eye(2) * args.sigma_zeta_scale,
            mu=torch.tensor(args.mu)
        ).to(self.device)
        
        self.get_data()
        self._select_optimizer()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu-1)
            device = torch.device('cuda:{}'.format(0))
            self.log_to_file(f'Use GPU: {self.args.gpu}', self.log_file)
        else:
            device = torch.device('cpu')
            self.log_to_file('Use CPU', self.log_file)
        return device

    def setup_hawkes_logging(self):
        """Setup logging directory and file for Hawkes training"""
        # Define the directory structure
        base_dir = self.args.res_dir  # e.g., './output/simvp_nighttime_mask'
        experiment_dir = os.path.join(base_dir, self.args.ex_name)  # e.g., './output/simvp_nighttime_mask/Debug'
        hawkes_dir = os.path.join(experiment_dir, 'hawkes_Debug')  # e.g., './output/simvp_nighttime_mask/Debug/hawkes_Debug'
        
        # Create directories if they don't exist
        os.makedirs(hawkes_dir, exist_ok=True)
        
        # Define the log filename
        log_filename = os.path.join(hawkes_dir, 'hawkes_log.log')
        print(f"Logging initialized. Log file: {log_filename}")
        
        # Add a timestamp separator when starting new training
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_filename, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"New Training Run - {timestamp}\n")
            f.write(f"{'='*50}\n\n")
        
        # Setup logging configuration
        logging.basicConfig(level=logging.INFO, filename=log_filename,
                            filemode='a', format='%(asctime)s - %(message)s')
        
        return log_filename

    def log_to_file(self, message, log_file):
        """Log message to file and print to console"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        
        print(f"Logging: {formatted_message}")  # Debug print
    
        print(f"Writing to log file: {log_file}")  # Debug print
        with open(log_file, 'a') as f:
            f.write(formatted_message + '\n')
            return True
        return False
    

    def _preparation(self):
        # Create directories for results
        self.path = osp.join(self.args.res_dir, 'hawkes_' + self.args.ex_name)
        check_dir(self.path)
        
        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

    def _load_nuwa_model(self):
        # Initialize NuwaDynamics model
        nuwa_model = Nvwa_enchane_SimVP(
            tuple(self.args.in_shape), 
            self.args.hid_S,
            self.args.hid_T, 
            self.args.N_S, 
            self.args.N_T, 
            args=self.args
        ).to(self.device)
        
        checkpoint = torch.load(self.args.nuwa_checkpoint, map_location=self.device)
        nuwa_model.load_state_dict(checkpoint['model_state_dict'])
        nuwa_model.eval()
        self.log_to_file(f'Loaded NuwaDynamics model from {checkpoint}', self.log_file)
        return nuwa_model

    def get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.hawkes_model.parameters(), 
            lr=self.args.hawkes_lr
        )
        return self.optimizer

    def _convert_to_events(self, predictions, threshold=0.5):
        """Convert NuwaDynamics predictions to event sequences"""
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

    def train_hawkes(self):
        self.log_to_file("Starting Hawkes process training...", '/home/ansingh/Nuwa_Hawkes/output/simvp_nighttime_mask/Debug/hawkes_Debug/hawkes_log.log')
        
        best_val_loss = float('inf')
        
        for epoch in range(self.args.hawkes_epochs):
            self.hawkes_model.train()
            train_losses = []
            count = 0
            
            # Log model parameters count
            count += sum(p.numel() for p in self.hawkes_model.parameters() if p.requires_grad)
            count += sum(p.numel() for p in self.nuwa_model.parameters() if p.requires_grad)
            if(not self.log_to_file(f"Total trainable parameters: {count}", self.log_file)):
                raise Exception('did not log after param count')

            
            train_pbar = tqdm(self.train_loader)
            for batch_idx, (batch_x, batch_y) in enumerate(train_pbar):
                batch_x = batch_x.float().to(self.device)
                if(not self.log_to_file(f"Processing batch {batch_idx + 1}/{len(self.train_loader)}", '/home/ansingh/Nuwa_Hawkes/output/simvp_nighttime_mask/Debug/hawkes_Debug/hawkes_log.log')):
                    raise Exception('did not log at batch proc')
                
                # Get NuwaDynamics predictions
                with torch.no_grad():
                    predictions = self.nuwa_model(batch_x)
                
                # Convert predictions to events
                events_list = self._convert_to_events(predictions, threshold=self.args.event_threshold)
                
                # Train Hawkes model
                self.optimizer.zero_grad()
                total_loss = 0
                
                for b, events in enumerate(events_list):
                    if events:  # Only process if events exist
                        hawkes_loss = self.hawkes_model(
                            image_sequence=predictions[b:b+1],
                            events=events,
                            T=1.0,
                            S=[(0, 1), (0, 1)]
                        )
                        total_loss += hawkes_loss
                
                # Average loss over batch
                if len(events_list) > 0:
                    batch_loss = total_loss / len(events_list)
                    batch_loss.backward()
                    self.optimizer.step()
                    train_losses.append(batch_loss.item())
                    train_pbar.set_description(f'Hawkes train loss: {batch_loss.item():.4f}')
                else:
                    self.log_to_file(f"Skipping batch {batch_idx} due to no events.", self.log_file)
            
            # Validation
            val_loss = self.validate_hawkes()
            
            # Save checkpoint if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_hawkes_checkpoint('best_hawkes_model')
                self.log_to_file("Saved new best model", self.log_file)
            
            # Log epoch results
            epoch_summary = (
                f"Epoch {epoch + 1}/{self.args.hawkes_epochs}\n"
                f"Train Loss: {np.mean(train_losses):.4f}\n"
                f"Validation Loss: {val_loss:.4f}"
            )
            self.log_to_file(epoch_summary, self.log_file)

    def validate_hawkes(self):
        self.hawkes_model.eval()
        val_losses = []
        total_events = 0
        valid_batches = 0
        
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(self.vali_loader):
                batch_x = batch_x.float().to(self.device)
                self.log_to_file(f"Validating batch {batch_idx + 1}/{len(self.vali_loader)}", self.log_file)
                
                # Get NuwaDynamics predictions
                predictions = self.nuwa_model(batch_x)
                
                # Convert predictions to events
                events_list = self._convert_to_events(predictions)
                
                # Compute Hawkes loss
                total_loss = 0
                batch_events = 0
                batch_valid = 0
                
                for b, events in enumerate(events_list):
                    if events:
                        try:
                            hawkes_loss = self.hawkes_model(
                                image_sequence=predictions[b:b+1],
                                events=events,
                                T=1.0,
                                S=[(0, 1), (0, 1)]
                            )
                            total_loss += hawkes_loss
                            batch_valid += 1
                            batch_events += len(events)
                        except Exception as e:
                            self.log_to_file(f"Validation error in batch {b}: {str(e)}", self.log_file)
                
                if batch_valid > 0:
                    batch_loss = total_loss / batch_valid
                    val_losses.append(batch_loss.item())
                    total_events += batch_events
                    valid_batches += batch_valid
                else:
                    self.log_to_file(f"Batch {batch_idx} skipped due to no valid events.", self.log_file)
        
        avg_loss = np.mean(val_losses) if val_losses else float('inf')
        self.log_to_file(f"Validation completed. Average Loss: {avg_loss:.4f}", self.log_file)
        return avg_loss

    def _normalize_events(self, events):
        """Normalize event times and locations."""
        times = torch.tensor([e[0] for e in events])
        locations = torch.tensor([e[1] for e in events])
        
        # Normalize times to [0,1]
        times = (times - times.min()) / (times.max() - times.min() + 1e-10)
        
        # Normalize locations to [-1,1] for each dimension
        for dim in range(2):
            loc_min = locations[:, dim].min()
            loc_max = locations[:, dim].max()
            locations[:, dim] = 2 * (locations[:, dim] - loc_min) / (loc_max - loc_min + 1e-10) - 1
        
        return [(t.item(), loc.tolist()) for t, loc in zip(times, locations)]

    def _save_hawkes_checkpoint(self, name):
        checkpoint = {
            'hawkes_model': self.hawkes_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': self.args.__dict__
        }
        torch.save(checkpoint, os.path.join(self.checkpoints_path, f'{name}.pth'))
        self.log_to_file(f"Checkpoint saved: {name}.pth", self.log_file)