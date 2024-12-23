import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

Image.MAX_IMAGE_PIXELS = None

class NighttimeDataset(Dataset):
    def __init__(self, data_path, tile_size=128, transform=None, max_files=None, max_patches=None):
        # self.files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.tif')]
        all_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.tif')]
        self.files = all_files[:max_files] if max_files else all_files
        self.tile_size = tile_size
        self.transform = transform or transforms.ToTensor()
        self.file_indices = []

        # Build file indices for each patch
        for img_path in self.files:
            img = Image.open(img_path)
            width, height = img.size
            num_patches = (height // self.tile_size) * (width // self.tile_size)
            # self.file_indices.extend([(img_path, i) for i in range(num_patches)])
            if max_patches:
                num_patches = min(num_patches, max_patches)
            self.file_indices.extend([(img_path, i) for i in range(num_patches)])

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        img_path, patch_idx = self.file_indices[idx]
        img = Image.open(img_path)
        img = np.array(img)

        # Calculate patch coordinates
        patches_per_row = img.shape[1] // self.tile_size
        row = (patch_idx // patches_per_row) * self.tile_size
        col = (patch_idx % patches_per_row) * self.tile_size

        # Extract patch
        patch = img[row:row + self.tile_size, col:col + self.tile_size]
        # print(f"Patch shape before transformation: {patch.shape}")

        # Transform to tensor
        if self.transform:
            patch = self.transform(patch)  # Shape: [1, H, W]
        # print(f"Patch shape after transformation: {patch.shape}")

        # Create input and output frames
        input_frames = patch.unsqueeze(0)  # Shape: [1, 1, H, W]
        output_frames = patch.unsqueeze(0)  # Shape: [1, 1, H, W]

        # Stack them together
        input_frames = torch.cat([input_frames, input_frames], dim=0)  # Shape: [2, 1, H, W]
        output_frames = torch.cat([output_frames, output_frames], dim=0)  # Shape: [2, 1, H, W]

        # print(f"Input shape: {input_frames.shape}, Output shape: {output_frames.shape}")

        return input_frames, output_frames

data_path = '/home/ansingh/NuwaDynamics_nightTime/data/nighttime/train'  
for file in os.listdir(data_path):
    if file.endswith('.tif'):
        with Image.open(os.path.join(data_path, file)) as img:
            print(f"{file}: {img.size}")

def load_data(batch_size, val_batch_size, data_root, num_workers, max_files=None, max_patches=None):
    """
    Loads the train and test datasets, handling .tif files and patchifying them.
    
    Args:
        batch_size (int): Batch size for training data.
        val_batch_size (int): Batch size for validation data.
        data_root (str): Path to the data directory.
        num_workers (int): Number of workers for the DataLoader.
    
    Returns:
        tuple: train_loader, vali_loader, test_loader, mean, std
    """
    train_dataset = NighttimeDataset(data_path=os.path.join(data_root, 'train/'), max_files=max_files, max_patches=max_patches)
    print(f"Number of patches in the train dataset: {len(train_dataset)}")
    test_dataset = NighttimeDataset(data_path=os.path.join(data_root, 'test/'), max_files=max_files, max_patches=max_patches)
    print(f"Number of patches in the test dataset: {len(test_dataset)}")
    
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0, persistent_workers=False)
    dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # Calculate mean and std for normalization 
    mean, std = 0,1
    return dataloader_train, None, dataloader_test, mean, std


