from API.dataloader_nighttime import NighttimeDataset
from torch.utils.data import DataLoader

# Initialize dataset
data_root = "./data/nighttime/train"  # Update this to your dataset path
tile_size = 32
batch_size = 1

dataset = NighttimeDataset(data_path=data_root, tile_size=tile_size)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

print(f"Dataset size: {len(dataset)}")

# Test the dataloader
for idx, (input_frames, output_frames) in enumerate(dataloader):
    print(f"Sample {idx}: Input shape: {input_frames.shape}, Output shape: {output_frames.shape}")  # Expecting tensors
    if idx >= 5:  # Limit output to first 5 samples for testing
        break
