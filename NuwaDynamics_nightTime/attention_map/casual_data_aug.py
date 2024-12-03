
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib as mpl
import random

attention_maps = np.load('/home/ansingh/NuwaDynamics/data/attention_map/Attention_map.npy')
print(attention_maps.shape)

mycolor=['#89ABC5','#6785BE','#5D46A3','#491463','#45113C','#4E1344','#87274F','#B15352']
cmap_color = mpl.colors.LinearSegmentedColormap.from_list('my_list', mycolor)
cmap = cmap_color

batch_index = random.randint(0, 10)
print(batch_index)
attention_map = attention_maps[batch_index][0]
attention_map = attention_map
plt.imshow(attention_map, cmap=cmap)
plt.colorbar()
plt.title('Attention Map Heatmap')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import random

time_index = 0
attentionmap = attention_maps[time_index][0]
attentionmap = attentionmap
print(attentionmap.shape)
plt.imshow(attentionmap)
column_sums = np.sum(attentionmap, axis=0)
largest_indices = np.argsort(column_sums)[-5:]
smallest_indices = np.argsort(column_sums)[:5]
largest_sums = column_sums[largest_indices]
smallest_sums = column_sums[smallest_indices]
largest_indices, largest_sums, smallest_indices, smallest_sums

import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import scipy.io as sio

class NighttimeDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.tif')]
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),  # Resize to match ViT input
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        img = self.transform(img)
        input_frames = img[:10, :, :]  # First 10 frames
        output_frames = img[10:, :, :]  # Remaining frames
        return input_frames, output_frames

def load_data(batch_size, val_batch_size, data_root, num_workers):
    train_dataset = NighttimeDataset(data_path=os.path.join(data_root, 'train/'))
    test_dataset = NighttimeDataset(data_path=os.path.join(data_root, 'test/'))
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  num_workers=num_workers)
    dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers)
    mean, std = train_dataset.mean, train_dataset.std
    return dataloader_train, None, dataloader_test, mean, std


if __name__ == '__main__':
    dataloader_train, dataloader_validation, dataloader_test, mean, std = load_data(batch_size=10, 
                                                                                    val_batch_size=10, 
                                                                                    data_root='/home/ansingh/NuwaDynamics/data/nighttime',
                                                                                    num_workers=8)
    for input_frames, output_frames in iter(dataloader_train):
        print(input_frames.shape, output_frames.shape)
        break



input_frames = input_frames.reshape(10,10,64,64)
print(input_frames.shape)
attention_maps = np.load('/home/ansingh/NuwaDynamics/data/attention_map/Attention_map.npy')
attention_maps = attention_maps.reshape(10,16,16)
print(attention_maps.shape)



import numpy as np

min_attention_indices_per_image = np.argmin(attention_maps.reshape(attention_maps.shape[0], -1), axis=1)
min_attention_positions = np.unravel_index(min_attention_indices_per_image, attention_maps.shape[1:])

min_attention_positions


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches



min_attention_indices_per_image = np.argmin(attention_maps.reshape(attention_maps.shape[0], -1), axis=1)
min_attention_positions = np.unravel_index(min_attention_indices_per_image, attention_maps.shape[1:])

def visualize_min_attention_patches(input_frames, min_attention_positions, patch_size=16):
    fig, axs = plt.subplots(1, input_frames.shape[0], figsize=(20, 20))
    for i, ax in enumerate(axs):
        img = input_frames[i, 0] 
        ax.imshow(img, cmap='jet')
        
        row, col = min_attention_positions
        rect = patches.Rectangle((col[i]*patch_size, row[i]*patch_size),
                                 patch_size, patch_size, linewidth=10, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    plt.show()

visualize_min_attention_patches(input_frames, min_attention_positions)


min_attention_indices_per_image = np.argmin(attention_maps.reshape(attention_maps.shape[0], -1), axis=1)
min_attention_positions = np.unravel_index(min_attention_indices_per_image, attention_maps.shape[1:])
def visualize_min_attention_patches_on_original_images(input_frames, min_attention_positions, patch_size=16):
    fig, axs = plt.subplots(1, len(input_frames), figsize=(20, 20))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(input_frames[i, 0], cmap='gray')
        row, col = min_attention_positions[0][i], min_attention_positions[1][i]
        patch_start_x, patch_start_y = col * patch_size, row * patch_size
        rect = patches.Rectangle((patch_start_x, patch_start_y), patch_size, patch_size, linewidth=100, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
visualize_min_attention_patches_on_original_images(input_frames, min_attention_positions)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def extract_causal_patches_from_attention(attention_maps, patch_size=16, threshold=0.8):
    """
    Extract causal patches from attention maps.

    Args:
        attention_maps (numpy.ndarray): Attention maps with shape [B, H, W].
        patch_size (int): Patch size.
        threshold (float): Threshold for identifying causal patches.

    Returns:
        List[List[Tuple[int, int]]]: Causal patch coordinates for each image.
    """
    causal_patches = []
    for attention in attention_maps:
        # Normalize attention map
        normalized_attention = attention / attention.max()

        # Identify causal regions
        causal_indices = np.argwhere(normalized_attention > threshold)

        # Convert indices to coordinates
        causal_coords = [(row * patch_size, col * patch_size) for row, col in causal_indices]
        causal_patches.append(causal_coords)

    return causal_patches


def visualize_causal_patches(image, causal_coords, patch_size=16):
    """
    Visualize causal patches on an image.

    Args:
        image (numpy.ndarray): Input image with shape [H, W].
        causal_coords (List[Tuple[int, int]]): List of causal patch coordinates.
        patch_size (int): Size of patches.

    Returns:
        None: Displays the visualization.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray")

    for coord in causal_coords:
        rect = patches.Rectangle(coord, patch_size, patch_size, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)

    plt.title("Causal Patches")
    plt.show()


import numpy as np

def mixup_patch(input_frame, min_attention_position, alpha=0.5):
    """
    Performs mixup enhancement on the minimum attention weight patch of the specified image and its right-hand patch.
    :param input_frame: single image data, assuming shape [C, H, W].
    :param min_attention_position: The position of the minimum attention weight patch, in the form (row, col).
    :param alpha: The mixup's mixing ratio.
    :return: The enhanced image.
    """
    patch_size = 16 
    row, col = min_attention_position
    start_x = col * patch_size
    start_y = row * patch_size
    
    if start_x + patch_size < input_frame.shape[2] - patch_size:
        target_patch = input_frame[:, start_y:start_y+patch_size, start_x:start_x+patch_size]
        right_patch = input_frame[:, start_y:start_y+patch_size, start_x+patch_size:start_x+2*patch_size]
        mixed_patch = alpha * target_patch + (1 - alpha) * right_patch
        input_frame[:, start_y:start_y+patch_size, start_x:start_x+patch_size] = mixed_patch
    
    return input_frame


input_frame = input_frames[0] 
min_attention_position = (min_attention_positions[0][0], min_attention_positions[1][0]) 
enhanced_frame = mixup_patch(input_frame, min_attention_position, alpha=0.5)
print(enhanced_frame.reshape(10, 1, 64, 64).shape)


plt.imshow(enhanced_frame[0])





