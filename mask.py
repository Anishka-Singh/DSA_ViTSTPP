import numpy as np
import os
import sys
# Open original image
patch_number = 1

def block_index_to_coordinates(block_index):
    if not 0 <= block_index < 64:
        raise ValueError("Block index must be between 0 and 63")
    # Block size
    block_size = 16

    # Calculate row and column index
    row_index = block_index // 8  # 8 blocks per row
    col_index = block_index % 8   # 8 blocks per column

    # Calculate the top-left coordinates of the block
    top_left_x = col_index * block_size
    top_left_y = row_index * block_size

    return (top_left_x, top_left_y)

def mixup_region(image_array, x, y, block_size):
    print(image_array.shape)
    _, h, w = image_array.shape
    print(h, w)
    x_start = max(x - block_size, 0)
    x_end = min(x + 2*block_size, w)
    y_start = max(y - block_size, 0)
    y_end = min(y + 2*block_size, h)
    # print(x_start, x_end, y_start, y_end)
    # Mix target region with surrounding regions
    region = image_array[:, y:y+block_size, x:x+block_size]
    surrounding_regions = []
    for i in range(int(x_start / block_size), int(x_end / block_size)):
        for j in range(int(y_start / block_size), int(y_end / block_size)):
            print(i, j)
            surrounding_region = image_array[:, j*block_size:(j+1)*block_size, i*block_size:(i+1)*block_size]
            surrounding_regions.append(surrounding_region)
    print(len(surrounding_regions))
    mixed_region = 0
    for i in range(len(surrounding_regions)):
        # print(surrounding_regions[i].shape)
        mixed_region += surrounding_regions[i] / len(surrounding_regions)

    # Place the mixed region back into the original image
    image_array[:, y:y+block_size, x:x+block_size] = mixed_region.astype(np.uint8)
    return image_array

# Load attention maps and images
attention_file_path = './data/all_attention_maps3.npy' # 833, 10, 1, 64, 64
attention_map = np.load(attention_file_path, allow_pickle=True)

image_file_path = './data/SEVIR_IR069_STORMEVENTS_2018_0101_0630.npy' # 833, 20, 128, 128
images = np.load(image_file_path, allow_pickle=True)

# Ensure the number of attention maps matches the number of images
if len(attention_map) != len(images):
    print(f"Number of attention maps: {len(attention_map)}, number of images: {len(images)}")
    # sys.exit()
else:
    print("The number of attention maps and images matches")

patch_number = 1
s = 0

processed_images = []

for batch_attention_maps, image_array in zip(attention_map, images):
    # Only process the first half of the image
    image_array_front = image_array[:10, :, :]  # Extract the first half
    image_array_back = image_array[10:, :, :]   # Extract the second half
    # print(batch_attention_maps.shape)
    # Calculate the sum of each block's columns
    column_sums = np.sum(batch_attention_maps, axis=2)
    min_weight_columns = np.argpartition(column_sums, patch_number, axis=2)[:, :, :patch_number]
    min_weight_columns = min_weight_columns.reshape(batch_attention_maps.shape[0], -1)
    # print(min_weight_columns)

    # Iterate through each attention map
    for col_index_all in min_weight_columns:
        for col_index in col_index_all:
            x, y = block_index_to_coordinates(col_index)
            # print(y.dtype)
            image_array = mixup_region(image_array_front, x, y, 16)  # Use block size 16 for mixup
    # Concatenate the first half and the second half
    processed_image = np.concatenate((image_array, image_array_back), axis=0)
    # Save the processed image to the list
    processed_images.append(processed_image)

# Save the processed images array to a new .npy file
processed_images = np.array(processed_images)
print(processed_images.shape)
np.save('./data/mask3.npy', processed_images)
