import argparse
import os
import torch
import tifffile as tiff
from nvwa_upstream_pretrain import VisionTransformer
from attention_map.casual_data_aug import extract_causal_patches_from_attention, visualize_causal_patches

def create_parser():
    parser = argparse.ArgumentParser(description="Pipeline for processing nighttime .tif data with Vision Transformer.")

    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Device to use (cuda/cpu)')
    parser.add_argument('--data_root', default='./data/nighttime/', type=str, help='Directory with .tif images')
    parser.add_argument('--output_dir', default='./output/', type=str, help='Directory to save results')

    # Model parameters
    parser.add_argument('--img_size', default=128, type=int, help='Image size for Vision Transformer')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch size for Vision Transformer')
    parser.add_argument('--embed_dim', default=128, type=int, help='Embedding dimension for Vision Transformer')
    parser.add_argument('--depth', default=4, type=int, help='Number of transformer layers')
    parser.add_argument('--num_heads', default=4, type=int, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', default=4.0, type=float, help='MLP ratio in Vision Transformer')
    parser.add_argument('--drop_ratio', default=0.1, type=float, help='Dropout rate in Vision Transformer')

    return parser

def load_tif_images(data_root):
    """
    Load .tif images from the data directory.
    Args:
        data_root (str): Path to the directory with .tif images.
    Returns:
        list of (str, torch.Tensor): List of tuples containing file paths and corresponding image tensors.
    """
    file_paths = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith(".tif")]
    images = []
    for file_path in file_paths:
        image = tiff.imread(file_path)
        image = torch.tensor(image).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        images.append((file_path, image))
    return images

def process_images(vit, images, output_dir, patch_size, attention_threshold=0.8):
    """
    Process images through Vision Transformer, extract causal patches, and save results.
    Args:
        vit (nn.Module): Vision Transformer model.
        images (list): List of (file_path, image_tensor) tuples.
        output_dir (str): Directory to save results.
        patch_size (int): Patch size for attention map.
        attention_threshold (float): Threshold for causal patch extraction.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file_path, image in images:
        print(f"Processing: {file_path}")

        # Pass image through Vision Transformer
        with torch.no_grad():
            reconstructed_image, attention_maps = vit(image, return_attention=True)

        # Extract causal patches
        causal_patches = extract_causal_patches_from_attention(attention_maps, patch_size, attention_threshold)

        # Save reconstructed image
        reconstructed_image_path = os.path.join(output_dir, f"reconstructed_{os.path.basename(file_path)}")
        tiff.imwrite(reconstructed_image_path, reconstructed_image[0, 0].numpy())
        print(f"Reconstructed image saved to: {reconstructed_image_path}")

        # Visualize causal patches
        causal_patch_image_path = os.path.join(output_dir, f"causal_patches_{os.path.basename(file_path).replace('.tif', '.png')}")
        visualize_causal_patches(image[0, 0].numpy(), causal_patches[0], save_path=causal_patch_image_path)
        print(f"Causal patch visualization saved to: {causal_patch_image_path}")

def main(args):
    """
    Main function for the pipeline.
    Args:
        args (Namespace): Parsed command-line arguments.
    """
    # Load .tif images
    images = load_tif_images(args.data_root)
    if not images:
        print(f"No .tif files found in {args.data_root}")
        return

    # Initialize Vision Transformer
    vit_params = {
        "img_size": args.img_size,
        "patch_size": args.patch_size,
        "in_c": 1,
        "out_chans": 1,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "drop_ratio": args.drop_ratio,
    }
    vit = VisionTransformer(**vit_params)
    vit.eval()

    # Process images and save results
    process_images(vit, images, args.output_dir, args.patch_size)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)

