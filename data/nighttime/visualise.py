# import rasterio
# import matplotlib.pyplot as plt
# import os

# # Function to visualize a raster file
# def visualize_raster(file_name):
#     try:
#         with rasterio.open(file_name) as src:
#             data = src.read(1)  # Read the first band
#             profile = src.profile  # Metadata about the raster file

#         # Display raster information
#         print(f"Visualizing: {file_name}")
#         print("Raster Metadata:")
#         for key, value in profile.items():
#             print(f"  {key}: {value}")

#         # Plot the raster data
#         plt.figure(figsize=(10, 8))
#         plt.imshow(data, cmap='gray')
#         plt.colorbar(label="Pixel Intensity")
#         plt.title(f"Raster Visualization - {os.path.basename(file_name)}")
#         plt.xlabel("X Coordinate")
#         plt.ylabel("Y Coordinate")
#         plt.show()
#     except Exception as e:
#         print(f"Error visualizing {file_name}: {e}")

# # List of files to visualize
# raster_files = [
#     'F101992.v4b_web.avg_vis.tif',
#     'F101992.v4b_web.cf_cvg.tif',
#     'F101992.v4b_web.stable_lights.avg_vis.tif'
# ]

# # Visualize each raster file
# for raster_file in raster_files:
#     if os.path.exists(raster_file):
#         visualize_raster(raster_file)
#     else:
#         print(f"File not found: {raster_file}")
import rasterio

# Open the raster file
file_path = "/Users/Anishka/Desktop/Thesis/NuwaDynamics/data/nighttime/F101992.v4b_web.avg_vis.tif"  # Replace with your file path
with rasterio.open(file_path) as src:
    num_channels = src.count  # Number of bands
    print(f"Number of channels (bands): {num_channels}")
