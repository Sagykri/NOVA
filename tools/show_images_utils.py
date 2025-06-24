import os
from typing import Tuple
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from src.preprocessing.preprocessing_utils import rescale_intensity, fit_image_shape

# Utility functions

def show_label(path):
    path_l = path.split("/")
    return " ".join(path_l[-7:-4]) + "\n" + " ".join(path_l[-4:])

def process_tif(path):
    """
    Read and process the image.

    Parameters:
        path (str): Path to the image file.

    Returns:
        ndarray: Processed image.
    """
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    img = fit_image_shape(img, (1024, 1024))
    img = rescale_intensity(img)
    return img
    
def show_processed_tif(path, mark_tile = False):
    # read the image stack
    img = process_tif(path)
    
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)  # Lock visual size

    ax.imshow(img, cmap='gray')
    put_tiles_grid(image=img, ax=ax)
    ax.axis('off')
    ax.set_title(show_label(path), color='purple')

    if mark_tile:
        x, y = get_tile_location(mark_tile['tile_index'], mark_tile['img_shape'], mark_tile['tile_shape'])
        rect = plt.Rectangle((x, y), mark_tile['tile_shape'][0], mark_tile['tile_shape'][1], 
                             edgecolor='red', linewidth=2, fill=False)
        ax.add_patch(rect)

    # print(f"Img shape: {img.shape}")
    plt.show()

def get_tile_location(tile_index: int, img_shape:Tuple[int, int], tile_shape:Tuple[int, int]):
    """
    Compute the top-left corner (x, y) location of a tile in the original image.

    Args:
        tile_index (int): Index of the tile in the flattened tiles array.
        img_shape (Tuple[int, int]): Shape of the original image (width, height).
        tile_shape (Tuple[int, int]): Shape of a tile (tile_width, tile_height).

    Returns:
        Tuple[int, int]: (x, y) coordinates of the tile's top-left corner in the image.
    """
    img_height, img_width = img_shape  
    tile_height, tile_width = tile_shape

    n_cols = img_width // tile_width  # Number of tiles per row
    row = tile_index // n_cols  # Get row index
    col = tile_index % n_cols  # Get column index

    x = col * tile_width  # X position (left)
    y = row * tile_height  # Y position (top)
    
    return (x, y)
    
def put_tiles_grid(image, ax):
    # assumes 1000x1000 image
    # Add dashed grid lines for 64 blocks
    num_blocks = 10
    block_size = 100
    for i in range(1, num_blocks):
        # Draw horizontal dashed lines
        ax.plot([0, 1000], [i * block_size, i * block_size], linestyle='--', lw=1, alpha=0.5, color='pink')
        # Draw vertical dashed lines
        ax.plot([i * block_size, i * block_size], [0, 1000], linestyle='--', lw=1, alpha=0.5, color='pink')
    # Remove x and y axis labels
    ax.set_xticks([])
    ax.set_yticks([])

def extract_image_metadata(base_dir, FILE_EXTENSION='.tiff', KEY_BATCH='Batch'):
    """
    Traverse through a directory structure and extract metadata for images.

    Args:
        base_dir (str): The base directory containing the images.
        FILE_EXTENSION (str): Expected extension (e.g., '.tiff'), will also allow short version like '.tif'.
        KEY_BATCH (str): Name of batch prefix (case-insensitive, e.g., 'Batch').

    Returns:
        pd.DataFrame: DataFrame with extracted image metadata.
    """
    # Normalize file extensions
    ext_main = FILE_EXTENSION.lower()
    ext_alt = ext_main.replace('ff', 'f') if ext_main.endswith('ff') else ext_main + 'f'
    allowed_exts = {ext_main, ext_alt}

    data = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() not in allowed_exts:
                continue
            
            file_path = os.path.join(root, file)
            parts = root.split(os.sep)

            # Match batch in a case-insensitive way
            batch = next((p for p in parts if p.lower().startswith(KEY_BATCH.lower())), None)
            panel = next((p for p in parts if p.startswith('panel')), None)
            rep = next((p for p in parts if p.startswith('rep')), None)

            try:
                batch_idx = parts.index(batch)
                cell_line = parts[batch_idx + 1]
                condition = parts[batch_idx + 3]
            except (ValueError, IndexError, TypeError):
                cell_line = condition = None

            marker = parts[-1]

            data.append({
                'Path': file_path,
                'RootFolder': base_dir,
                'Marker': marker,
                'Condition': condition,
                'CellLine': cell_line,
                'Batch_Rep': f'{batch}/{rep}' if batch and rep else None,
                'Rep': rep,
                'Batch': batch,
                'Panel': panel
            })

    df = pd.DataFrame(data)
    df['Image_Name'] = df['Path'].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    return df