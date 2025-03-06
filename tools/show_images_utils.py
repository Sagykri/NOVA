import os
import sys
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from matplotlib.backends.backend_pdf import PdfPages
import re
from src.preprocessing.preprocessing_utils import rescale_intensity, fit_image_shape

# Utility functions

def show_images(
    df: pd.DataFrame, 
    marker: str, 
    samples: int = 5, 
    show_DAPI: bool = True, 
    batch: int = None, 
    rep: int = None, 
    condition: str = None, 
    cell_line: str = None,
    panel: str = None,
    funova: bool = False,
    image_id: str = None
) -> None:
    """
    Display images from a DataFrame based on specific criteria and optionally show corresponding DAPI images.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing image data.
        marker (str): The marker to filter and display images for.
        samples (int, optional): Number of images to display. Defaults to 5.
        show_DAPI (bool, optional): Whether to display corresponding DAPI images. Defaults to True.
        batch (int, optional): The batch number to filter by. Defaults to None.
        rep (int, optional): The replicate number to filter by. Defaults to None.
        condition (str, optional): The condition to filter by. Defaults to None.
        cell_line (str, optional): The cell line to filter by. Defaults to None.

    Returns:
        None
    """
    df = get_specific_imgs(
        df, marker=marker, batch=batch, rep=rep, 
        condition=condition, cell_line=cell_line, panel = panel, image_id = image_id
    ).sample(frac=1, random_state=1)  # Shuffle the filtered DataFrame

    for ind, target_path in enumerate(df.Path.values[:samples]):
        print(ind + 1)
        # Display the target image
        show_processed_tif(target_path)
        print(target_path)

        if show_DAPI:
            # Display the corresponding DAPI image
            dapi_file_name = get_dapi_path(target_path, marker, funova=funova)
            print(dapi_file_name)
            show_processed_tif(dapi_file_name)
            print('--------------------------------')  

def get_dapi_path(path, marker1, marker2='DAPI', funova=False):
    """
    Modify the given path to generate a DAPI file name.

    Parameters:
        path (str): Original file path.
        marker1 (str): Marker to be replaced in the path.
        marker2 (str): Marker to replace with in the path.
        funova (bool): If True, change the digit after 'ch' in the file name to '1'.

    Returns:
        str: Modified path for the DAPI file.
    """
    # Replace marker1 with marker2
    new_path = path.replace(marker1, marker2)

    if funova:
        # Change the digit after 'ch' to '1'
        new_path = re.sub(r'ch\d+', 'ch1', new_path)

    return new_path

def show_label(path):
    path_l = path.split("/")
    return path_l[-7:]

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
    
    # show the image with grid 
    fig, ax = plt.subplots(figsize=(7,7))
    plt.imshow(img, cmap='gray')
    put_tiles_grid(image=img, ax=ax)
    plt.axis('off')
    plt.title(show_label(path), color='purple')
    print(f"Img shape: {img.shape}")

    if mark_tile!=False:
        x,y = get_tile_location(mark_tile['tile_index'], mark_tile['img_shape'], mark_tile['tile_shape'])
        plt.gca().add_patch(plt.Rectangle((x, y), mark_tile['tile_shape'][0], mark_tile['tile_shape'][1], edgecolor='red', linewidth=2, fill=False))
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

def get_specific_imgs(
    df: pd.DataFrame, 
    marker: str = None, 
    batch: int = None, 
    rep: int = None, 
    condition: str = None, 
    cell_line: str = None,
    panel: str = None,
    image_id: str = None
) -> pd.DataFrame:
    """
    Filter a DataFrame to retrieve specific rows based on the given parameters.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing image data.
        marker (str, optional): The marker to filter by. Defaults to None.
        batch (int, optional): The batch number to filter by. Defaults to None.
        rep (int, optional): The replicate number to filter by. Defaults to None.
        condition (str, optional): The condition to filter by. Defaults to None.
        cell_line (str, optional): The cell line to filter by. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame filtered based on the specified criteria.
    """
    filtered_df = df.copy()
    if marker is not None:
        filtered_df = filtered_df[filtered_df['Marker'] == marker]
    if batch is not None:
        filtered_df = filtered_df[filtered_df['Batch'] == f'Batch{str(batch)}']
    if rep is not None:
        filtered_df = filtered_df[filtered_df['Rep'] == f'rep{str(rep)}']
    if condition is not None:
        filtered_df = filtered_df[filtered_df['Condition'] == condition]
    if cell_line is not None:
        filtered_df = filtered_df[filtered_df['CellLine'] == cell_line]
    if panel is not None:
        filtered_df = filtered_df[filtered_df['Panel'] == f'panel{panel}']
    if image_id is not None:
        filtered_df = filtered_df[filtered_df['image_id'] == image_id]
    return filtered_df

def create_img_pdf_report(df, marker, condition, output_file, reps=8, batches=3, samples=3):
    """
    Create a PDF report of images where each page corresponds to a batch, with rows for each rep and images for each condition.

    Parameters:
        df (pd.DataFrame): DataFrame with image data.
        condition (str): Condition to filter by ('Untreated' or 'Stress').
        output_file (str): Path to save the PDF file.
    """
    with PdfPages(f'{marker}_{output_file}') as pdf:
        for batch in range(1, batches+1):
            fig, axes = plt.subplots(reps, samples, figsize=(12, 24))  # 8 reps, 3 images each
            fig.suptitle(f"Batch {batch} - Condition: {condition}", fontsize=16)

            for rep in range(1, reps+1):
                print(marker, batch, rep, condition)
                images = get_specific_imgs(
                    df, marker=marker, batch=batch, rep=rep, condition=condition, cell_line=KEY_WT)
                images = images.sample(n=samples, random_state=1)

                for i, path in enumerate(images['Path'][:3]):
                    img = process_tif(path)

                    ax = axes[rep - 1, i]
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    ax.set_title(f"Rep {rep} - Img {i+1}")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)
            
def extract_image_metadata(base_dir, FILE_EXTENSION='.tiff', KEY_BATCH='Batch'):
    """
    Traverse through a directory structure and extract metadata for images.

    Args:
        base_dir (str): The base directory containing the images.

    Returns:
        pd.DataFrame: A DataFrame containing metadata with columns 
                      ['Path', 'RootFolder', 'Marker', 'Condition', 'CellLine', 
                       'Batch_Rep', 'Rep', 'Batch', 'Panel'].
    """
    # Prepare a list to store extracted data
    data = []

    # Traverse through all directories and files in the base directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(FILE_EXTENSION):  # Filter only .tif files
                # Construct the full file path
                file_path = os.path.join(root, file)

                # Extract components from the file path
                parts = root.split(os.sep)
                batch = next((p for p in parts if p.startswith(KEY_BATCH)), None)
                panel = next((p for p in parts if p.startswith('panel')), None)
                condition = parts[parts.index(batch) + 3] if batch else None  # "Condition" is 3 levels after "Batch"
                cell_line = parts[parts.index(batch) + 1] if batch else None  # "CellLine" is the first level after "Batch"
                rep = next((p for p in parts if p.startswith('rep')), None)
                marker = parts[-1]  # "Marker" is the last folder

                # Store the data
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

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    df['image_id'] = df['Path'].apply(lambda path: path.split('/')[-1].split('.')[0])
    return df