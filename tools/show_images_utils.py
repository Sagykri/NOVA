import os
from typing import Tuple
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from matplotlib.backends.backend_pdf import PdfPages
import re
import numpy as np
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
    image_id: str = None,
    change_ch: bool = False,
    dapi_name: str = 'DAPI'
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
        change_ch (bool, optional): If True, change the channel digit in the DAPI file name. Defaults to False.
        dapi_name (str, optional): The name of the DAPI channel to look for. Defaults to 'DAPI'.
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
            dapi_file_name = get_dapi_path(target_path, marker, dapi_name, change_ch = change_ch)
            print(dapi_file_name)
            show_processed_tif(dapi_file_name)
            print('--------------------------------')  

def get_dapi_path(path, marker1, marker2='DAPI', change_ch=False):
    """
    Modify the given path to generate a DAPI file name.

    Parameters:
        path (str): Original file path.
        marker1 (str): Marker to be replaced in the path.
        marker2 (str): Marker to replace with in the path.
        change_ch (bool, optional): If True, change the channel digit in the DAPI file name. Defaults to False.

    Returns:
        str: Modified path for the DAPI file.
    """
    dir_path, file_name = os.path.split(path)

    # Replace marker in file name only
    dir_path = dir_path.replace(marker1, marker2)

    if change_ch:
        # Look for any DAPI file in dir_path
        dapi_ch = None
        for f in os.listdir(dir_path):
            match = re.search(r'ch(\d+)', f)
            if match:
                dapi_ch = match.group(1)
                break

        if not dapi_ch:
            raise ValueError(f"No DAPI channel found in {dir_path}")

        # Replace ch digit in the name
        file_name = re.sub(r'ch\d+', f'ch{dapi_ch}', file_name)

    return os.path.join(dir_path, file_name)

def show_label(path):
    path_l = path.split("/")
    return " ".join(path_l[-7:-4]) + "\n" + " ".join(path_l[-4:])

def process_tif(path, expected_shape=(1024, 1024)):
    """
    Read and process the image.

    Parameters:
        path (str): Path to the image file.

    Returns:
        ndarray: Processed image.
    """
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    img = fit_image_shape(img, expected_shape)
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
    
def put_tiles_grid(image, ax, block_size = 128):
    # Add dashed grid lines for num_blocks blocks
    num_blocks = image.shape[0] // block_size  # Assuming square blocks
    
    for i in range(1, num_blocks):
        # Horizontal lines
        ax.plot([0, image.shape[1]], [i * block_size, i * block_size],
                linestyle='--', lw=1, alpha=0.5, color='pink')
        # Vertical lines
        ax.plot([i * block_size, i * block_size], [0, image.shape[0]],
                linestyle='--', lw=1, alpha=0.5, color='pink')
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
        filtered_df = filtered_df[filtered_df['CellLine'].str.contains(cell_line, na=False)]
    if panel is not None:
        filtered_df = filtered_df[filtered_df['Panel'] == f'panel{panel}']
    if image_id is not None:
        filtered_df = filtered_df[filtered_df['Image_Name'] == image_id]
    return filtered_df

def create_img_pdf_report(df, marker, condition, output_file, reps=8, batches=3, samples=3, WT_KEY = 'Control',
                          path_key = 'Path', folder_to_save = ''):
    """
    Create a PDF report of images where each page corresponds to a batch, with rows for each rep and images for each condition.

    Parameters:
        df (pd.DataFrame): DataFrame with image data.
        condition (str): Condition to filter by ('Untreated' or 'Stress').
        output_file (str): Path to save the PDF file.
    """
    with PdfPages(f'{folder_to_save}/{marker}_{output_file}') as pdf:
        for batch in range(1, batches+1):
            fig, axes = plt.subplots(reps, samples, figsize=(8, 8)) 
            fig.suptitle(f"Batch {batch} - Condition: {condition}", fontsize=16)

            for rep in range(1, reps+1):
                print(marker, batch, rep, condition)
                images = get_specific_imgs(
                    df, marker=marker, batch=batch, rep=rep, condition=condition, cell_line=WT_KEY)
                images = images.sample(n=samples, random_state=1)

                for i, path in enumerate(images[path_key]):
                    img = process_tif(path)

                    ax = axes[rep - 1, i]
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    ax.set_title(f"Rep {rep} - Img {images.iloc[i]['Image_Name']} ")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)
            
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

def extract_processed_image_metadata(base_dir, FILE_EXTENSION='.npy', KEY_BATCH='Batch'):
    """
    Traverse through a directory structure and extract metadata for processed images.

    Args:
        base_dir (str): The base directory containing the processed images.

    Returns:
        pd.DataFrame: A DataFrame containing metadata with columns 
                      ['Path', 'RootFolder', 'Marker', 'Condition', 'CellLine', 
                       'Batch_Rep', 'Rep', 'Batch', 'Panel', 'Image_Name'].
    """
    data = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(FILE_EXTENSION):
                file_path = os.path.join(root, file)

                parts = root.split(os.sep)
                batch = next((p for p in parts if p.startswith(KEY_BATCH)), None)
                cell_line = parts[parts.index(batch) + 1] if batch else None
                condition = parts[parts.index(batch) + 2] if batch else None
                marker = parts[parts.index(batch) + 3] if batch else None

                rep = None
                panel = None

                if file.startswith("rep"):
                    rep_parts = file.split('_')
                    rep = rep_parts[0]  # e.g., "rep2"
                    panel_part = [p for p in rep_parts if p.startswith('panel')]
                    if panel_part:
                        panel = panel_part[0]  # e.g., "panelK"

                # Extract Image_Name using regex
                match = re.search(r'(r\d{2}c\d{2}f\d+[-]ch\d+t\d+)', file)
                Image_Name = match.group(1) if match else None

                data.append({
                    'Path': file_path,
                    'RootFolder': base_dir,
                    'Marker': marker,
                    'Condition': condition,
                    'CellLine': cell_line,
                    'Batch_Rep': f'{batch}/{rep}' if batch and rep else None,
                    'Rep': rep,
                    'Batch': batch,
                    'Panel': panel,
                    'Image_Name': Image_Name
                })

    return pd.DataFrame(data)

def get_combined_metadata(
    raw_base_dir: str,
    processed_base_dir: str,
    raw_file_extension: str = '.tiff',
    key_batch: str = 'Batch',
    processed_FILE_EXTENSION: str = '.npy') -> pd.DataFrame:
    """
    Combine metadata from raw and processed images into a single DataFrame.
    Args:
        raw_base_dir (str): The base directory containing the raw images.
        raw_file_extension (str): Expected extension for raw images (e.g., '.tiff').
        key_batch (str): Name of batch prefix (case-insensitive, e.g., 'Batch').
        processed_base_dir (str, optional): The base directory containing the processed images. Defaults to None.
        processed_FILE_EXTENSION (str): Expected extension for processed images (e.g., '.npy'). Defaults to '.npy'.
    Returns:
        pd.DataFrame: A DataFrame containing combined metadata with columns
                      ['Path_raw', 'RootFolder_raw', 'Path_processed', 'RootFolder_processed',
                       'Marker', 'Condition', 'CellLine', 'Batch_Rep', 'Rep', 'Batch', 'Panel', 'Image_Name'].
    """
    df_raw = extract_image_metadata(raw_base_dir, raw_file_extension, key_batch)
    df_processed = extract_processed_image_metadata(processed_base_dir, processed_FILE_EXTENSION, key_batch)
    # Rename columns in df_raw
    df_raw = df_raw.rename(columns={
        'Path': 'Path_raw',
        'RootFolder': 'RootFolder_raw',
    })

    # Rename columns in df_processed
    df_processed = df_processed.rename(columns={
        'Path': 'Path_processed',
        'RootFolder': 'RootFolder_processed',
    })

    # Define columns to match on (common metadata columns)
    match_cols = ['Marker', 'Condition', 'CellLine', 'Batch_Rep', 'Rep', 'Batch', 'Panel', 'Image_Name']

    # Drop duplicates to avoid exploding the merge if raw has multiple matches
    df_raw_unique = df_raw.drop_duplicates(subset=match_cols)

    # Merge df_processed with raw to get Path_raw, RootFolder_raw
    df_combined = df_processed.merge(
        df_raw_unique[match_cols + ['Path_raw', 'RootFolder_raw']],
        on=match_cols,
        how='left'
    )
    for col in ['Marker', 'Condition', 'CellLine', 'Rep', 'Batch', 'Panel']:
        print(col, 'values are:', np.unique(df_combined[col]))
    return df_combined


def plot_overlay(df: pd.DataFrame, 
    marker: str, 
    samples: int = 5, 
    show_DAPI: bool = True, 
    batch: int = None, 
    rep: int = None, 
    condition: str = None, 
    cell_line: str = None,
    panel: str = None,
    image_id: str = None,
    change_ch: bool = False,
    dapi_name: str = 'DAPI'
    ) -> None:
    """
    Plot an overlay of marker and its corresponding DAPI image.

    Args:
        df (pd.DataFrame): The input DataFrame containing image data.
        marker (str): The marker to filter and display images for.
        samples (int, optional): Number of images to display. Defaults to 5.
        batch (int, optional): The batch number to filter by. Defaults to None.
        rep (int, optional): The replicate number to filter by. Defaults to None.
        condition (str, optional): The condition to filter by. Defaults to None.
        cell_line (str, optional): The cell line to filter by. Defaults to None.
        change_ch (bool, optional): If True, change the channel digit in the DAPI file name. Defaults to False.
        dapi_name (str, optional): The name of the DAPI channel to look for. Defaults to 'DAPI'.
    Returns:
        None
    """
    df = get_specific_imgs(
        df, marker=marker, batch=batch, rep=rep, 
        condition=condition, cell_line=cell_line, panel = panel, image_id = image_id
    ).sample(frac=1, random_state=1)  # Shuffle the filtered DataFrame
    print(len(df), "images found:", marker)

    for index, target_path in enumerate(df.Path.values[:samples]):
        print(index + 1)
        marker_path = df.Path.iloc[index]
        marker_name = df.Marker.iloc[index]
        
        dapi_path = get_dapi_path(marker_path, marker_name, dapi_name, change_ch==change_ch)
        
        # Load images
        marker_img = process_tif(marker_path)
        dapi_img = process_tif(dapi_path)
        
        # Make RGB overlay
        overlay = np.zeros((*marker_img.shape, 3), dtype=np.float32)
        overlay[..., 0] = rescale_intensity(marker_img)  # Red: marker
        overlay[..., 1] = rescale_intensity(dapi_img)    # Green: DAPI

        # Plot
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.imshow(overlay)
        ax.axis('off')
        ax.set_title(f"Overlay: {marker_name} + DAPI\n{os.path.basename(marker_path)}", fontsize=10)
        plt.show()
        print('--------------------------------')  