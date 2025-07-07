import os
import pandas as pd 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import mplcursors
import re
import matplotlib.colors as mcolors
from matplotlib.widgets import RectangleSelector
import random
from skimage.measure import shannon_entropy 
import psutil
import tifffile as tiff  # for TIFF writing
from skimage.exposure import rescale_intensity

from src.preprocessing.preprocessing_utils import get_image_focus_quality 
from src.figures.umap_plotting import __format_UMAP_axes, __format_UMAP_legend
from tools.show_images_utils import process_tif


def load_and_process_data(umaps_dir, path_to_umap, df_brenner=None, print_validations=False):
    # Load data
    with open(umaps_dir + path_to_umap, "rb") as f:
        data = pickle.load(f)
    
    umap_embeddings = data["umap_embeddings"]
    label_data = data["label_data"]
    paths = data['paths']
    config_data = data["config_data"]
    config_plot = data["config_plot"]
    
    # Regex pattern to extract Batch, Condition, Rep, Raw Image Name, Panel, Cell Line, and Tile
    pattern = re.compile(
    r".*/([Bb]atch\d+)/([^/]+)/([^/]+)/([^/]+)/"                # Batch / Cell_Line / Condition / Marker
        r"(rep\d+)_([^/]*_panel(\w+)_.*)_processed\.npy/(\d+)"      # Rep / Image_Name, Panel, Tile
    )

    colnames = ["Batch", "CellLine", "Condition", "Marker", "Rep", "Image_Name", "Panel", "Tile"]

    # Determine case by path dimensionality
    if paths.ndim == 1:
        parsed_data = [pattern.match(path).groups() for path in paths if pattern.match(path)]
        df_umap_tiles = pd.DataFrame(parsed_data, columns=colnames)
        df_umap_tiles['Path'] = paths

    elif paths.ndim == 2:
        df_umap_tiles = pd.DataFrame({"Path_List": paths.tolist()})
        random_path_data = []

        for path_list in df_umap_tiles["Path_List"]:
            random_path = random.choice(path_list)
            match = pattern.match(random_path)
            groups = list(match.groups()) if match else [None] * len(colnames)
            groups.append(random_path)
            random_path_data.append(groups)

        df_info = pd.DataFrame(random_path_data, columns=colnames + ['Path'])

        # Merge
        df_umap_tiles = pd.concat([df_umap_tiles, df_info], axis=1)

    else:
        raise ValueError(f"Unsupported path shape: {paths.shape}")
    
    df_umap_tiles['Path'] = [path.split('.npy')[0]+'.npy' for path in df_umap_tiles['Path']]
    df_umap_tiles["Image_Name"] = df_umap_tiles["Image_Name"].str.extract(r"^(.*?)_panel")
    
    df_umap_tiles = fix_deltaNLS_metadata(df_umap_tiles)

    if (df_brenner is not None) and (paths.ndim == 1):      
        # Merge df with df_brenner to get Target_Sharpness_Brenner
        df_umap_tiles = df_umap_tiles.merge(
            df_brenner[["Batch", "Rep", "Image_Name", "Condition", "Marker", "CellLine", "Panel", "Target_Sharpness_Brenner"]],
            on=["Batch", "Rep", "Image_Name", "Condition", "Marker", "CellLine", "Panel"],
            how="left"
        )
        df_umap_tiles["Target_Sharpness_Brenner"] = df_umap_tiles["Target_Sharpness_Brenner"].round()
    
    try:
        df_umap_tiles[["Row", "Column", "FOV"]] = df_umap_tiles["Image_Name"].str.extract(r"r(\d+)c(\d+)f(\d+)")
        df_umap_tiles[["Row", "Column", "FOV"]] = df_umap_tiles[["Row", "Column", "FOV"]].astype(int)
    except:
        try:
            df_umap_tiles["FOV"] = df_umap_tiles["Image_Name"].str.extract(r"s(\d+)").astype(int)
            df_umap_tiles["FOV"] = df_umap_tiles["FOV"].apply(lambda x: x % 100)
            df_umap_tiles[["Row", "Column"]] = np.nan
        except:
            print('No FOV information in image name')
    df_umap_tiles["Cell_Line_Condition"] = df_umap_tiles["CellLine"] + "__" + df_umap_tiles["Condition"]

    if print_validations:
        print('Validations')
        print(f'length:  df_umap_tiles: {len(df_umap_tiles)}, label_data: {len(label_data)}, umap_embeddings: {len(umap_embeddings)}')
        for col in ['Batch', 'Rep', 'Panel', 'Condition', 'CellLine', 'Marker']:
            print(col, np.unique(df_umap_tiles[col]))
    
    return umap_embeddings, label_data, config_data, config_plot, df_umap_tiles

def set_colors_by_brenners(sharpness_values, bins=10):
    # Ensure bins is at least 2 to avoid single-value percentile issue
    bins = max(bins, 2)
    percentiles = np.percentile(sharpness_values, np.linspace(0, 100, bins + 1))
    
    def get_blue_shade(value):
        bin_idx = np.searchsorted(percentiles, value, side='right') - 1
        bin_idx = min(max(bin_idx, 0), bins - 1)  # Ensure valid bin index
        return plt.cm.Blues(0.2 + 0.8 * (bin_idx / (bins - 1)))  # Avoid very light colors
    
    return [get_blue_shade(val) for val in sharpness_values], percentiles, plt.cm.Blues

def construct_target_path(df, index, df_site_meta):
    row = df.iloc[index]
    
    # Extract relevant information
    batch = row["Batch"]
    image_name = row["Image_Name"]
    condition = row['Condition']
    marker = row['Marker']
    cell_line = row['CellLine']
    rep = row['Rep']
    panel = row['Panel']
    
    temp = df_site_meta.loc[(df_site_meta.Batch == batch) & (df_site_meta.Image_Name == image_name) &
                       (df_site_meta.Condition == condition) & (df_site_meta.Marker == marker) &
                       (df_site_meta.CellLine == cell_line) & (df_site_meta.Rep == rep) & (df_site_meta.Panel == f'panel{panel}')].Path.values
    if len(temp)>1:
        print('❌ There is more then one file matching the batch and image name.')
        return -1
    elif len(temp)==0:
        print('❌ No matching images found, try adjusting the images dir.')
        return -1
    else:
        return temp[0]

def compute_entropy(image: np.ndarray) -> float:
    """
    Compute the entropy of an image.
    
    Args:
        image (np.ndarray): Input grayscale or single-channel image.
    
    Returns:
        float: Shannon entropy of the image.
    """
    return shannon_entropy(image)

def compute_snr(image: np.ndarray) -> float:
    """
    Compute the Signal-to-Noise Ratio (SNR) of an image.
    
    Args:
        image (np.ndarray): Input grayscale or single-channel image.
    
    Returns:
        float: SNR value.
    """
    signal = np.mean(image)
    noise = np.std(image)
    return 20 * np.log10(signal / noise) if noise > 0 else float("inf")

def process_tile(df, index):
    """
    Load and process a tile from the given DataFrame row index.
    
    Returns:
        marker: normalized marker channel (2D array)
        nucleus: normalized nucleus channel (2D array)
        overlay: RGB overlay image (H, W, 3)
    """
    path = df.Path.loc[index]
    tile = int(df.Tile.loc[index])

    image = np.load(path)
    site_image = image[tile]

    marker = np.clip(site_image[:, :, 0], 0, 1)
    nucleus = np.clip(site_image[:, :, 1], 0, 1)

    overlay = np.zeros((*marker.shape, 3), dtype=np.float32)
    overlay[..., 0] = marker
    overlay[..., 1] = nucleus

    return marker, nucleus, overlay
    
def show_processed_tile(df, index=0):
    """
    Processes and displays the tile image from the dataset.

    This function extracts the marker and nucleus channels from the specified tile,
    creates an RGB overlay, and displays all three views side by side.

    Parameters:
    - df: DataFrame containing 'Path', 'Tile', and 'Image_Name' columns.
    - index: Row index in df to process and display (default is 0).

    Plots:
    - marker: 2D image of the normalized marker channel
    - nucleus: 2D image of the normalized nucleus channel
    - overlay: RGB overlay image as a (H, W, 3) array
    """
    marker, nucleus, overlay = process_tile(df, index)
    image_name = df.Image_Name.loc[index]
    tile = df.Tile.loc[index]

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))

    ax[0].set_title(f'{image_name}/{tile} - Marker', fontsize=11)
    ax[0].imshow(marker, cmap='gray', vmin=0, vmax=1)
    ax[0].axis('off')

    ax[1].set_title(f'{image_name}/{tile} - Nucleus', fontsize=11)
    ax[1].imshow(nucleus, cmap='gray', vmin=0, vmax=1)
    ax[1].axis('off')

    ax[2].set_title(f'{image_name}/{tile} - Overlay', fontsize=11)
    ax[2].imshow(overlay)
    ax[2].axis('off')

    plt.show()

def save_processed_tile(df, index, folder_path, high_resolution=True):
    """
    Processes and saves a tile image (marker, nucleus, and overlay) as TIFF files.

    Parameters:
    - df: DataFrame containing 'Path', 'Tile', and 'Image_Name' columns.
    - index: Row index in df to save.
    - folder_path: Destination folder for output files.
    - high_resolution: Whether to save in 16-bit with 600 DPI (default: True).
    """
    try:
        marker, nucleus, overlay = process_tile(df, index)
        full_path = df.Path.loc[index]
        batch = df.Batch.loc[index]
        tile = df.Tile.loc[index]

        # Get relative path inside batch
        rel_path = full_path.split(batch, 1)[-1].lstrip(os.sep).replace('.npy', '')

        # Replace / or \ with .
        rel_path = rel_path.replace('/', '.').replace('\\', '.')

        # Final base name: batch.rel_path.tile
        base = f"{batch}.{rel_path}.{tile}"

        factor = 65535 if high_resolution else 255
        dtype = np.uint16 if high_resolution else np.uint8
        dpi = 600 if high_resolution else 300
        meta_data = {'dpi': (dpi, dpi)}

        tiff.imwrite(os.path.join(folder_path, f"{base}_marker.tiff"),
                     (marker * factor).astype(dtype), metadata=meta_data)
        tiff.imwrite(os.path.join(folder_path, f"{base}_nucleus.tiff"),
                     (nucleus * factor).astype(dtype), metadata=meta_data)
        tiff.imwrite(os.path.join(folder_path, f"{base}_overlay.tiff"),
                     (overlay * factor).astype(dtype), metadata=meta_data)

    except Exception as e:
        print(f"❌ Failed saving tile at index {index}: {e}")

def improve_brightness(img, contrast_factor, brightness_factor):
    in_range = (0.1,0.8)
    out_range = (0,1)
    img_normalized = rescale_intensity(img, in_range, out_range)
    img_normalized += brightness_factor
    # Clip values to keep them within the 0-1 range
    img_normalized = img_normalized.clip(0, 1)
    return img_normalized

def save_processed_tile_new(
    df,
    tile_index,
    colormap,
    tile_pixel_size_in_um,
    tile_scalebar_length_in_um=5,
    contrast_factor=1,
    brightness_factor=0.1
):
    """
    Processes and saves marker, nucleus, and overlay tile images.
    """
    # Get marker and nucleus channels + empty overlay
    marker, nucleus, _ = process_tile(df, tile_index)

    # Adjust channels
    marker_adj = improve_brightness(marker, contrast_factor, brightness_factor)
    nucleus_adj = improve_brightness(nucleus, contrast_factor, brightness_factor)

    # Build overlay from adjusted channels
    overlay = np.zeros((*marker.shape, 3), dtype=np.float32)
    overlay[..., 0] = marker_adj  # Red
    overlay[..., 1] = nucleus_adj  # Green

    # Calculate scale bar in pixels
    scalebar_pixels = tile_scalebar_length_in_um / tile_pixel_size_in_um

    # Generic saver
    def save_image(img, cmap, filename):
        fig = plt.figure(figsize=(100/127, 100/127), dpi=127) #100/dpi,100/dpi
        if cmap:
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(img)
        plt.axis('off')
        plt.margins = (0, 0)
        plt.hlines(y=90, xmin=85 - scalebar_pixels, xmax=85, color='white', linewidth=2)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.savefig(filename, dpi=127, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # Prepare filename base
    full_path = df.Path.loc[tile_index]
    batch = df.Batch.loc[tile_index]
    tile = df.Tile.loc[tile_index]

    # Get relative path inside batch
    rel_path = full_path.split(batch, 1)[-1].lstrip(os.sep).replace('.npy', '')

    # Replace / or \ with .
    rel_path = rel_path.replace('/', '.').replace('\\', '.')

    # Final base name: batch.rel_path.tile
    filename_base = f"{batch}.{rel_path}.{tile}"

    # Save images
    save_image(marker_adj, colormap, f"{filename_base}_marker.png")
    save_image(nucleus_adj, colormap, f"{filename_base}_nucleus.png")
    save_image(overlay, None, f"{filename_base}_overlay.png")

def save_processed_tif_image(path, index, folder_path, high_resolution=True):
    """
    Processes and saves a TIFF image from the given path.
    Parameters:
    - path (str): Path to the image file.
    - index (int): Index in df for naming the output file.
    - folder_path (str): Destination folder for the output file.
    - high_resolution (bool): Whether to save in 16-bit with 600 DPI (default: True).
    """
    try:
        img = process_tif(path)

        factor = 65535 if high_resolution else 255
        dtype = np.uint16 if high_resolution else np.uint8
        dpi = 600 if high_resolution else 300
        meta = {'dpi': (dpi, dpi)}

        output_path = os.path.join(folder_path, f"{index}_{os.path.basename(path).replace('.tif', '')}.tiff")
        tiff.imwrite(output_path, (img * factor).astype(dtype), metadata=meta)
    except Exception as e:
        print(f"❌ Failed saving processed TIFF from {path}: {e}")

def extract_umap_data(base_dir):
    """
    Extracts umap metadata from folder structures using regex parsing.

    Parameters:
    - base_dir (str): Base directory where folders are located.

    Returns:
    - pd.DataFrame: DataFrame containing extracted metadata.
    """

    # Regex to extract metadata from folder names
    folder_pattern = re.compile(
        r"(?i)batch(?P<batch>\d+)_.*?(?P<rep>all_reps|rep\d+)_"
        r"(?P<cell_line>.+?)_"  # non-greedy
        r"(?P<condition>Untreated|stress|all_conditions)_"
        r"(?P<markers>.+?)_(?:colored_by|coloring)_(?P<coloring>.+)"
    )

    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
    data = []

    for umap_type in ("SINGLE_MARKERS", "MULTIPLE_MARKERS", "MULTIPLEX_MARKERS"):
        folder_path = os.path.join(base_dir, umap_type)
        if not os.path.exists(folder_path):
            continue

        for root, _, files in os.walk(folder_path):
            parent_folder = os.path.basename(root)
            match = folder_pattern.search(parent_folder)

            if match:
                batch = match.group("batch")
                rep = match.group("rep")
                cell_line_raw = match.group("cell_line")
                cell_line = cell_line_raw.replace("_", ",") if cell_line_raw != "all_cell_lines" else "all_cell_lines"
                condition = match.group("condition")
                markers = match.group("markers")
                coloring = match.group("coloring")
            else:
                batch, rep, cell_line, condition, markers, coloring = "Unknown", "all_reps", "Unknown", "Unknown", "Unknown", "Unknown"

            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    folder_path_relative = os.path.relpath(root, base_dir)
                    image_name = os.path.splitext(file)[0]
                    marker = image_name if umap_type == "SINGLE_MARKERS" else markers
                    data.append([
                        folder_path_relative, image_name, umap_type,
                        batch, rep, cell_line, condition, marker, coloring
                    ])

    df = pd.DataFrame(data, columns=[
        "Path", "Image_Name", "Umap_Type", "Batch", "Rep",
        "CellLine", "Condition", "Marker", "Coloring"
    ])
    
    df = fix_deltaNLS_metadata(df)

    return df

def fix_deltaNLS_metadata(df):
    """
    Fixes the deltaNLS marker names in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing UMAP data.
    """
    ## Fix for deltaNLS 
    df["Marker"] = df["Marker"].replace({
        "TDP43B": "TDP43"
    })
    return df


def get_umap_pickle_path(df_umaps, batch, umap_type, reps, coloring, marker, cell_line, condition='all_conditions', base_dir="/"):
    """
    Constructs the path to the pickle file based on the provided parameters.

    Parameters:
    - df_umaps (pd.DataFrame): DataFrame containing UMAP data.
    - batch (str): Batch number (as string).
    - umap_type (int): UMAP type (0, 1, or 2).
    - reps (str): Replicate group (e.g., "all_reps").
    - coloring (str): Coloring type (e.g., "CONDITIONS").
    - marker (str): Marker name (e.g., "Nuclear-speckles-SON").
    - cell_line (str): Cell line (e.g., "all_cell_lines").
    - condition (str): Condition (e.g., "all_conditions").
    - base_dir (str): Base directory where files are stored (default: "/").

    Returns:
    - str: Full path to the pickle file.
    """
    # print(f"Batch: {batch}, UMAP Type: {umap_type}, Reps: {reps}, Coloring: {coloring}, Marker: {marker}, Cell Line: {cell_line}, Condition: {condition}")

    try:
        # Filter the DataFrame to find the matching folder path
        filtered_df = df_umaps[
            (df_umaps["Batch"] == str(batch)) &
            (df_umaps["Umap_Type"] == umap_type) &
            (df_umaps["Rep"] == reps) &
            (df_umaps["Coloring"] == coloring) &
            (df_umaps["Marker"] == marker) &
            (df_umaps["CellLine"] == cell_line) &
            (df_umaps["Condition"] == condition)
        ]

        # Extract the folder path
        folder_path_values = filtered_df["Path"].values

        # Extract the image name
        image_name_values = filtered_df["Image_Name"].values

        if len(folder_path_values) == 0:
            raise ValueError("No matching folder path found for the given parameters.")

        if len(folder_path_values) > 1:
            print('More than one pickle file matched, taking the first')

        folder_path = folder_path_values[0]  # Assuming one match
        image_name = image_name_values[0]  # Assuming one match

        # Construct the full path to the pickle file
        pickle_path = os.path.join(base_dir, folder_path, f"{image_name}_plot_data.pkl")

        return pickle_path
    except:
        return -1

def plot_fov_histogram(df, selected_indices_global, return_fig=False):
    """
    Plots a histogram comparing the total FOV distribution and the selected subset.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing FOV data.
        selected_indices_global (list): Indices of selected data points.
        return_fig (bool): Whether to return the matplotlib figure object (Plot when False).
    """
    fov_counts = df["FOV"].value_counts().sort_index()
    fov_selected_counts = df.loc[selected_indices_global, "FOV"].value_counts().sort_index()

    # Prevent automatic display
    plt.ioff()

    fig = plt.figure(figsize=(10, 4))
    plt.bar(fov_counts.index - 0.2, fov_counts.values, width=0.4, label="All Data", alpha=0.7)
    plt.bar(fov_selected_counts.index + 0.2, fov_selected_counts.values, width=0.4, label="Selected", alpha=0.7, color="orange")
    plt.xlabel("FOV")
    plt.ylabel("Count")
    plt.title("FOV Distribution Histogram: All Data vs Selected")
    plt.legend()
    plt.tight_layout()

    if return_fig:
        plt.close(fig)  # Prevent display in Jupyter
        return fig
    else:
        plt.ion()  # Re-enable automatic display
        plt.show()


def plot_fov_heatmaps(df, selected_indices_global, fov_grid, return_fig=False):
    """
    Generates heatmaps based on the scan fov grid for:
    1. Total FOV distribution
    2. Selected FOV distribution
    3. Percentage of selected points vs. total
    
    Parameters:
        df (pd.DataFrame): DataFrame containing FOV data.
        selected_indices_global (list): Indices of selected data points.
        fov_grid (np.ndarray): 2D array defining FOV positions.
        return_fig (bool): Whether to return the matplotlib figure object (Plot when False).
    """
    # Create heatmaps initialized with NaNs
    heatmap_all = np.full(fov_grid.shape, np.nan)
    heatmap_selected = np.full(fov_grid.shape, np.nan)
    heatmap_percentage = np.full(fov_grid.shape, np.nan)

    # Compute FOV counts
    fov_counts = df["FOV"].value_counts().sort_index()
    fov_selected_counts = df.loc[selected_indices_global, "FOV"].value_counts().sort_index()

    # Fill heatmaps
    for i in range(fov_grid.shape[0]):
        for j in range(len(fov_grid[i])):
            fov = fov_grid[i][j]
            if fov != -1:
                all_count = fov_counts.get(fov, 0)
                selected_count = fov_selected_counts.get(fov, 0)

                if all_count > 0:
                    heatmap_all[i, j] = all_count
                    heatmap_selected[i, j] = selected_count
                    heatmap_percentage[i, j] = (selected_count / all_count) * 100  # Compute percentage

    # Prevent automatic display
    plt.ioff()
    # Plot heatmaps
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))

    # Heatmap for all data
    im1 = axs[0].imshow(heatmap_all, cmap="Blues", aspect="auto", interpolation="nearest")
    axs[0].set_title("FOV Distribution - All Data")
    fig.colorbar(im1, ax=axs[0], label="Count")
    axs[0].set_xticks([]), axs[0].set_yticks([])

    # Heatmap for selected data
    im2 = axs[1].imshow(heatmap_selected, cmap="Oranges", aspect="auto", interpolation="nearest")
    axs[1].set_title("FOV Distribution - Selected Data")
    fig.colorbar(im2, ax=axs[1], label="Count")
    axs[1].set_xticks([]), axs[1].set_yticks([])

    # Heatmap for percentage of selected data relative to all
    im3 = axs[2].imshow(heatmap_percentage, cmap="Reds", aspect="auto", interpolation="nearest")
    axs[2].set_title("FOV Selection % (Selected / All)")
    fig.colorbar(im3, ax=axs[2], label="Percentage")
    axs[2].set_xticks([]), axs[2].set_yticks([])

    fig.suptitle("FOV Heatmaps", fontsize=14)

    if return_fig:
        plt.close(fig)  # Prevent display in Jupyter
        return fig
    else:
        plt.ion()
        plt.show()

def get_lsf_mem_limit_gb():
    res_req = os.environ.get("LSB_EFFECTIVE_RSRCREQ") or ""
    match = re.search(r"mem=(\d+(?:\.\d+)?)", res_req)
    if match:
        return round(float(match.group(1)) / 1024, 2)  # Convert MB to GB
    return None

def get_ram_usage_gb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    mem_gb = mem_bytes / (1024 ** 3)
    return mem_gb

def check_memory_status(print_status=False, lim_percent=0.9):
    """
    Check the memory usage of the current process and compare it to the LSF memory limit.
    If the usage exceeds 90% of the limit, a warning is printed.
    """
    limit = get_lsf_mem_limit_gb()
    usage = get_ram_usage_gb()

    if limit is not None:
        if print_status:
            print(f"Memory Usage is: {usage:.2f} GB, LSF Memory Limit: {limit:.2f} GB")
        if usage > lim_percent * limit:
            print(f"🚨 Memory usage is above {100*lim_percent}% of allocated LSF limit ({usage:.2f} GB out of {limit:.2f} GB)! \n \
                  Please restart kernel or allocate more memory.")