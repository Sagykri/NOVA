import os
import sys
import pandas as pd 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import mplcursors
import re
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import RectangleSelector
import fnmatch

from src.figures.umap_plotting import __format_UMAP_axes, __format_UMAP_legend


def load_and_process_data(umaps_dir, path_to_umap, dfb=None):
    # Load data
    with open(umaps_dir + path_to_umap, "rb") as f:
        data = pickle.load(f)
    
    umap_embeddings = data["umap_embeddings"]
    label_data = data["label_data"]
    paths = data['paths']
    config_data = data["config_data"]
    config_plot = data["config_plot"]
    
    # Regex pattern to extract Batch, Condition, Rep, Raw Image Name, Panel, Cell Line, and Tile
    pattern = re.compile(r".*/(Batch\d+)/.*/(Untreated|stress)/.*/(rep\d+)_(r\d+c\d+f\d+-ch\d+t\d+)_(panel\w+)_(.+)_processed\.npy/(\d+)")
    
    # Parsing the paths
    parsed_data = [pattern.match(path).groups() for path in paths if pattern.match(path)]
    
    # Convert to DataFrame
    df = pd.DataFrame(parsed_data, columns=["Batch", "Condition", "Rep", "Image_Name", "Panel", "Cell_Line", "Tile"])
    df['Path'] = [path.split('.npy')[0]+'.npy' for path in paths]
    
    if dfb is not None:
        # Merge df with dfb to get Target_Sharpness_Brenner
        df = df.merge(dfb[["Batch", "Image_Name", "Target_Sharpness_Brenner"]], 
                      on=["Batch", "Image_Name"], 
                      how="left")
        df["Target_Sharpness_Brenner"] = df["Target_Sharpness_Brenner"].round()
    # Apply function and expand results into two new columns
    # df[["Sum_Channel_0", "Sum_Channel_1"]] = df.apply(compute_sums, axis=1, result_type="expand")
    df[["Row", "Column", "FOV"]] = df["Image_Name"].str.extract(r"r(\d+)c(\d+)f(\d+)")
    df[["Row", "Column", "FOV"]] = df[["Row", "Column", "FOV"]].astype(int)

    print('Validations')
    for col in ['Batch', 'Rep', 'Panel', 'Cell_Line']:
        print(col, np.unique(df[col]))
    print(f'length:  df: {len(df)}, label_data: {len(label_data)}, umap_embeddings: {len(umap_embeddings)}')
    
    return umap_embeddings, label_data, config_data, config_plot, df


def compute_sums(row):
    """Computes sum of two channels for the given row."""
    image = np.load(row["Path"])  # Load image from path
    tile = int(row["Tile"])  # Extract tile number
    site_image = image[tile]  # Select tile
    return np.sum(site_image[:, :, 0]), np.sum(site_image[:, :, 1])  # Compute sums


def set_colors_by_brenners(sharpness_values, bins=10):
    # Ensure bins is at least 2 to avoid single-value percentile issue
    bins = max(bins, 2)
    percentiles = np.percentile(sharpness_values, np.linspace(0, 100, bins + 1))
    
    def get_blue_shade(value):
        bin_idx = np.searchsorted(percentiles, value, side='right') - 1
        bin_idx = min(max(bin_idx, 0), bins - 1)  # Ensure valid bin index
        return plt.cm.Blues(0.2 + 0.8 * (bin_idx / (bins - 1)))  # Avoid very light colors
    
    return [get_blue_shade(val) for val in sharpness_values]

# Global storage for selected indices (needed for external access)
selected_indices_global = []
rect_selector = None  # Persistent RectangleSelector

def plot_umap_embeddings(
    umap_embeddings: np.ndarray,
    label_data: np.ndarray,
    config_data,
    config_plot,
    df_data=None,
    title: str = None,
    dpi: int = 500,
    figsize: tuple = (6, 5),
    cmap: str = 'tab20',
    ari_score: float = None,
    RECOLOR_BY_BRENNER=True,
    dilute: int = 1, bins: int = 10
):
    """Plots UMAP embeddings with interactive hovering for labels, with optional data dilution."""
    global rect_selector, selected_indices_global  # Keep selector alive
    if umap_embeddings.shape[0] != label_data.shape[0]:
        raise ValueError("The number of embeddings and labels must match.")
    
    # Apply dilution
    umap_embeddings = umap_embeddings[::dilute]
    label_data = label_data[::dilute]
    df = df_data.copy().iloc[::dilute].reset_index() if df_data is not None else None

    annotations_dict = {}; colors_dict = {}; scatter_mappings = {}

    if df is not None:
        image_names_dict = {idx: row.Image_Name for idx, row in df.iterrows()}
        if "Target_Sharpness_Brenner" in df.columns:
            brenner_scores_dict = {idx: row.Target_Sharpness_Brenner for idx, row in df.iterrows()}
            annotations_dict = {
                idx: f"{idx}: {image_names_dict.get(idx, 'Unknown')}\nBrenner Score: {brenner_scores_dict.get(idx, 'N/A')}"
                for idx in df.index
            }
            if RECOLOR_BY_BRENNER:
                df["Color"] = set_colors_by_brenners(df["Target_Sharpness_Brenner"].fillna(0), bins=bins)
                colors_dict = {idx: row.Color for idx, row in df.iterrows()}
        else:
            annotations_dict = {idx: f"{idx}: {image_names_dict.get(idx, 'Unknown')}" for idx in df.index}
    
    name_key, color_key = config_plot['MAPPINGS_ALIAS_KEY'], config_plot['MAPPINGS_COLOR_KEY']
    marker_size, alpha = config_plot['SIZE'], config_plot['ALPHA']
    cmap = plt.get_cmap(cmap)
    unique_groups = np.unique(label_data)
    name_color_dict = config_plot.get('COLOR_MAPPINGS', {group: cmap(i / (len(unique_groups) - 1)) for i, group in enumerate(unique_groups)})

    fig, ax = plt.subplots(figsize=figsize)
    scatter_objects = []
    legend_labels = []
    for group in unique_groups:
        group_indices = np.where(label_data == group)[0]
        if RECOLOR_BY_BRENNER and df is not None and "Target_Sharpness_Brenner" in df.columns:
            rgba_colors = [colors_dict.get(idx, "#000000") for idx in group_indices]
        else:
            base_color = name_color_dict[group][color_key]
            rgba_colors = [mcolors.to_rgba(base_color, alpha=alpha)] * len(group_indices)
        
        scatter = ax.scatter(
            umap_embeddings[group_indices, 0],
            umap_embeddings[group_indices, 1],
            s=marker_size,
            alpha=alpha,
            c=rgba_colors,
            marker='o',
        )
        scatter_objects.append(scatter)
        legend_labels.append(name_color_dict[group][name_key])
        scatter_mappings[scatter] = group_indices.tolist()
    
    # Add legend
    ax.legend(scatter_objects, legend_labels, loc="upper right", title="Groups")
    
    # Enable interactive hovering with precomputed labels
    cursor = mplcursors.cursor(scatter_objects, hover=True)
    
    @cursor.connect("add")
    def on_hover(sel):
        scatter_obj = sel.artist
        scatter_index = sel.index
        
        if scatter_obj in scatter_mappings:
            actual_index = scatter_mappings[scatter_obj][scatter_index]  # Correct mapping
            sel.annotation.set_text(annotations_dict.get(actual_index, "Unknown"))
        else:
            sel.annotation.set_text("Unknown")
            
    # **Rectangle Selection Functionality**
    def on_select(eclick, erelease):
        """Handles rectangle selection and stores selected point indices."""
        global selected_indices_global
        selected_indices_global.clear()  # Ensure it's reset each time

        if eclick.xdata is None or erelease.xdata is None:
            return  # Ignore invalid selections

        x_min, x_max = sorted([eclick.xdata, erelease.xdata])
        y_min, y_max = sorted([eclick.ydata, erelease.ydata])

        selected_indices_global.extend(
            idx for idx, x_val, y_val in all_points
            if x_min <= x_val <= x_max and y_min <= y_val <= y_max
        )

        print("Selected Indices:", selected_indices_global)  # Output in Jupyter Notebook

    # Attach Rectangle Selector (global storage prevents garbage collection)
    rect_selector = RectangleSelector(ax, on_select, interactive=True, useblit=False)
    
    ax.set_title(title if title else "UMAP Projection")
    __format_UMAP_legend(ax, marker_size)
    __format_UMAP_axes(ax, title)
    fig.tight_layout()
    plt.show()
    return selected_indices_global

def construct_target_path(df, index, df_meta):
    row = df.iloc[index]
    
    # Extract relevant information
    batch = row["Batch"]
    image_name = row["Image_Name"]
    
    temp = df_meta.loc[(df_meta.Batch == batch) & (df_meta.image_id == image_name)].Path.values
    if len(temp)>1:
        print('There is more then one file matching the batch and image name')
        return
    else:
        return temp[0]
    
from skimage.measure import shannon_entropy 
from src.preprocessing.preprocessing_utils import get_image_focus_quality 

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
    
def show_processed_tile(df, index=0):
    """
    Displays the processed tile image from the dataset.

    Parameters:
    - df: DataFrame containing 'Path', 'Tile', and 'Image_Name' columns.
    - index: Row index in df to visualize (default is 0).
    """
    path = df.Path.loc[index]
    tile = int(df.Tile.loc[index])
    image_name = df.Image_Name.loc[index]
    
    # Load the image
    image = np.load(path)
    site_image = image[tile]  # Extract the specific tile

    # Plot target and nucleus images
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    # print(site_image[:, :, 0].min(), site_image[:, :, 0].max(), site_image[:, :, 0].mean())
    ax[0].set_title(f'{image_name}/{tile} - Target: {round(get_image_focus_quality(site_image[:, :, 0]), 2)}', fontsize=11)
    ax[0].imshow(site_image[:, :, 0], cmap='gray', vmin=0, vmax=1)
    ax[0].set_axis_off()

    ax[1].set_title(f'{image_name}/{tile} - Nucleus: {round(get_image_focus_quality(site_image[:, :, 1]), 2)}', fontsize=11)
    ax[1].imshow(site_image[:, :, 1], cmap='gray', vmin=0, vmax=1)
    ax[1].set_axis_off()

    plt.show()
    return site_image


def extract_umap_data(
    base_dir,
    folder_mapping=None,
    valid_cell_lines=None,
    valid_conditions=None,
    valid_markers=None,
    batches=None
):
    """
    Extracts image metadata from folder structures using regex parsing.

    Parameters:
    - base_dir (str): Base directory where folders are located.
    - folder_mapping (dict): Dictionary mapping folder names to umap_type values.
    - valid_cell_lines (set): Set of valid cell lines.
    - valid_conditions (set): Set of valid conditions.
    - valid_markers (list): List of valid markers.
    - batches (list): List of batch numbers.

    Returns:
    - pd.DataFrame: DataFrame containing extracted metadata.
    """
    
    # Default folder mappings if not provided
    if folder_mapping is None:
        folder_mapping = {
            "SINGLE_MARKERS": 0,
            "MULTIPLE_MARKERS": 1,
            "MULTIPLEX_MARKERS": 2
        }

    # Default valid cell lines
    if valid_cell_lines is None:
        valid_cell_lines = {
            "Control-1001733", "Control-1017118", "Control-1025045", "Control-1048087",
            "C9orf72-HRE-1008566", "C9orf72-HRE-981344", "TDP--43-G348V-1057052", "TDP--43-N390D-1005373",
            "all_cell_lines"
        }

    # Default valid conditions
    if valid_conditions is None:
        valid_conditions = {"Untreated", "stress", "all_conditions"}

    # Default valid markers
    if valid_markers is None:
        valid_markers = [
            "all_markers", 'DNA_RNA_DEFECTS_MARKERS', 'PROTEOSTASIS_MARKERS', 'NEURONAL_CELL_DEATH_SENESCENCE_MARKERS',
            'SYNAPTIC_NEURONAL_FUNCTION_MARKERS', "DAPI", "Stress-initiation", "mature-Autophagosome",
            "Cytoskeleton", "Ubiquitin-levels", "UPR-IRE1a", "UPR-ATF4", "UPR-ATF6", "impaired-Autophagosome",
            "Autophagy", "Aberrant-splicing", "Parthanatos-late", "Nuclear-speckles-SC35", "Splicing-factories",
            "TDP-43", "Nuclear-speckles-SON", "DNA-damage-pH2Ax", "Parthanatos-early", "Necrosis",
            "Necroptosis-HMGB1", "Neuronal-activity", "DNA-damage-P53BP1", "Apoptosis",
            "Necroptosis-pMLKL", "Protein-degradation", "Senescence-signaling"
        ]

    # Default batch numbers
    if batches is None:
        batches = ['1', '2', '3', '4']

    # Regex pattern to extract details from parent folder names
    folder_pattern = re.compile(
        r"Batch(?P<batch>\d+)_.*?(?P<rep>all_reps|rep\d+)_"
        r"(?P<cell_line>[A-Za-z0-9-_,]+|all_cell_lines)_"
        r"(?P<condition>Untreated|stress|all_conditions)_"
        r"(?P<markers>[A-Za-z0-9-_,]+|all_markers|all_markers\(\d+\)|without_[A-Za-z0-9-]+)_colored_by_(?P<coloring>.+)"
    )

    # Prepare a list to store image data
    data = []

    # Iterate over the folders
    for folder, umap_type in folder_mapping.items():
        folder_path = os.path.join(base_dir, folder)
        
        # Ensure the folder exists
        if os.path.exists(folder_path):
            # Walk through all subdirectories
            for root, _, files in os.walk(folder_path):
                parent_folder = os.path.basename(root)  # Extract parent folder name
                match = folder_pattern.search(parent_folder)

                if match:
                    batch = match.group("batch")
                    rep = match.group("rep")

                    # Extract cell lines (handles multiple cell lines separated by '_')
                    cell_line_raw = match.group("cell_line")
                    if cell_line_raw == "all_cell_lines":
                        cell_line = "all_cell_lines"
                    else:
                        cell_lines = cell_line_raw.split("_")
                        cell_line = ",".join([cl for cl in cell_lines if cl in valid_cell_lines])
                        if not cell_line:
                            cell_line = "Unknown"

                    # Extract condition
                    condition = match.group("condition")

                    # Extract markers, including handling 'all_markers(digit)' and 'without_markerX'
                    markers_raw = match.group("markers")
                    if markers_raw.startswith("without_"):
                        markers = markers_raw.replace("without_", "without ")
                    elif markers_raw.startswith("all_markers(") and markers_raw.endswith(")"):
                        markers = markers_raw  # Keep all_markers(digit) format
                    else:
                        markers = markers_raw if markers_raw in valid_markers else "Unknown"
                        
                    # Extract coloring (everything after 'colored_by_')
                    coloring = match.group("coloring")
                else:
                    batch, rep, cell_line, condition, markers, coloring = "Unknown", "all_reps", "Unknown", "Unknown", "Unknown", "Unknown"

                # Filter only image files
                image_files = fnmatch.filter(files, "*.png") + fnmatch.filter(files, "*.jpg") + fnmatch.filter(files, "*.jpeg") + fnmatch.filter(files, "*.tiff") + fnmatch.filter(files, "*.bmp")
                
                for image in image_files:
                    folder_path_relative = os.path.relpath(root, base_dir)  # Store only the folder path
                    image_name = os.path.splitext(image)[0]  # Remove file extension
                    marker = image_name if umap_type == 0 else markers  # Use image_name as marker for SINGLE_MARKERS
                    data.append([folder_path_relative, image_name, umap_type, batch, rep, cell_line, condition, marker, coloring])

    # Create and return a DataFrame
    return pd.DataFrame(data, columns=["folder_path", "image_name", "umap_type", "batch", "rep", "cell_line", "condition", "markers", "coloring"])

import os

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
    try:
        # Filter the DataFrame to find the matching folder path
        filtered_df = df_umaps[
            (df_umaps["batch"] == str(batch)) &
            (df_umaps["umap_type"] == umap_type) &
            (df_umaps["rep"] == reps) &
            (df_umaps["coloring"] == coloring) &
            (df_umaps["markers"] == marker) &
            (df_umaps["cell_line"] == cell_line) &
            (df_umaps["condition"] == condition)
        ]

        # Extract the folder path
        folder_path_values = filtered_df["folder_path"].values

        if len(folder_path_values) == 0:
            raise ValueError("No matching folder path found for the given parameters.")

        folder_path = folder_path_values[0]  # Assuming one match

        # Extract the image name
        image_name_values = filtered_df["image_name"].values
        if len(image_name_values) == 0:
            raise ValueError("No matching image name found for the given parameters.")

        image_name = image_name_values[0]  # Assuming one match

        # Construct the full path to the pickle file
        pickle_path = os.path.join(base_dir, folder_path, f"{image_name}_plot_data.pkl")

        return pickle_path
    except:
        return -1

import numpy as np
import matplotlib.pyplot as plt

def plot_fov_histogram(df, selected_indices_global):
    """
    Plots a histogram comparing the total FOV distribution and the selected subset.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing FOV data.
        selected_indices_global (list): Indices of selected data points.
    """
    fov_counts = df["FOV"].value_counts().sort_index()
    fov_selected_counts = df.loc[selected_indices_global, "FOV"].value_counts().sort_index()

    plt.figure(figsize=(10, 4))
    plt.bar(fov_counts.index - 0.2, fov_counts.values, width=0.4, label="All Data", alpha=0.7)
    plt.bar(fov_selected_counts.index + 0.2, fov_selected_counts.values, width=0.4, label="Selected", alpha=0.7, color="orange")
    plt.xlabel("FOV"), plt.ylabel("Count"), plt.legend(), plt.show()


def plot_fov_heatmaps(df, selected_indices_global, fov_grid):
    """
    Generates heatmaps for:
    1. Total FOV distribution
    2. Selected FOV distribution
    3. Percentage of selected points vs. total
    
    Parameters:
        df (pd.DataFrame): DataFrame containing FOV data.
        selected_indices_global (list): Indices of selected data points.
        fov_grid (np.ndarray): 2D array defining FOV positions.
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

    # Plot heatmaps
    fig, axs = plt.subplots(1, 3, figsize=(8, 6))

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

    plt.show()

def filter_umap_data(umap_embeddings: np.ndarray, label_data: np.ndarray, df_image_stats: pd.DataFrame, filters: dict):
    """
    Filters umap_embeddings, label_data, and df_image_stats based on values in filters.

    Args:
        umap_embeddings (np.ndarray): 2D array of shape (N, 2) containing UMAP embeddings.
        label_data (np.ndarray): 1D array of shape (N,) containing labels for each embedding.
        df_image_stats (pd.DataFrame): DataFrame containing image statistics.
        filters (dict): Dictionary where keys are column names in df_image_stats and values are lists of allowed values.

    Returns:
        np.ndarray: Filtered umap_embeddings.
        np.ndarray: Filtered label_data.
        pd.DataFrame: Filtered df_image_stats.
    """
    # Apply all filters to df_image_stats
    mask = np.ones(len(df_image_stats), dtype=bool)  # Start with all True
    for column, values in filters.items():
        mask &= df_image_stats[column].apply(lambda x: any(str(x).startswith(prefix) for prefix in values))

    # Apply mask to all data
    filtered_umap = umap_embeddings[mask]
    filtered_labels = label_data[mask]
    filtered_df = df_image_stats.iloc[list(mask)].copy().reset_index()

    return filtered_umap, filtered_labels, filtered_df