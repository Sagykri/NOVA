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

from src.preprocessing.preprocessing_utils import get_image_focus_quality 
from src.figures.umap_plotting import __format_UMAP_axes, __format_UMAP_legend


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
    site_image = image[tile]
    marker = site_image[:, :, 0]
    nucleus = site_image[:, :, 1]

    # Normalize
    marker = np.clip(marker, 0, 1)
    nucleus = np.clip(nucleus, 0, 1)

    # Create RGB overlay: Red for marker, Green for nucleus
    overlay = np.zeros((*marker.shape, 3))
    overlay[..., 0] = marker      # Red channel = marker
    overlay[..., 1] = nucleus     # Green channel = nucleus
    # Blue remains 0

    # Plot target, nucleus, and overlay
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))

    ax[0].set_title(f'{image_name}/{tile} - Marker', fontsize=11)
    ax[0].imshow(marker, cmap='gray', vmin=0, vmax=1)
    ax[0].set_axis_off()

    ax[1].set_title(f'{image_name}/{tile} - Nucleus', fontsize=11)
    ax[1].imshow(nucleus, cmap='gray', vmin=0, vmax=1)
    ax[1].set_axis_off()

    ax[2].set_title(f'{image_name}/{tile} - Overlay', fontsize=11)
    ax[2].imshow(overlay)
    ax[2].set_axis_off()

    plt.show()

    return site_image

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

        if len(folder_path_values) == 0:
            raise ValueError("No matching folder path found for the given parameters.")

        if len(folder_path_values) > 1:
            print('More than one pickle file matched, taking the first')

        folder_path = folder_path_values[0]  # Assuming one match

        # Extract the image name
        image_name_values = filtered_df["Image_Name"].values
        if len(image_name_values) == 0:
            raise ValueError("No matching image name found for the given parameters.")

        image_name = image_name_values[0]  # Assuming one match

        # Construct the full path to the pickle file
        pickle_path = os.path.join(base_dir, folder_path, f"{image_name}_plot_data.pkl")

        return pickle_path
    except:
        return -1

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

    plt.show()

def filter_umap_data(umap_embeddings: np.ndarray, label_data: np.ndarray, df_umap_tiles: pd.DataFrame, filters: dict):
    """
    Filters umap_embeddings, label_data, and df_umap_tiles based on values in filters.

    Args:
        umap_embeddings (np.ndarray): 2D array of shape (N, 2) containing UMAP embeddings.
        label_data (np.ndarray): 1D array of shape (N,) containing labels for each embedding.
        df_umap_tiles (pd.DataFrame): DataFrame containing image statistics.
        filters (dict): Dictionary where keys are column names in df_umap_tiles and values are lists of allowed values.

    Returns:
        np.ndarray: Filtered umap_embeddings.
        np.ndarray: Filtered label_data.
        pd.DataFrame: Filtered df_umap_tiles.
    """
    # Apply all filters to df_umap_tiles
    mask = np.ones(len(df_umap_tiles), dtype=bool)  # Start with all True
    for column, values in filters.items():
        mask &= df_umap_tiles[column].apply(lambda x: any(str(x).startswith(prefix) for prefix in values))

    # Apply mask to all data
    filtered_umap = umap_embeddings[mask]
    filtered_labels = label_data[mask]
    filtered_df = df_umap_tiles.iloc[list(mask)].copy().reset_index()

    return filtered_umap, filtered_labels, filtered_df

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

def check_memory_status():
    limit = get_lsf_mem_limit_gb()
    usage = get_ram_usage_gb()

    if limit is not None:
        print(f"Memory Usage is: {usage:.2f} GB, LSF Memory Limit: {limit:.2f} GB")
        if usage > 0.9 * limit:
            print("🚨 Memory usage is above 90% of allocated LSF limit! Please restart kernel or allocate more memory.")