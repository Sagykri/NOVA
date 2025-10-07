# ------------------- #
# ----- IMPORTS ----- #
# ------------------- #
import re
import os
import sys
import numpy as np
import pandas as pd

import cv2
from skimage.filters import threshold_otsu

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

os.environ['NOVA_HOME'] = '/home/projects/hornsteinlab/Collaboration/NOVA'
sys.path.insert(1, os.getenv('NOVA_HOME'))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")



# ------------------- #
# ----- CONFIGS ----- #
# ------------------- #

colors = [(0, 'black'), (1, 'red')] # 0 represents the start (black), 1 represents the end (red)
cmap_black_to_red = LinearSegmentedColormap.from_list("BlackToRed", colors)
colors = [(0, 'black'), (1, 'royalblue')] # 0 represents the start (black), 1 represents the end (red)
cmap_black_to_blue = LinearSegmentedColormap.from_list("BlackToBlue", colors)
colors = [(0, 'black'), (1, 'green')] # 0 represents the start (black), 1 represents the end (red)
cmap_black_to_green = LinearSegmentedColormap.from_list("BlackToGreen", colors)

colors = {'DCP1A': cmap_black_to_red, 'TDP43': cmap_black_to_green, 'DAPI': cmap_black_to_blue}

# ---------------------------- #
# ----- INPUT DATA UTILS ----- #
# ---------------------------- #

def load_tiles_to_dataframe(paths, is_new_Coyne=False):
    
    # --- Create a DataFrame of preprocesed tiles
    # each row single tile, include metadata extracted from each path
    tile_records = []
    for path in paths:
        if 'CD41' in path:
            continue

        # Extract meta-data
        _,_,_,_,_,_,_,_,_,_,_, batch, cell_line, cond, marker_name, filename = path.split(os.sep)

        # To extract the first token from the filename that starts with 's' and is followed by digits
        match = re.search(r's\d+', filename)
        site_num = match.group() if match else None
        
        if is_new_Coyne:
            # patient ID is part of cell line label
            part = cell_line.split('-')
            patient_id = part[1]
            cell_line = part[0]
        else:
            # patient ID is "rep"
            patient_id = [part for part in filename.split('_') if part.startswith('rep')][0]
            

        # Load the processed numpy tiles
        all_tiles = np.load(path) # shape (N, 100, 100, 2)

        num_tiles = all_tiles.shape[0]

        # Create a record per tile - collect rows into a list of dictionaries
        for i in range(num_tiles):
            tile = all_tiles[i]
            tile_records.append({
                'path': path,
                'patient_id': patient_id,
                'batch': batch,
                'cell_line': cell_line,
                'condition': cond,
                'marker': marker_name,
                'filename': filename,
                'tile_index': i,
                'site_num': site_num,
                'tile_data': tile #.flatten()  # tile as 1D
            })

    # Convert list of dictionaries to DataFrame
    tiles_df = pd.DataFrame(tile_records)


    print(f"Created DataFrame of {tiles_df.shape[0]} tiles with metadata.")
    return tiles_df


def sites_paths_to_dataframe(paths):
    
    # --- Create a DataFrame of metadata of preprocesed siles
    site_records = []
    for path in paths:
        #print(path, path.split(os.sep))

        if 'CD41' in path:
            continue

        # Extract meta-data
        _,_,_,_,_,_,_,_,_,_, batch, cell_line, panel, condition, rep, marker_name, filename = path.split(os.sep)

        # Create a record per tiff image - collect rows into a list of dictionaries
        site_records.append({
            'path': path,
            'batch': batch,
            'cell_line': cell_line,
            'panel': panel,
            'condition': condition,
            'rep': rep, 
            'marker': marker_name,
            'filename': filename,

        })

    # Convert list of dictionaries to DataFrame
    sites_df = pd.DataFrame(site_records)


    print(f"Created DataFrame of {sites_df.shape[0]} sites with their paths and metadata (to suuport lazy loading).")
    return sites_df


# -------------------------------------- #
# ----- COLOCALIZATION SCORE UTILS ----- # 
# -------------------------------------- #

def match_tiles_and_score_fraction_overlap(df, dcp1a_threshold_method, tdp43_threshold_method, rotated_tdp=False):
    """
    Matches DCP1A tiles with corresponding TDP-43 tiles and computes fraction_overlap.
    
    Parameters:
    - df: DataFrame with tile data (must include patient_id, cell_line, site_num, tile_index, marker, tile_data)
    - rotated_tdp: If True, rotates the TDP channel 90 degrees to the left (for null testing)
    - dcp1a_threshold_method: thresholding method for DCP1A (e.g., 'mean+2sd' or 'otsu')
    - tdp43_threshold_method: thresholding method for TDP-43
    
    Returns:
    - A copy of the DCP1A tile dataframe with an added 'fraction_overlap' column
    """

    df_DCP1A = df[df['marker'].isin(['DCP1A'])].copy()
    df_TDP43 = df[df['marker'].isin(['TDP43'])].copy()
    print(f"df_DCP1A.shape: {df_DCP1A.shape}, and df_TDP43.shape {df_TDP43.shape}")

    if rotated_tdp:
        print("Note: estimating rotated validation — TDP-43 channel will be rotated 90° left.")

    print(f'Matching multiplexed tiles and scoreing tiles "fraction_overlap".')
    for i, dcp1a_tile in df_DCP1A.iterrows():

        patient_id = dcp1a_tile['patient_id']
        cell_line = dcp1a_tile['cell_line']
        site_num = dcp1a_tile['site_num']
        tile_index = dcp1a_tile['tile_index']
        
        
        try:
            # for each image in DCP1A, find it's TDP-43 and DAPI channels
            assoc_tile = df_TDP43[
                                (df_TDP43['patient_id'] == patient_id) &                
                                (df_TDP43['cell_line'] == cell_line) &
                                (df_TDP43['marker'] == 'TDP43') &
                                (df_TDP43['tile_index'] == tile_index) &
                                (df_TDP43['site_num'] == site_num) 
            ].tile_data.values[0]

            dcp1a_tile = dcp1a_tile['tile_data'][:,:,0]
            tdp_tile = assoc_tile[:,:,0]
            dapi_tile = assoc_tile[:,:,1]
            
            # Rotates the TDP channel 90 degrees to the left (np.rot90) before scoring fraction_overlap
            if rotated_tdp:
                tdp_tile = np.rot90(tdp_tile, k=1)  # 90 degrees counterclockwise
            
            # Scoreing tiles "fraction_overlap"
            df_DCP1A.loc[i, 'fraction_overlap'] = score_image_fraction_overlap(
                dcp1a_tile,
                tdp_tile,
                dcp1a_threshold_method,
                tdp43_threshold_method
            )
            
        except IndexError as e:
            #print(e, cell_line, site_num, tile_index)
            # The TDP tile is filtered. move on 
            df_DCP1A.loc[i, 'fraction_overlap'] = None

    n_missing = df_DCP1A['fraction_overlap'].isna().sum()
    print(f"Tiles without matched TDP43: {n_missing}")
    return df_DCP1A

def score_image_fraction_overlap(dcp1a_tile, tdp_tile, dcp1a_threshold_method, tdp43_threshold_method, plot_masks=False):

    # --- Define thresholds for each marker (custom or statistical)
    def get_threshold(img, method="mean+2sd"):
        
        # first define what counts as an “on” pixel (positive signal) in each channel:
        if method == "mean+3sd":
            return img.mean() + 3 * img.std()
        elif method == "mean+2.5sd":
            return img.mean() + 2.5 * img.std()
        elif method == "mean+2sd":
            return img.mean() + 2 * img.std()
        elif method == "mean+1sd":
            return img.mean() + 1 * img.std()
        elif method == "mean+0.5sd":
            return img.mean() + 0.5 * img.std()
        elif method == "otsu":
            return threshold_otsu(img)
        else:
            raise ValueError("Unsupported method")
    
    def calc_score(dcp1a_mask, tdp_mask, plot_masks=False):
        """
            Calculate conditional "overlap fraction" (score between 0 and 1)
            If it’s close to 1, TDP-43 is highly colocalized with P-bodies. If it’s near 0, colocalization is weak.
        """
        overlap_pixels = np.logical_and(dcp1a_mask, tdp_mask).sum()
        total_dcp1a_pixels = dcp1a_mask.sum()

        fraction_overlap = overlap_pixels / total_dcp1a_pixels
        return fraction_overlap

    # --- Compute binary masks
    dcp1a_thresh = get_threshold(dcp1a_tile, method=dcp1a_threshold_method) 
    tdp_thresh   = get_threshold(tdp_tile, method=tdp43_threshold_method)
            
    # Thresholding (“on” pixels): intensity thresholding to binarize the images.
    dcp1a_mask = dcp1a_tile > dcp1a_thresh
    tdp_mask   = tdp_tile > tdp_thresh
    
        
    # --- Score individual image tiles for colocalization (fraction_overlap)
    fraction_overlap = calc_score(dcp1a_mask, tdp_mask)
    
    # --- Plot masking
    if plot_masks: 
        plot_masks(dcp1a_tile, tdp_tile, dcp1a_mask, tdp_mask)
        print(f"Colocalization fraction: {fraction_overlap:.3f}")
    
    return fraction_overlap

def match_sites_and_score_fraction_overlap(df, dcp1a_threshold_method, tdp43_threshold_method):
    
    dcp1a_channel_num, tdp_channel_num = 'ch2', 'ch3'
    
    df_DCP1A = df[df['marker'].isin(['DCP1A'])].copy()
    df_TDP43 = df[df['marker'].isin(['V5'])].copy()
    print(f"df_DCP1A.shape: {df_DCP1A.shape}, and df_TDP43.shape {df_TDP43.shape}")
    
    print(f'Matching multiplexed sites and scoreing sites for "fraction_overlap" and "fraction_overlap_rotated".')
    for i, dcp1a_site in df_DCP1A.iterrows():

        batch = dcp1a_site['batch']
        cell_line = dcp1a_site['cell_line']
        panel = dcp1a_site['panel']
        condition = dcp1a_site['condition']
        rep = dcp1a_site['rep']
        filename = dcp1a_site['filename']

        # Change file name
        tdp_expected_filename = filename.replace(dcp1a_channel_num, tdp_channel_num)
        
        
        try:
            # for each image in DCP1A, find it's TDP-43 and DAPI channels
            assoc_site = df_TDP43[
                                (df_TDP43['batch'] == batch) &                
                                (df_TDP43['cell_line'] == cell_line) &
                                (df_TDP43['panel'] == panel) &
                                (df_TDP43['condition'] == condition) &
                                (df_TDP43['rep'] == rep) &
                                (df_TDP43['marker'] == 'V5') &
                                (df_TDP43['filename'] == tdp_expected_filename) 
            ]

            # Get path to tiff files
            tdp_site_path = assoc_site.path.values[0]
            dcp1a_site_path = dcp1a_site.path
            
            # Load the processed TIFF sites into numpy array shape (1024, 1024)
            tdp_site = cv2.imread(tdp_site_path, cv2.IMREAD_ANYDEPTH)
            dcp1a_site = cv2.imread(dcp1a_site_path, cv2.IMREAD_ANYDEPTH)

            # Rotates the TDP channel 90 degrees to the left (np.rot90) before scoring fraction_overlap
            tdp_site_rotated = np.rot90(tdp_site, k=1)  # 90 degrees counterclockwise
            
            # Scoreing tiles "fraction_overlap"
            df_DCP1A.loc[i, 'fraction_overlap'] = score_image_fraction_overlap(
                dcp1a_site,
                tdp_site,
                dcp1a_threshold_method,
                tdp43_threshold_method
            )
            
            # Scoreing sites "fraction_overlap_rotated"
            df_DCP1A.loc[i, 'fraction_overlap_rotated'] = score_image_fraction_overlap(
                dcp1a_site,
                tdp_site_rotated,
                dcp1a_threshold_method,
                tdp43_threshold_method
            )
            
        except IndexError as e:
            #print(e, cell_line, site_num, tile_index)
            # The TDP tile is filtered. move on 
            df_DCP1A.loc[i, 'fraction_overlap'] = None
            df_DCP1A.loc[i, 'fraction_overlap_rotated'] = None

    n_missing = df_DCP1A['fraction_overlap'].isna().sum()
    print(f"Sites without matched TDP43: {n_missing}")
    return df_DCP1A


# -------------------------- #
# ----- PLOTTING UTILS ----- # 
# -------------------------- #

def set_palette_and_labels_per_group(unique_groups, color_mapping_config):
    
    color_mapping = color_mapping_config
    # ------------------------------------- #
    # Define a color map per cell_line (plot with cell_line-specific colors)
    palette = {}
    for g in unique_groups:
        if g in color_mapping: palette[color_mapping[g]['alias']] = color_mapping[g]['color']
    # Rename groups to aliases
    label_mapping = {k: v["alias"] for k, v in color_mapping.items() if k in unique_groups}
    
    return palette, label_mapping
            
def plot_masks(dcp1a_tile, tdp_tile, dcp1a_mask, tdp_mask):
    def pretty(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # --- Plot 5 images: DCP1A, TDP-43, DAPI, DCP1A_mask, TDP_mask
    fig, axs = plt.subplots(1, 4, figsize=(15, 3))
    axs = axs.flatten()
    plt.subplots_adjust(wspace=0.2) # Adjust vertical spacing between rows

    imgs_to_show = [dcp1a_tile, tdp_tile, dcp1a_mask, tdp_mask]
    titles = ['DCP1A', 'TDP-43', 'DCP1A mask', 'TDP-43 mask']
    cmaps = [colors['DCP1A'], colors['TDP43'],  'gray', 'gray']

    for i, (img, title) in enumerate(zip(imgs_to_show, titles)):
        ax = axs[i]
        # Note to vmin vmax!! important 
        ax.imshow(img, cmap=cmaps[i], vmin=0, vmax=1) 
        ax.set_title(title, fontsize=10)
        pretty(ax)

    plt.show()
