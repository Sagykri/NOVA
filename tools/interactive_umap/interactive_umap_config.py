"""
This file defines UMAP configuration dictionaries used for loading embeddings, images, and layout metadata.

To add a new config:
1. Create a new dictionary following the structure of the existing `config_*` variables.
   The dictionary should include the following keys:
   - 'paths': a dictionary containing:
       - 'umaps_folder' (str): path to the folder containing all UMAPs
       - 'csv_path' (str or None): path to a CSV with Brenner metrics
       - 'images_dir' (list): list of image directories containing the raw images ('sites')
   - 'layouts': a dictionary defining panel-wise tile FOV layouts, or None

Example:
config_example = {
    'paths': {
        'umaps_folder': '/path/to/umaps',
        'csv_path': '/path/to/brenner.csv',
        'images_dir': ['/path/to/images/']
    },
    'layouts': None
}

2. Name the variable starting with `config_`.  
   It will automatically be added to the `configs` dictionary and accessible by name
   Example usage:
   from interactive_umap_config import configs  
   config = configs["config_example"]
"""

import numpy as np

momaps_home = "/home/projects/hornsteinlab/Collaboration/MOmaps"
momaps_input = f"{momaps_home}/input/images/raw"
momaps_figures = f"{momaps_home}/outputs/vit_models/finetuned_model/figures"
momaps_preprocessing = f"{momaps_home}/outputs/preprocessing"

## Day 8 Neurons ##
config_d8 = {
    'paths':{
    'umaps_folder' : f'{momaps_figures}/neurons_iu/UMAPs/',
    'csv_path' : f'{momaps_preprocessing}/spd/brenner/raw_metrics_all_batches_all_metrics_site_fix.csv',
    'images_dir' : [f'{momaps_input}/SpinningDisk/batch9/',
                    f'{momaps_input}/SpinningDisk/batch6/',]
    },
    'layouts': None
}

## Alyssa  ##
config_alyssa = {
    'paths':{
    'umaps_folder' : f'{momaps_figures}/AlyssaCoyne_7tiles_iu/UMAPs/',
    'csv_path' : None,
    'images_dir' : [f'{momaps_input}/AlyssaCoyne/MOmaps_iPSC_patients_TDP43_PB_CoyneLab/',]
    },
    'layouts': None
}

## deltaNLS ##
config_deltaNLS = {
    'paths':{
    'umaps_folder' : f'{momaps_figures}/deltaNLS_iu/UMAPs/',
    'csv_path' : f'{momaps_preprocessing}/spd/brenner/raw_metrics_all_batches_brenner_site_dNLS.csv',
    'images_dir' : [f'{momaps_input}/SpinningDisk/deltaNLS_sort/' ]
    },
    'layouts': None
}

## Day 18 Neurons ##
config_d18 = {
    'paths':{
    'umaps_folder' : f'{momaps_figures}/neurons_d18_iu/UMAPs/',
    'csv_path' : f'{momaps_preprocessing}/Opera18Days_Reimaged/brenner/raw_metrics_230724.csv',
    'images_dir' : [f'{momaps_input}/Opera18DaysReimaged_sorted/',]
    },
    'layouts': None
}

## Funova ##
# --- Define FOV Layouts Based on Batch and Panel ---
batch_panels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

# Standard FOV Layout (for most panels in batches 1-2)
base_fov_layout = np.array([[-1, 2, 3, 4, 5, 6, 7, 8, 9, -1],
    [-1, 17, 16, 15, 14, 13, 12, 11, 10, -1],
    [-1, 18, 19, 20, 21, 22, 23, 24, 25, -1],
    [-1, 33, 32, 31, 30, 29, 28, 27, 26, -1],
    [-1, 34, 35, 36, 37, 38, 39, 40, 41, -1],
    [-1, 49, 48, 47, 46, 45, 44, 43, 42, -1],
    [-1, 50, 51, 52, 53, 1, 54, 55, 56, -1],
    [-1, 64, 63, 62, 61, 60, 59, 58, 57, -1],
    [-1, 65, 66, 67, 68, 69, 70, 71, 72, -1],
    [-1, 80, 79, 78, 77, 76, 75, 74, 73, -1],
    [-1, 81, 82, 83, 84, 85, 86, 87, 88, -1],
    [-1, 96, 95, 94, 93, 92, 91, 90, 89, -1],
    [-1, -1, -1, 97, 98, 99, 100, -1, -1, -1]], dtype=object)

# Alternative Layout for Panel K
panel_k_layout = np.array([
    [-1, 2, 3, 4, 5, 6, 7, 8, 9, -1],
    [-1, 17, 16, 15, 14, 13, 12, 11, 10, -1],
    [-1, 18, 19, 20, 21, 22, 23, 24, 25, -1],
    [-1, 33, 32, 31, 30, 29, 28, 27, 26, -1],
    [-1, 34, 35, 36, 37, 38, 39, 40, 41, -1],
    [-1, 49, 48, 47, 46, 45, 44, 43, 42, -1],
    [-1, 50, 51, 52, 53, 1, 54, 55, 56, -1],
    [-1, 63, 62, 61, 60, 59, -1, 58, 57, -1],
    [-1, 64, 65, 66, 67, 68, 69, 70, 71, -1],
    [-1, 79, 78, 77, 76, 75, 74, 73, 72, -1],
    [-1, 80, 81, 82, 83, 84, 85, 86, 87, -1],
    [-1 , 95, 94, 93, 92, 91, 90, 89, 88, -1],
    [-1, -1, 96, 97, 98, 99, 100, -1, -1, -1]
], dtype=object)

# Alternative Layout for Panel L
panel_l_layout = np.array([
    [-1, 2, 3, 4, 5, 6, 7, 8, 9, -1],
    [-1, 17, 16, 15, 14, 13, 12, 11, 10, -1],
    [-1, 18, 19, 20, 21, 22, 23, 24, 25, -1],
    [-1, 33, 32, 31, 30, 29, 28, 27, 26, -1],
    [-1, 34, 35, 36, 37, 38, 39, 40, 41, -1],
    [-1, 49, 48, 47, 46, 45, 44, 43, 42, -1],
    [-1, 50, 51, 52, 53, 1, 54, 55, 56, -1],
    [-1, 64, 63, 62, 61, 60, 59, 58, 57, -1],
    [-1, 65, 66, 67, 68, 69, 70, 71, 72, -1],
    [-1, 79, 78, 77, -1, 76, 75, 74, 73, -1],
    [-1, 80, 81, 82, 83, 84, 85, 86, 87, -1],
    [96, 95, 94, 93, 92, 91, 90, 89, 88, -1],
    [-1, -1, -1, 97, 98, 99, 100, -1, -1, -1]
], dtype=object)

exp_4_layout = np.array([
    [-1, 2, 3, 4, 5, 6, 7, 8, 9, -1],
    [-1, 17, 16, 15, 14, 13, 12, 11, 10, -1],
    [-1, 18, 19, 20, 21, 22, 23, 24, 25, -1],
    [-1, 33, 32, 31, 30, 29, 28, 27, 26, -1],
    [-1, 34, 35, 36, 37, 38, 39, 40, 41, -1],
    [-1, 49, 48, 47, 46, 45, 44, 43, 42, -1],
    [-1, 50, 51, 52, 53, 1, 54, 55, 56, -1],
    [-1, 64, 63, 62, 61, 60, 59, 58, 57, -1],
    [-1, 65, 66, 67, 68, 69, 70, 71, 72, -1],
    [-1, 80, 79, 78, 77, 76, 75, 74, 73, -1],
    [-1, 81, 82, 83, 84, 85, 86, 87, 88, -1],
    [-1, 96, 95, 94, 93, 92, 91, 90, 89, -1],
    [-1, -1, -1, 97, 98, 99, 100, -1, -1, -1]
], dtype=object) 

# Dictionary of layouts for each batch
funova_fov_layouts = {
    "Batch1": {panel: (panel_k_layout if panel == "K" else panel_l_layout if panel == "L" else base_fov_layout.copy()) for panel in batch_panels},
    "Batch2": {panel: (panel_k_layout if panel == "K" else panel_l_layout if panel == "L" else base_fov_layout.copy()) for panel in batch_panels},
    "Batch3": {panel: exp_4_layout.copy() for panel in batch_panels},  # All panels in Batch 3 have the same layout
    "Batch4": {panel: exp_4_layout.copy() for panel in batch_panels}   # All panels in Batch 4 have the same layout
}

config_funova = {
    'paths':{
        'umaps_folder' : f'{momaps_figures}/funova/UMAPs',
        'csv_path' : "/home/projects/hornsteinlab/Collaboration/FUNOVA/outputs/preprocessing/brenner/raw_metrics_exp3_exp4.csv",
        'images_dir' : ['/home/projects/hornsteinlab/Collaboration/FUNOVA/input/images/raw',] 
    },
    'layouts': funova_fov_layouts
}

config_funova_finetuned = {
    'paths':{
        'umaps_folder' : '/home/projects/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/funova_finetuned_model/figures/funova/UMAPs',
        'csv_path' : "/home/projects/hornsteinlab/Collaboration/FUNOVA/outputs/preprocessing/brenner/raw_metrics_exp3_exp4.csv",
        'images_dir' : ['/home/projects/hornsteinlab/Collaboration/FUNOVA/input/images/raw',]
    },
    'layouts': funova_fov_layouts
}

# Collect all variables starting with "config" into a dictionary called `configs`.
# This allows easy access to all config objects via: configs['config_name']
configs = {
    k: v for k, v in globals().items()
    if k.startswith("config") and not callable(v)
}
