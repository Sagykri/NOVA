import numpy as np


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

default_paths_funova = {
    'umaps_folder' : '/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/funova_finetuned_model/figures/funova/UMAPs',
    'preprocessing_path' : '/home/labs/hornsteinlab/Collaboration/FUNOVA/outputs/preprocessing/brenner',
    'csv_name1' : 'raw_metrics280125_exp3.csv',
    'csv_name2' : 'raw_metrics110225_exp4.csv',
    'images_dir' : '/home/labs/hornsteinlab/Collaboration/FUNOVA/input/images/raw'
    }
