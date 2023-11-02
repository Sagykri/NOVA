"""
Split the U2OS preprocesed files to numpy with 16 tiles of 100x100 each
"""
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import numpy as np

# PROCESSED_FOLDER_ROOT = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps','input','images','processed')
# batch2_5_folder = ["batch_2_5"]

PROCESSED_FOLDER_ROOT = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps','input','images','processed',)
input_folder = "220714/U2OS/"

OUTPUT_SPD_PROCESSED_FOLDER = os.path.join(PROCESSED_FOLDER_ROOT, 'Confocal', 'U2OS_spd_format')

cell_line = 'U2OS'

# Main folder
data_folder_path = os.path.join(PROCESSED_FOLDER_ROOT, input_folder)
# condition folders
for condition in sorted(os.listdir(data_folder_path)):
    condition_folder = os.path.join(data_folder_path, condition)
    print(f"condition_folder: {condition_folder}")
    # rename stress condition 
    if condition=='unstressed': condition = 'Untreated'
    elif condition=='stressed': condition = 'stress'
    # marker folders
    for marker in sorted(os.listdir(condition_folder)):
        marker_folder = os.path.join(condition_folder, marker)
        print(f"marker_folder: {marker_folder}")
        marker_file_name =  os.path.join(marker_folder, sorted(os.listdir(marker_folder))[0]) # only one file per marker
        tiles = np.load(marker_file_name)
        print(f"tiles shape: {tiles.shape}")
        n_tiles = tiles.shape[0]
        if n_tiles>0:
            site_number = 1
            for i in range(0, n_tiles, 16):
                
                # Store "data" with 16 tiles (or less) at a time
                start = i
                if i+16<=n_tiles: 
                    end = i+16 
                else: 
                    end = n_tiles                        
                data = tiles[start:end,...]
                
                # save the file (and create subfolders if needed)
                save_path = os.path.join(OUTPUT_SPD_PROCESSED_FOLDER, cell_line)
                if not os.path.exists(save_path):
                    os.mkdir(save_path, mode=0o777)
                save_path = os.path.join(save_path, condition)
                if not os.path.exists(save_path):
                    os.mkdir(save_path, mode=0o777)
                save_path = os.path.join(save_path, marker)
                if not os.path.exists(save_path):
                    os.mkdir(save_path, mode=0o777)
                # final npy file name
                save_path = os.path.join(save_path, 'rep1_s'+str(site_number)+'_processed.npy')
                np.save(save_path,data)
                site_number += 1
                
                print(f"Saved {save_path} with shape {data.shape}")
                        
                
                    
                    
                    
                    
                    
                
                
                
