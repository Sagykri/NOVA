import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import numpy as np

PROCESSED_FOLDER_ROOT = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps','input','images','processed')
batch2_folders = ["220814_neurons",
                "220818_neurons",
                "220831_neurons",
                "220908", "220914"]

BATCH2_SPD_PROCESSED_FOLDER = os.path.join(PROCESSED_FOLDER_ROOT, 'spd2','SpinningDisk','batch2')


for folder in batch2_folders:
    data_folder_path = os.path.join(PROCESSED_FOLDER_ROOT, folder)
    for cell_line in os.listdir(data_folder_path):
        cell_line_folder = os.path.join(data_folder_path, cell_line)
        for condition in os.listdir(cell_line_folder):
            condition_folder = os.path.join(cell_line_folder, condition)
            for marker in os.listdir(condition_folder):
                marker_folder = os.path.join(condition_folder, marker)
                marker_file_name =  os.path.join(marker_folder, os.listdir(marker_folder)[0]) # only one file per marker
                tiles = np.load(marker_file_name)
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
                        save_path = os.path.join(BATCH2_SPD_PROCESSED_FOLDER, cell_line)
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