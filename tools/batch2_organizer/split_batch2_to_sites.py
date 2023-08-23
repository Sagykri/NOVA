import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import cv2
import numpy as np
from skimage import io
import multiprocessing

# SOURCE 
RAW_FOLDER_ROOT = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps','input','images','raw')
batch2_folders = ["220814_neurons",
                "220818_neurons",
                "220831_neurons",
                "220908", "220914"]
# DESTINATION 
BATCH2_SITES_FOLDER = os.path.join(RAW_FOLDER_ROOT, 'batch_2_sites')

# TILES SIZE
tile_size = 13413 // 10  # Assuming equal split into 100 tiles


def split_image_into_tiles(image, file_name, output_folder, tile_size, ):
    
    print(f"\nsaving {file_name}")

    height, width = image.shape[:2]
    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size
    
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            left = x * tile_size
            upper = y * tile_size
            right = left + tile_size
            lower = upper + tile_size
            tile = image[upper:lower, left:right]
            tile_filename = f"{output_folder}/site_{x}_{y}_{file_name}"
            # writing the site image
            cv2.imwrite(tile_filename, tile)

def extract_markers_from_filename(filename):
    filename_no_ext = filename.split('.tif')[0]
    markers_str = filename_no_ext.split('-')[1]
    markers = markers_str.split('_')
    # markers.remove('DAPI')
    
    # Flip the order
    markers = markers[::-1]
    markers = [(i, m) for i, m in enumerate(markers)]
    
    return markers
      
def count_tif_files_recursive(folder_path):
    tif_count = 0
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".tif"):
                tif_count += 1
    
    return tif_count

def generate_panels_list(size):
    letter_list = []
    current_letter = ord('A')
    
    for _ in range(size):
        letter_list.append("panel"+chr(current_letter))
        current_letter += 1
    
    return letter_list

def main(folder):
    
        data_folder_path = os.path.join(RAW_FOLDER_ROOT, folder)
        print(f"\n\nFolder: {data_folder_path}")
        
        for cell_line in sorted(os.listdir(data_folder_path)):
            orig_cell_line_folder = os.path.join(data_folder_path, cell_line)
            cell_line_folder = os.path.join(BATCH2_SITES_FOLDER, cell_line)
            
            for condition in sorted(os.listdir(orig_cell_line_folder)):
                orig_condition_folder = os.path.join(orig_cell_line_folder, condition)
                # Need to artificially create panel folders - folder for each file  
                n_panels = count_tif_files_recursive(orig_condition_folder)
                panels = generate_panels_list(n_panels)
                
                # Align to new format
                if condition=='unstressed': condition='Untreated'
                if condition=='stressed': condition='stress'
                
                for markers_file_name, panel in zip(sorted(os.listdir(orig_condition_folder)), panels):
                    print(f"\n\nmarkers_file_name: {markers_file_name}")
                    
                    # Load the tif image (containes 4 channels) 
                    image = io.imread(os.path.join(orig_condition_folder, markers_file_name))
                    #print(image.shape)
                    markers = extract_markers_from_filename(markers_file_name)
                    panel_folder = os.path.join(cell_line_folder, panel)        
                    rep_folder = os.path.join(panel_folder, condition, 'rep1')

                    
                    for c_num, marker_dir_name in markers:
                        # take the relevant channel
                        c_image = image[:,:,c_num]
                        marker_folder = os.path.join(rep_folder, marker_dir_name)
                        # create marker folder
                        if not os.path.exists(marker_folder):
                            os.makedirs(marker_folder)
                        # save the initial of the original file name
                        new_file_name = markers_file_name.split('DAPI')[0] + marker_dir_name + '.tif'
                        save_path = os.path.join(marker_folder, new_file_name)
                        # Split original channel image to "sites" (tiles) and save uner proper path
                        split_image_into_tiles(c_image, new_file_name, marker_folder, tile_size=tile_size)

if __name__ == "__main__":
    # multiprocessing - create all tasks
    processes = [multiprocessing.Process(target=main, args=(folder,)) for folder in batch2_folders]
    # start all processes
    for process in processes:
        process.start()