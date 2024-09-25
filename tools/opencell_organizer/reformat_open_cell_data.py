# TODO: MOVE TO A DIFFERENT FILE/FOLDER + UTILIZE CONFIGURATION (SAGY WROTE THIS TODO)


import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import numpy as np
import pandas as pd
from multiprocessing import Pool

from src.common.base_config import BaseConfig

# Paths
SOURCE_DATA = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps', 'OpenCell', 'downloaded_files')
DESTINATION_DATA = os.path.join(BaseConfig().PROCESSED_FOLDER_ROOT, 'spd2','SpinningDisk','OpenCell','WT','Untreated')

def load_opencell_npy_data(i):

    images_file = f'image_data0{i}.npy'
    labels_file = f'label_data0{i}.csv'
    
    data = np.load(os.path.join(SOURCE_DATA, images_file))
    labels_df = pd.read_csv(os.path.join(SOURCE_DATA, labels_file))
    assert data.shape[0] == labels_df.shape[0]
    print(f"\n\n\nLoaded OpenCell data (images and labels): {images_file} and {labels_file}")
    return data, labels_df

def create_marker_folder(marker_name):
    # Generate marker folder structure DESTINATION_DATA/marker_name/*.npy
    folder_path = os.path.join(DESTINATION_DATA, marker_name)
    # Check if marker folder exists, if not create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def extract_marker_images(marker_name, labels_df, data):
    # Create mask object from labels_df, to be used for filtering the npy file
    mask = labels_df['name'] == marker_name
    # Get all rows in the images npy file that correspond to the current marker        
    marker_data = data[mask]
    assert mask.sum()==marker_data.shape[0]
    print(f"\n\nextract_marker_images *** {marker_name} *** size {marker_data.shape}")
    
    # Each numpy is 4 channels: [target  protein, ç, nuclear distance, nuclear segmentation]
    # We need to take only the first channel (target protein) and the 2nd channel (nucleus)
    marker_data = marker_data[:,:,:,[0,1]]
    ##print(marker_data.shape)
    
    return marker_data

def reformat_opencell_file(file_num):
    
    # Save how many images were in each file and each marker
    stats = {}
    
    # Load OpenCell data (images and labels) - file number "file_num"
    data, labels_df = load_opencell_npy_data(file_num)
    
    # Every opencell npy file contains images from many markers, we wish to save images of same marker together.
    # Iterate by markers in the original npy file (e.g., image_data00.npy)
    unique_markers = sorted(labels_df['name'].unique()) ##[0:2]

    for marker in unique_markers:
        # Save the new formatted data under marker folders
        folder_path = create_marker_folder(marker_name=marker)
        # Get images of this marker only
        marker_data = extract_marker_images(marker, labels_df, data)
        # Split this image data to small npy files
        split_marker_data_to_small_npys_and_save(marker_data, folder_path)
        
        stats[marker] = marker_data.shape[0]
        
    return stats

def split_marker_data_to_small_npys_and_save(marker_data, folder_path):
    """
    Generate many npy files for each marker data, where number of tiles is approximatly 16 tile per npy file
    Save image data under appropriate marker folder
    """
    n_tiles = marker_data.shape[0]
    
    if n_tiles>0:
        for i in range(0, n_tiles, 16):
            # Store npy file with 16 tiles (or less) at a time
            start = i
            if i+16<=n_tiles: 
                end = i+16 
            else: 
                end = n_tiles                        
            marker_data_tmp = marker_data[start:end,...]
            #print(marker_data_tmp.shape)
            
            # Save the file (and create subfolders if needed)
            # final npy file name
            save_path = os.path.join(folder_path, 'rep1_s'+str(i)+'_processed.npy')
            np.save(save_path, marker_data_tmp)
            
            #print(f"Saved {save_path} with shape {marker_data_tmp.shape}")


def sum_by_key(dicts_list):
    result = {}
    for d in dicts_list:
        for marker, count in d.items():
            result[marker] = result.get(marker, 0) + count
    return result

all_stats = []
with Pool(10) as mp_pool:    
    for results in mp_pool.map(reformat_opencell_file, ([file_num for file_num in range(0,10)])):
        all_stats.append(results) 
    mp_pool.close()
    mp_pool.join()
    
    all_stats = sum_by_key(all_stats)
    
    print(f"\n\nFinal stats: {all_stats} \n\nNumber of images treated: {sum(all_stats.values())}. ")
    
    print("\n\nDone!!")
    


    


