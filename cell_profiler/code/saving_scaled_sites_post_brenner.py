
import os
import sys
import cv2
import logging
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
from functools import partial
import cellprofiler_core
import nova_utils as utils
import cell_profiler_utils as cp_utils

NOVA_HOME = os.getenv("NOVA_HOME")
# To be able to import from other scripts under "cell_profiler"
sys.path.insert(1, NOVA_HOME)
# ------------------------------------------------------------- #
# Paths to local directories 
# ------------------------------------------------------------- #
# Input folder name (raw sites)
dataset_name = 'TDP43_WT_OE_PB_experiment_sorted'
DATA_INPUT_DIR = os.path.join(NOVA_HOME, 'input', 'images', 'raw' ,dataset_name)
# , "batch3","iw11-NGN", "PanelX"
# Path to filtred and rescale 
OUTPUT_DIR = os.path.join(NOVA_HOME, 'cell_profiler', 'outputs', 'filtered_by_brenner_post_rescale_outputs', 'TDP43_WT_OE_PB_experiment_sorted')
LOG_DIR_PATH = os.path.join(NOVA_HOME, 'cell_profiler', 'logs')
# ------------------------------------------------------------- #


    
def filter_images_by_brenner_post_rescale_intensity(image_files, input_folder_name, markers_focus_boundries):
    """
    Filters and saves image pairs (marker and DAPI) based on Brenner focus thresholds after intensity rescaling.
    This function processes a list of image file paths, validating each marker/DAPI pair using Brenner focus thresholds
    specific to the provided dataset. Valid image pairs are saved into an organized output directory structure, and the
    paths to the saved files are returned.
    Args:
        image_files (list of str): List of image file paths, expected to be ordered as marker/DAPI pairs.
        input_folder_name (str): Name of the input folder, used to construct the output directory path.
        markers_focus_boundries (pd.DataFrame): DataFrame containing Brenner focus thresholds for each marker.
    Returns:
        list of pathlib.Path: List of file paths to the saved, filtered images.
    Raises:
        ValueError: If the input image file list is empty or the folder structure cannot be determined.
    Notes:
        - The function expects the image files to be organized such that every two consecutive files form a marker/DAPI pair.
        - Only pairs where both marker and DAPI images pass the Brenner focus threshold are saved.
        - The output directory structure mirrors the input, starting from the 'batchX' folder up to the marker name.
        - Existing files are not overwritten; if a file already exists, it is skipped.
    """
    filtered_image_files = []
    saved_file_paths = []
    
    # Step 2: Loop through image files two at a time (marker and DAPI)
    for i in range(0, len(image_files), 2):
        pair = image_files[i:i+2]
        if len(pair) < 2:
            continue

        marker, dapi = pair[0], pair[1] 

        # Step 3: Validate DAPI image
        dapi_image, valid_dapi_path = utils.cp_get_valid_site_image(dapi, markers_focus_boundries)
        if valid_dapi_path:
            # Step 4: Validate marker image
            marker_image, valid_marker_path = utils.cp_get_valid_site_image(marker, markers_focus_boundries)
            
            if valid_marker_path:
                filtered_image_files.append((marker_image, valid_marker_path))
                filtered_image_files.append((dapi_image, valid_dapi_path))

    # Step 5: Print debug info
    print(f" Total input files: {len(image_files)}")
    print(f" Filtered files: {len(filtered_image_files)}")

    # Step 6: Extract folder structure from 'batchX' up to marker
    if image_files:
        first_file_path = Path(image_files[0])
        parts = first_file_path.parts
        batch_index = next(i for i, part in enumerate(parts) if part.startswith("batch"))
        marker_index = len(parts) - 2  # The folder before the file is the marker
        subdirs = parts[batch_index:marker_index]  # Stop before marker
        subdir_path = Path(*subdirs)
    else:
        raise ValueError("Could not extract subdirs names from sample path.")
#
    # Step 7: Create the output directory based on the input folder name 
    output_dir = os.path.join(OUTPUT_DIR, subdir_path)
    print("Output dir:", output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 8: Save each processed image in marker-specific subfolder
    for image, original_path in filtered_image_files:
        
        original_path = Path(original_path)
        marker_name = original_path.parent.name  # marker folder (e.g. "DCP1A")

        marker_output_dir = os.path.join(output_dir, marker_name)
        Path(marker_output_dir).mkdir(parents=True, exist_ok=True)

        dst_path = os.path.join(marker_output_dir, original_path.name)
        
        #  Skip if file already exists
        if Path(dst_path).exists():
            # logging.info(f" Skipping existing file: {dst_path}")
            saved_file_paths.append(dst_path)
            continue
        if image is not None:
            #print(" Saving processed image:", dst_path)
            cv2.imwrite(str(dst_path), image)
            saved_file_paths.append(dst_path)  # APPEND the saved path
        else:
            logging.info(f" Skipping invalid image: {original_path}" )

    # Step 9: Return saved file paths
    return saved_file_paths

def save_filter_sites(input_and_output_path_list, dataset_name, markers_focus_boundries):
    
    # The marker folder to read site images from and where to save the filtered site images     
    input_folder_name, output_folder = input_and_output_path_list[0], input_and_output_path_list[1]
    logging.info(f"Filtering images: reading input data from {input_folder_name}")
    
    # Collect image file paths for a given marker and its associated DAPI channel
    image_files = cp_utils.collect_image_names_per_marker(input_folder_name, dataset_name)
    
    # Filter the images using Brenner focus measure, rescale intensity, and save filtered images
    filtered_image_files = filter_images_by_brenner_post_rescale_intensity(image_files, input_folder_name, markers_focus_boundries)

    logging.info(f"Filtering complete. {len(filtered_image_files)} images passed the filter for: {input_folder_name}")
    return None


def main(dataset_name):
    """
    Filter images by Brenner threshold, rescale intensity, save filtered images to a new location,
    """
    
    # Step 1: Load Brenner focus thresholds
    markers_focus_boundries = utils.cp_load_markers_focus_boundries(dataset_name)
    
    # create a process pool that uses all cpus
    with Pool(5) as pool:
        # call the analyze_marker() function for each marker folder in parallel
        for _ in pool.map(partial(save_filter_sites, dataset_name=dataset_name, markers_focus_boundries=markers_focus_boundries), 
                                cp_utils.find_marker_folders(batch_path=DATA_INPUT_DIR, 
                                                            output_dir=OUTPUT_DIR, 
                                                            depth=6, 
                                                            markers_to_include=[])):
            pass  # No need to do anything here
            
    logging.info("Terminating the java utils and process pool (killing all tasks...)")
    # stop java                
    cellprofiler_core.utilities.java.stop_java()
    # forcefully terminate the process pool and kill all tasks
    pool.terminate()
    
        


    
if __name__ == '__main__': 
    try:   
        # Define the log file once in the begining of the script
        cp_utils.set_logging(log_file_path=os.path.join(LOG_DIR_PATH, datetime.now().strftime('log_%d_%m_%Y_%H_%M')))
    
        main(dataset_name)
    except Exception as e:
        logging.info(f"Error:{e}")
        logging.exception(e)
        
    logging.info(f"\n\nDone!")  

 