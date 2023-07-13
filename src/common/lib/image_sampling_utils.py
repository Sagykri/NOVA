#from multiprocessing import Pool
#from datetime import datetime
from glob import glob 
import numpy as np
import logging
import pathlib
import random
import os
import sys


# Global paths
BATCH_TO_RUN = 'batch2' 

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR = os.path.join(BASE_DIR,'input','images','processed','spd2','SpinningDisk')
INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)


def find_marker_folders(batch_path, depth=5):
    """Returns paths of all marker folder in a batch (assumed to be depth of 5 levels)
    works with recursion
    Args:
        batch_path (string):  full path of batch folder
        depth (int, optional): depth of marker sub-folders. Defaults to 5.
        Note: Markers are assumend to be always in a given constant "depth"

    Yields:
        string: a path of marker folder
    """
    
    # Recursively list files and directories up to a certain depth
    depth -= 1
    with os.scandir(batch_path) as input_data_folder:
        
        for entry in input_data_folder:
            
            # if that's not a marker directory, recursion...
            if entry.is_dir() and depth > 0:
                yield from find_marker_folders(entry.path, depth)
                
            # if that's a marker directory
            elif depth==0: 
                marker_name = os.path.basename(entry.path)
                if marker_name=='DAPI':
                    continue
                else:
                    # This is a list of arguments, used as the input of analyze_marker()
                    yield entry.path

def sample_image_names_per_marker(input_data_dir, sample_size=1):
    """
    For a given target marker, this function samples file names of images 
    (each image is stored in npy of (n_tiles, 100, 100, 2), AKA target and DAPI marker 
    
    Args:
        input_data_dir (string): full path of marker directory
        Note: "input_data_dir" has to point to a marker directory
        sample_size (int, optional): how  many images to sample. Defaults to 1.

    Returns:
        _type_: _description_
    """
    
    
    print(f"\nsample_image_names_per_marker: {input_data_dir}. {sample_size} images per marker.")
    
    # This will hold the full path of n images (n is defined by "sample_size") of the marker
    filenames = random.sample(os.listdir(input_data_dir), sample_size)
    
    files_list = []
    # Target marker
    for target_file in filenames:
        filename, ext = os.path.splitext(target_file)
        if ext == '.npy':
            image_filename = os.path.join(input_data_dir, target_file)
    
            # Add to list
            files_list.append(image_filename)
        
        else:
            print(f"sampled file {target_file} was not a npy. re-sampling.. ")
            continue

    #("\n\n\nThe files sampled", files_list)
            
    return files_list

def sample_images_all_markers(cell_line_path=None, sample_size_per_markers=1, num_markers=10):
        """Samples random raw images for a given batch 

        Args:
            cell_line_path (string): path to cell line images
            sample_size_per_markers (int, optional): how many images to sample for each marker. Defaults to 1.
            num_markers (int, optional): how many markers to sample. Defaults to 10.

        Returns:
            list: list of paths (strings) 
        """
        
        sampled_images = []
        sampled_markers = set()
        
        # Get a list of all marker folders
        marker_subfolder = find_marker_folders(INPUT_DIR_BATCH, depth=3)
        
        # Sample n markers, and for each marker, sample k images (where n=num_markers and k=sample_size_per_markers)
        for marker_folder in marker_subfolder:
            
            n_images = 0
            
            if (len(sampled_markers) < num_markers):
            
                if (n_images<sample_size_per_markers):
            
                    sampled_marker_images = sample_image_names_per_marker(marker_folder, sample_size=sample_size_per_markers)
                    
                    if sampled_marker_images:
                        sampled_images.extend(sampled_marker_images)
                        sampled_markers.add(marker_folder)
                        
                        n_images += 1
                if (n_images==sample_size_per_markers): 
                    continue
            
        print("sampled_images:", len(sampled_images), "sampled_markers:", len(sampled_markers))
        return sampled_images

def sample_images_all_markers_all_lines(sample_size_per_markers=10):
    images_paths = []
    
    for cell_line in os.listdir(INPUT_DIR_BATCH):
        
        # get the full path of cell line images
        cell_line_path = os.path.join(INPUT_DIR_BATCH, cell_line)
        
        # Sample markers and then sample images of these markers. The returened value is a list of paths (strings) 
        paths = sample_images_all_markers(cell_line_path, sample_size_per_markers=sample_size_per_markers, num_markers=26)
        images_paths.extend(paths)
        
    return images_paths

if __name__ == '__main__':
    
    print("\n\n\nStart..")
    
    images = sample_images_all_markers_all_lines(1)
    
    print("\n\nTotal of", len(images), "images were sampled.")
    print("\n\n\n\nDone!")
    