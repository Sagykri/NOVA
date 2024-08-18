
import logging
import os
import numpy as np
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))


# from tensorflow.compat.v1.data import Datset

from src.common.lib.dataset import Dataset
from src.common.configs.dataset_config import DatasetConfig

class DatasetSPD(Dataset):
    """
    Dataset for SPD images
    """
    def __init__(self, conf: DatasetConfig, transform=None):
        self.markers_folders_depth = 3
        
        super().__init__(conf, transform)
    
    def __find_marker_folders(self, batch_path, depth=3):
        """Returns paths of all marker folder in a batch (works with recursion)
            Note: 
                    Markers are assumend to be always in a given constant "depth"
                    Depth is assumed to be depth of 3 levels for preprocessed images, and 5 for raw images
        
        Args:
            batch_path (string):  full path of batch folder
            depth (int, optional): depth of marker sub-folders. Defaults to 5.
            
        Yields:
            string: a path of marker folder
        """
        
        # Recursively list files and directories up to a certain depth
        depth -= 1
        with os.scandir(batch_path) as input_data_folder:
            
            for entry in sorted(input_data_folder, key=lambda e: e.name):
                
                # if that's not a marker directory, recursion...
                if entry.is_dir() and depth > 0:
                    yield from self.__find_marker_folders(entry.path, depth)
                    
                # if that's a marker directory
                elif depth==0: 
                    # This is a list of arguments, used as the input of analyze_marker()
                    yield entry.path
                        
    def _load_data_paths(self):
        """ Return processed images (paths to npy files) from given folders 
            (each item in input_folders is a batch folder path)
            (each image is stored in npy of (n_tiles, 100, 100, 2), AKA target and DAPI marker 
    

        Args:
            
            depth (int): number of levels to marker folders
            For example, is structure is: MOmaps/input/images/processed/spd2/SpinningDisk/batch/cell_line/condition/marker/
            and input_folder_path is a a batch folder (e.g., 'MOmaps/input/images/processed/spd2/SpinningDisk/batch8'), then depth should be 3.
                            
        
        """
        
        input_folders           =   self.input_folders
        condition_l             =   self.add_condition_to_label
        line_l                  =   self.add_line_to_label
        batch_l                 =   self.add_batch_to_label
        rep_l                   =   self.add_rep_to_label
        cell_type_l             =   self.add_type_to_label
        markers                 =   self.markers
        markers_to_exclude      =   self.markers_to_exclude
        cell_lines_include      =   self.cell_lines
        conds_include           =   self.conditions
        depth                   =   self.markers_folders_depth
        reps_include            =   self.reps

        labels_changepoints = [0]
        labels = []
        # List of strings, each element in the list is marker name (e.g., "NONO")
        unique_markers = []
        # List of strings, each element in the list is a path to a processed image (npy file) 
        processed_files_list = []

        np.random.seed(self.conf.SEED)
        
        for i, input_folder in enumerate(input_folders):
            # logging.info(f"Input folder: {input_folder}, depth used: {depth}")
            
            # Get a list of ALL target marker folder path folders (using recursion)
            marker_subfolder = self.__find_marker_folders(input_folder, depth=depth)
            
            for marker_folder in marker_subfolder:
                
                # Count how many npy files we have for this marker direcroty 
                n_images = 0
                
                #####################################
                # Extract experimental settings from marker folder path (avoid multiple nested for loops..)
                marker_name = os.path.basename(marker_folder)
                condition = marker_folder.split('/')[-2]
                cell_line = marker_folder.split('/')[-3]
    
                #####################################
                # Filter: cell line
                if cell_lines_include is not None and cell_line not in cell_lines_include:
                    # logging.info(f"Skipping cell line (not in cell lines list). {cell_line}")
                    continue
                # cell_line_folder_fullpath = os.path.join(input_folder, cell_line)
                
                # Filter: stress condition
                if conds_include is not None and condition not in conds_include:
                    # logging.info(f"Skipping condition (not in conditions list). {condition}")
                    continue
                # cond_folder_fullpath = os.path.join(cell_line_folder_fullpath, condition)
                
                # Filter: marker to include
                if markers is not None and marker_name not in markers:
                    # logging.info(f"Skipping marker (not in markers list). {marker_name}")
                    continue
                    
                # Filter: marker to exclude
                if markers_to_exclude is not None and marker_name in markers_to_exclude:
                    # logging.info(f"Skipping (in markers to exclude). {marker_name}")
                    continue
                #####################################
                                        
                #####################################
                # Hold a list of all processed images (name of npy files) of this marker
                filenames = sorted(os.listdir(marker_folder))
                
                # Target marker - loop on all sites (single npy files)
                for target_file in filenames:
                    filename, ext = os.path.splitext(target_file)
                    if ext == '.npy' or ext == '.tif_processed':
                        
                        # Filter: rep to include
                        if reps_include is not None:
                            filename_rep = filename.split('_',1)[0]
                            if filename_rep not in reps_include:
                                # logging.info(f"Skipping rep (not in reps list). {filename_rep}")
                                continue
                        
                        # Hold the full path of a processed image 
                        image_filename = os.path.join(marker_folder, target_file)
                        # Add to list: the full path of the npy file 
                        processed_files_list.append(image_filename)
                        
                        # logging.info(f"Filepath (npy): {image_filename}")
                        n_images += 1
                    else:
                        logging.info(f"file {target_file} is not a npy. moving on.. ")
                        continue
                    
                    # Save images label (same label to all site)
                    lbl = marker_name
                    if line_l:
                        lbl += f"_{cell_line}"
                    if condition_l:
                        lbl += f"_{condition}"
                    if batch_l:
                        batch_postfix = f"{os.path.basename(input_folder)}"
                        lbl += f"_{batch_postfix}"
                    if rep_l:
                        filename_rep = filename.split('_',1)[0]
                        lbl += f"_{filename_rep}"
                        
                    # Save all unique markers names
                    if lbl not in unique_markers: 
                        unique_markers.append(lbl)
                    
                    labels += [lbl]
                    
                    
                # Save when there is change between markers/conditions
                labels_changepoints.append(n_images)
                #####################################
                
                # # Save images label (same label to all site)
                # lbl = marker_name
                # if line_l:
                #     lbl += f"_{cell_line}"
                # if condition_l:
                #     lbl += f"_{condition}"
                # if batch_l:
                #     batch_postfix = f"{os.path.basename(input_folder)}"
                #     lbl += f"_{batch_postfix}"
                # if rep_l:
                #     filename_rep = filename.split('_',1)[0]
                #     lbl += f"_{filename_rep}"
                    
                # # Save all unique markers names
                # if lbl not in unique_markers: 
                #     unique_markers.append(lbl)
                
                # labels += [lbl] * n_images
                
                # Nancy: currently, data folder doesn't contain "neurons"/"microglia"
                #if cell_type_l:
                #    lbl += f"_{cur_cell_type}"
                        
                        
        processed_files_list = np.asarray(processed_files_list)
        unique_markers = np.asarray(unique_markers)
        
        #####################################
        # Save the labels for entire input_folder
        labels = np.asarray(labels).reshape(-1, 1)
        logging.info(f"{len(processed_files_list)}, {labels.shape}")
        
        #####################################
        
        return processed_files_list, labels, unique_markers
