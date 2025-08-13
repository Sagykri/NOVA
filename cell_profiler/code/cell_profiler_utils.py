"""
Preprocessing_utils 
To be able to import from other scripts under "cell_profiler"
"""

# Packages 
from contextlib import contextmanager
from datetime import datetime
from glob import glob 
import pandas as pd
import numpy as np
import logging
import pathlib
import cv2
import sys
import os

# Cell profiler utils 
import cellprofiler_core.pipeline
import cellprofiler_core.preferences
import cellprofiler_core.utilities.java
import saving_scaled_sites_post_brenner as site_post_brenner


NOVA_HOME = os.getenv("NOVA_HOME")
BASE_DIR = os.path.join('/home','projects','hornsteinlab','Collaboration','NOVA')
sys.path.insert(1, BASE_DIR)


def set_logging(log_file_path, level=logging.INFO, format=' INFO: %(message)s'):
    formatter = '%(asctime)s %(levelname)-8s %(message)s'
    handlers = [logging.FileHandler(log_file_path + '.log'), logging.StreamHandler()]
    logging.basicConfig(level=level, format=formatter, handlers=handlers, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(__doc__)
    return None


def init_cell_profiler(pipeline_path):
    
    logging.info("\n\nInitializing Cell Profiler..")

    cellprofiler_core.preferences.set_headless()
    cellprofiler_core.utilities.java.start_java()
    
    pipeline = loading_cellprofiler_pipeline(pipeline_path)
    
    return pipeline


def loading_cellprofiler_pipeline(pipeline_path):

    logging.info(f"loading_cellprofiler_pipeline {pipeline_path}")
    
    my_pipeline = cellprofiler_core.pipeline.Pipeline()
    my_pipeline.load(pipeline_path)
    
    logging.info("Finished loading_cellprofiler_pipeline..")
    return my_pipeline


def set_marker_output_folder(path):
    """
    Create and set the output folder for CellProfiler results.

    Parameters
    ----------
    path : str or pathlib.Path
        The full path to the desired output folder. If the folder does not exist,
        it will be created.

    Returns
    -------
    None

    Examples
    --------
    set_marker_output_folder("/home/projects/cell_profiler/outputs/batch1/DCP1A")

    Notes
    -----
    This function does not move or copy files; it only sets the output directory.
    Useful when running CellProfiler pipelines programmatically.
    """
    
    # Create the actual output folders in Wexac
    if not os.path.exists(path):
        os.makedirs(path)                
    
    # Set the default output folder for Cell Profiler
    logging.info(f"set_marker_output_folder: {path}")
    cellprofiler_core.preferences.set_default_output_directory(path)
                    
    return None

def collect_image_names_per_marker(input_data_dir, dataset_name=None):
    """
    Collect image file paths for a given marker and its associated DAPI channel.

    This function scans the input marker directory, identifies all image files 
    related to the target marker, and pairs each marker image with its corresponding 
    DAPI image based on the dataset naming convention.

    Parameters
    ----------
    input_data_dir : str or pathlib.Path
        Path to the directory containing marker images. This must point to a specific 
        marker folder (e.g., ".../batch1/DCP1A").
    dataset_name : str, optional
        Name of the dataset. If `'new_dNLS'`, DAPI file names are matched using 
        the "rXXcXXfXXX" prefix and "-ch1t1.tiff". Otherwise, the "Coyne" convention 
        is assumed, where DAPI images are matched by site suffix.

    Returns
    -------
    list of pathlib.Path
        A list of file paths containing both marker and DAPI images, paired 
        in sequential order. Each marker image is immediately followed by 
        its associated DAPI image.

    Notes
    -----
    - Input directory must point directly to a marker folder, not a batch folder.
    - The function relies on file naming conventions specific to the dataset.
    - This function does not validate file existence beyond simple glob matching.

    Examples
    --------
         collect_image_names_per_marker(
             "/home/projects/images/batch1/DCP1A",
             dataset_name="new_dNLS"
         )
    [PosixPath('/.../DCP1A/r01c01f001-ch2t1.tiff'),
     PosixPath('/.../DAPI/r01c01f001-ch1t1.tiff'),
     ...]
    """
    logging.info(f"collect_image_names_per_marker: {input_data_dir}")
    
    # This will hold the names of the all images of the marker (matched to it's DAPI channel)
    file_list = []
    # Define rep directory
    rep_dir = pathlib.Path(input_data_dir).parent.resolve()
    
    # Target marker
    for file in os.listdir(input_data_dir):
        filename, ext = os.path.splitext(file)
        if ext == '.tif' or ext == '.tiff':
            # Marker (target) tif file
            image_filename = os.path.join(input_data_dir, file)
            file_list.append(pathlib.Path(image_filename))    # CP requires absolute file paths
            
            nucleus_folder = os.path.join(rep_dir, "DAPI")
            # Find the file name of the associated DAPI site
            if dataset_name in ['OPERA_indi_sorted', 'OPERA_dNLS_6_batches_NOVA_sorted']:
                nucleus_filepath = filename.split("-")[0]+"-ch1t1.tiff"
                nucleus_filepath = glob(f"{nucleus_folder}/{nucleus_filepath}")[0]
            else:
                # Coyne
                site = filename.split('_')[-1]
                nucleus_filepath = glob(f"{nucleus_folder}/*_{site}{ext}")[0]
            file_list.append(pathlib.Path(nucleus_filepath))
                        
    return file_list

def paths_as_uri(image_files):
    # Turn the paths to URI as expected in Cell Profiler
    files = [file.as_uri() for file in image_files]
    return files

def extract_cell_profilers_features(image_files, pipeline):

    logging.info(f"\n\nextract_cell_profilers_features: {image_files}")     
    pipeline.read_file_list(image_files)
    logging.info("done pipeline.read_file_list()")
    output_measurements = pipeline.run()     
    logging.info("\n\ndone pipeline.run()")
    return None     
                     
                
def analyze_marker(input_and_output_path_list, pipeline_path):
    """
    """

    global pipeline
    pipeline = init_cell_profiler(pipeline_path)
    
    input_data_dir, output_folder = input_and_output_path_list[0], input_and_output_path_list[1]
    
    logging.info(f"Analyzing marker: reading input data from {input_data_dir}, saving results in {output_folder}")

    set_marker_output_folder(path=output_folder)

    image_files = collect_image_names_per_marker(input_data_dir)
  
    image_files = paths_as_uri(image_files)
    
    extract_cell_profilers_features(image_files, pipeline)
    
    return f"\n\nFinished extracting features for {input_data_dir}"
    
    
def run_cell_profiler_pipeline(filtered_image_files, pipeline_path, dataset_name='OPERA_indi_sorted',input_data_dir=None):
    """
    Initialize CellProfiler pipeline, convert paths to URI, and run the pipeline to extract features.
    """
    global pipeline
    pipeline = init_cell_profiler(pipeline_path)
    
    logging.info(f"File list recived: {filtered_image_files} ")
    # Ensure filtered_image_files is a list of paths
    input_data_dir, output_folder = filtered_image_files[0], filtered_image_files[1]
    
    logging.info(f"zzzzzzzzrun_cell_profiler_pipeline: {filtered_image_files} input_data_dir: {input_data_dir}, output_folder: {output_folder}")
    # Set the output folder for CellProfiler results
    
    set_marker_output_folder(path=output_folder)
    
    logging.info(f"set_marker_output_folder: saving results in {output_folder}")
    
    # collect image file paths for a given marker and its associated DAPI channel
    
    image_files_paths =  collect_image_names_per_marker(input_data_dir, dataset_name)
    
    # Convert filtered image file paths to URIs
    
    image_files_uri = paths_as_uri(image_files_paths)

    # Run CellProfiler pipeline on filtered images
    extract_cell_profilers_features(image_files_uri, pipeline)

    if input_data_dir:
        return f"\n\nFinished extracting features for {input_data_dir}"
    return "\n\nFinished extracting features"



def find_marker_folders(batch_path, output_dir, depth=4, markers_to_include=[], _root_batch_path=None):
    """ 
    For a given batch (defined by "batch_path") it "walks" to its subfolders (until "depth" is reached) 
    and returns for every marker a list of relevant paths (AKA, [input_path, output_path] )
    
    Note: Markers are assumed to be always in a given constant "depth"
    """    
    # Initialize root_batch_path only once
    if _root_batch_path is None:
        _root_batch_path = batch_path

    main_dir = str(pathlib.Path(_root_batch_path).parent)
    
    # Recursively list files and directories up to a certain depth
    depth -= 1
    with os.scandir(batch_path) as input_data_folder:
        for entry in input_data_folder:
            ##print(entry, depth)
            if entry.is_dir(): 
                
                # replace the prefix of the full path 
                output_folder = entry.path.replace(main_dir, output_dir)
                #logging.info(f"\n\nXXXX batch:{batch_path}, entry: {entry}, depth: {depth} main_dir:{main_dir} output_folder:{output_folder}" )
                
            # if that's not a marker directory, recursion...
            if entry.is_dir() and depth > 0:
                yield from find_marker_folders(entry.path, output_dir, depth, markers_to_include, _root_batch_path)
            
            # if that's a marker directory
            elif (depth==0) and (entry.is_dir()):
                marker_name = os.path.basename(entry.path)
                #skip nucleus 
                if marker_name=='DAPI':
                    continue
                
                #if analyzing specific marker, skip all but that one
                elif len(markers_to_include)==0 or (marker_name in markers_to_include):
                    logging.info(f'Running {marker_name} analysis')
                    yield [entry.path, output_folder]   


# def analyze_marker_with_filtering(input_and_output_path_list, pipeline_path, dataset_name='new_dNLS'):
#     """
#     """

#     global pipeline
#     pipeline = init_cell_profiler(pipeline_path)
    
#     input_data_dir, output_folder = input_and_output_path_list[0], input_and_output_path_list[1]
#     logging.info(f"Analyzing marker: reading input data from {input_data_dir}, saving results in {output_folder}")

#     set_marker_output_folder(path=output_folder)

#     image_files = collect_image_names_per_marker(input_data_dir, dataset_name)
    
#     filtered_image_files = filter_images(image_files, dataset_name)
    
#     image_files = paths_as_uri(filtered_image_files)
    
#     extract_cell_profilers_features(image_files, pipeline)
    
#     return f"\n\nFinished extracting features for {input_data_dir}"
