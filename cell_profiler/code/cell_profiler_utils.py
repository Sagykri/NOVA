"""
Preprocessing_utils 
To be able to import from other scripts under "cell_profiler"
"""
# Packages 
import fcntl
import os
import atexit
import logging
import javabridge

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
import cellprofiler_core
import cellprofiler_core.pipeline
import cellprofiler_core.preferences
from cellprofiler_core.utilities import java as cpjava

import saving_scaled_sites_post_brenner as site_post_brenner



run_cell_profiler_pipeline_HOME = os.getenv("NOVA_HOME")
BASE_DIR = os.path.join('/home','projects','hornsteinlab','Collaboration','NOVA')
sys.path.insert(1, BASE_DIR)

# =========================
# JVM + pipeline management
# =========================
_JAVA_STARTED = False
_PIPELINE = None


def _stop_java_safely():
    """Stop JVM if running (safe to call many times)."""
    global _JAVA_STARTED
    try:
        if javabridge.get_env() is not None:
            cpjava.stop_java()
    finally:
        _JAVA_STARTED = False

def _start_java_once(max_heap='2g'):
    """Start JVM once per process; tolerate different start_java signatures."""
    global _JAVA_STARTED
    if _JAVA_STARTED or javabridge.get_env() is not None:
        _JAVA_STARTED = True
        return

    # Must set headless before starting Java
    cellprofiler_core.preferences.set_headless()

    for kwargs in (
        {"max_heap_size": max_heap},  # common CP 4.x
        {"max_heap": max_heap},       # some builds
        {"heap_size": max_heap},      # older wrappers
        {},                           # fallback
    ):
        try:
            cpjava.start_java(**kwargs)
            atexit.register(_stop_java_safely)
            _JAVA_STARTED = True
            logging.info(f"Started JVM with kwargs={kwargs or 'default'}")
            return
        except TypeError:
            continue
        except Exception as e:
            logging.debug(f"cpjava.start_java(**{kwargs}) failed: {e}")

    raise RuntimeError("Failed to start Java VM (unknown start_java signature).")

def loading_cellprofiler_pipeline(pipeline_path: str):
    """Load a .cppipe into a Pipeline object."""
    logging.info(f"Loading CellProfiler pipeline: {pipeline_path}")
    p = cellprofiler_core.pipeline.Pipeline()
    p.load(pipeline_path)
    logging.info("Pipeline loaded.")
    return p

def init_cell_profiler(pipeline_path: str, max_heap='2g'):
    """
    Idempotent: starts JVM once and loads pipeline once.
    Safe to call many times from other functions.
    """
    global _PIPELINE
    _start_java_once(max_heap=max_heap)
    if _PIPELINE is None:
        _PIPELINE = loading_cellprofiler_pipeline(pipeline_path)
    return _PIPELINE


def set_logging(log_file_path, level=logging.INFO, format=' INFO: %(message)s'):
    formatter = '%(asctime)s %(levelname)-8s %(message)s'
    handlers = [logging.FileHandler(log_file_path + '.log'), logging.StreamHandler()]
    logging.basicConfig(level=level, format=formatter, handlers=handlers, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(__doc__)
    return None


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
    cellprofiler_core.preferences.set_temporary_directory(path)
                    
    return None

def collect_image_names_per_marker(input_data_dir, dataset_name=None):
    """
    Collect marker image paths together with their matched DAPI channel images.

    This function scans a given marker directory, gathers all `.tif/.tiff` files,
    and for each marker image locates its corresponding DAPI image in the sibling
    "DAPI" folder. Matching is based on dataset-specific filename conventions.

    Parameters
    ----------
    input_data_dir : str or pathlib.Path
        Path to a single marker folder (e.g. ".../batch1/panelE/Untreated/rep1/DCP1A").
        The function will look for a sibling "DAPI" directory at the same level.
    dataset_name : str, optional
        Dataset naming convention to use for matching:
        - If in {"OPERA_indi_sorted", "OPERA_dNLS_6_batches_NOVA_sorted"}:
          DAPI files are matched by prefix (e.g. "rXXcXXfXXX-ch1t1.tiff").
        - Otherwise ("Coyne" convention):
          DAPI files are matched by site suffix (the substring after the last "_").

    Returns
    -------
    list of pathlib.Path
        List of absolute file paths, alternating marker and DAPI images.
        Each marker file is immediately followed by its associated DAPI file.

    Notes
    -----
    - Assumes that a "DAPI" folder exists as a sibling to the marker folder.
    - Raises an IndexError if no DAPI file is found for a marker image.
    - File pairing strictly depends on filename patterns; no image content is checked.
    - Output ordering is marker1 → DAPI1 → marker2 → DAPI2 → ...

    Examples
    --------
     collect_image_names_per_marker(
         "/home/projects/.../batch1/panelE/Untreated/rep1/DCP1A",
         dataset_name="OPERA_dNLS_6_batches_NOVA_sorted"
     )
    [PosixPath('/.../DCP1A/r01c01f001-ch2t1.tiff'),
     PosixPath('/.../DAPI/r01c01f001-ch1t1.tiff'),
     PosixPath('/.../DCP1A/r01c01f002-ch2t1.tiff'),
     PosixPath('/.../DAPI/r01c01f002-ch1t1.tiff'),
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
            if dataset_name in ['OPERA_indi_sorted', 'OPERA_dNLS_6_batches_NOVA_sorted','TDP43_WT_OE_PB_experiment_sorted','Sorbitol_experiment_PBs_TDP43_sorted','zstack_collapse_2nd_imaging_sorted']:
                nucleus_filepath = filename.split("-")[0]+"-ch1t1.tiff"
                nucleus_filepath = glob(f"{nucleus_folder}/{nucleus_filepath}")[0]
            elif dataset_name in {'NIH', 'indi-image-pilot', 'indi-image-pilot-20241128'}:
                # marker and DAPI share the same basename (e.g., s19.tif)
                #filename = filename.split("_")[1]
                nucleus_filepath = os.path.join(nucleus_folder, f"{filename}{ext}")
                if not os.path.exists(nucleus_filepath):
                    logging.warning(f"NIH: missing DAPI for {file} at {nucleus_filepath}")
                    continue
            else:
                # Coyne
                site = filename.split('_')[-1]
                nucleus_filepath = glob(f"{nucleus_folder}/*_{site}{ext}")[0]
            file_list.append(pathlib.Path(nucleus_filepath))
                        
    return file_list

def collect_images_unbiased(input_data_dir, dataset_name=None):
    """
    Collect marker image paths together with their matched DAPI channel images.

    This function scans a given marker directory, gathers all `.tif/.tiff` files,
    and for each marker image locates its corresponding DAPI image in the sibling
    "DAPI" folder. Matching is based on dataset-specific filename conventions.

    Parameters
    ----------
    input_data_dir : str or pathlib.Path
        Path to a single marker folder (e.g. ".../batch1/panelE/Untreated/rep1/DCP1A").
        The function will look for a sibling "DAPI" directory at the same level.
    dataset_name : str, optional
        Dataset naming convention to use for matching:
        - If in {"OPERA_indi_sorted", "OPERA_dNLS_6_batches_NOVA_sorted"}:
          DAPI files are matched by prefix (e.g. "rXXcXXfXXX-ch1t1.tiff").
        - Otherwise ("Coyne" convention):
          DAPI files are matched by site suffix (the substring after the last "_").

    Returns
    -------
    list of pathlib.Path
        List of absolute file paths, alternating marker and DAPI images.
        Each marker file is immediately followed by its associated DAPI file.

    Notes
    -----
    - Assumes that a "DAPI" folder exists as a sibling to the marker folder.
    - Raises an IndexError if no DAPI file is found for a marker image.
    - File pairing strictly depends on filename patterns; no image content is checked.
    - Output ordering is marker1 → DAPI1 → marker2 → DAPI2 → ...

    Examples
    --------
     collect_image_names_per_marker(
         "/home/projects/.../batch1/panelE/Untreated/rep1/DCP1A",
         dataset_name="OPERA_dNLS_6_batches_NOVA_sorted"
     )
    [PosixPath('/.../DCP1A/r01c01f001-ch2t1.tiff'),
     PosixPath('/.../DAPI/r01c01f001-ch1t1.tiff'),
     PosixPath('/.../DCP1A/r01c01f002-ch2t1.tiff'),
     PosixPath('/.../DAPI/r01c01f002-ch1t1.tiff'),
     ...]
    """
    logging.info(f"collect_images_unbiased: {input_data_dir}")
    
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
            logging.info(f'marker filename path: {image_filename}')
            file_list.append(pathlib.Path(image_filename))    # CP requires absolute file paths
            
            nucleus_folder = os.path.join(rep_dir, "DAPI")
            # Find the file name of the associated DAPI site
            if dataset_name in ['OPERA_indi_sorted', 'OPERA_dNLS_6_batches_NOVA_sorted','TDP43_WT_OE_PB_experiment_sorted','Sorbitol_experiment_PBs_TDP43_sorted','zstack_collapse_2nd_imaging_sorted']:
                nucleus_filepath = filename.split("-")[0]+"-ch1t1.tiff"
                nucleus_filepath = glob(f"{nucleus_folder}/{nucleus_filepath}")[0]
            elif dataset_name in {'NIH', 'indi-image-pilot', 'indi-image-pilot-20241128'}:
                # marker and DAPI share the same site number but DAPI has ch1 and marker ch2 as prefix
                filename = filename.split("_")[1]
                nucleus_filepath = os.path.join(nucleus_folder, f"ch1_{filename}{ext}")
                logging.info(f"filename: {filename}, nucleus_filepath: {nucleus_filepath}")
                if not os.path.exists(nucleus_filepath):
                    logging.warning(f"NIH: missing DAPI for {file} at {nucleus_filepath}")
                    continue
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
                     
             
# def run_cell_profiler_pipeline(image_files, pipeline_path, dataset_name, dry_run = False, debug=False, max_sets=4):
#     """
#     Initialize CellProfiler pipeline, convert paths to URI, and run the pipeline to extract features.
#     """
#     global pipeline
#     pipeline = init_cell_profiler(pipeline_path)
    
#     logging.info(f"File list recived: {image_files} ")
    
#     # Ensure image_files is a list of paths
#     input_data_dir, output_folder = image_files[0], image_files[1]
#     logging.info(f"run_cell_profiler_pipeline: {image_files} input_data_dir: {input_data_dir}, output_folder: {output_folder}")
    
#     # Set the output folder for CellProfiler results
#     set_marker_output_folder(path=output_folder)
#     logging.info(f"set_marker_output_folder: saving results in {output_folder}")
    
#     # collect image file paths for a given marker and its associated DAPI channel
#     image_files_paths =  collect_image_names_per_marker(input_data_dir, dataset_name)
    
#     # --- DEBUG SLICE: keep only the first N (marker,DAPI) pairs ---
#     if debug:
#         keep = max_sets * 2  # each set is 2 files: marker, then its DAPI
#         if len(image_files_paths) > keep:
#             logging.info(f"DEBUG mode ON: limiting to first {max_sets} pairs "
#                          f"({keep} files) out of {len(image_files_paths)}")
#             image_files_paths = image_files_paths[:keep]
#         else:
#             logging.info(f"DEBUG mode ON: total files <= requested limit "
#                          f"({len(image_files_paths)} <= {keep}); using all.")

#     logging.info(f"Collected {len(image_files_paths)} image files: {image_files_paths}")
    
#     # Convert image file paths to URIs
#     image_files_uri = paths_as_uri(image_files_paths)

#     # # Run CellProfiler pipeline on images
#     if dry_run:
#         print("DRY RUN! skipping processing....")
#     else:
#         extract_cell_profilers_features(image_files_uri, pipeline)

#     return f"\n\nFinished extracting features for {input_data_dir}"

def run_cell_profiler_pipeline(image_files, pipeline_path, dataset_name, debug=False, unbiased = False, max_sets=4):
    """
    image_files = [input_dir, output_folder]
    Fresh pipeline per task + serialized export to prevent CSV mixing.
    """
    input_data_dir, output_folder = image_files[0], image_files[1]

    # Point CP to THIS task's output & temp dir
    set_marker_output_folder(path=output_folder)
    logging.info(
        "CP dirs | output=%s | temp=%s",
        cellprofiler_core.preferences.get_default_output_directory(),
        cellprofiler_core.preferences.get_temporary_directory()
    )

    # Start JVM if needed, but DO NOT reuse a cached pipeline in workers
    _start_java_once(max_heap='512m')
    pipeline = loading_cellprofiler_pipeline(pipeline_path)  # fresh per task

    # Build file list for this input and register with pipeline
    if unbiased:
        logging.info('running unbiased analysis - using collect_images_unbiased')
        image_files_paths = collect_images_unbiased(input_data_dir, dataset_name)
    else:
        logging.info('running standard analysis - using collect_image_names_per_marker')
        image_files_paths = collect_image_names_per_marker(input_data_dir, dataset_name)
    
    if debug:
        image_files_paths = image_files_paths[: max_sets]  # each set is 2 files: marker, then its DAPI
        logging.info(f"Test set size{len(image_files_paths)}")
    image_files_uri = paths_as_uri(image_files_paths)
    logging.info(f"Collected {len(image_files_paths)} images for {input_data_dir}")
    pipeline.read_file_list(image_files_uri)

    # Serialize ExportToSpreadsheet in this folder
    lock_path = os.path.join(output_folder, ".cp_export.lock")
    m = None
    with open(lock_path, "w") as _lf:
        fcntl.flock(_lf, fcntl.LOCK_EX)  # blocks if another worker is exporting here
        try:
            logging.info("Export lock ACQUIRED: %s", output_folder)
            m = pipeline.run()            # runs modules + ExportToSpreadsheet
            logging.info("pipeline.run() DONE: %s", input_data_dir)
        finally:
            try:
                if m is not None:
                    m.close()             # ensure HDF5 is closed even on errors
            except Exception:
                logging.exception("Failed to close Measurements")
            fcntl.flock(_lf, fcntl.LOCK_UN)
            logging.info("Export lock RELEASED: %s", output_folder)

    return f"\n\nFinished extracting features for {input_data_dir}"

   



def find_marker_folders(batch_path, output_dir, depth=4, 
                        markers_to_include=[], _root_batch_path=None,
                        include_nucleus=False):
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
                yield from find_marker_folders(entry.path, output_dir, depth, 
                                               markers_to_include, _root_batch_path,
                                               include_nucleus=include_nucleus)
            
            # if that's a marker directory
            elif (depth==0) and (entry.is_dir()):
                marker_name = os.path.basename(entry.path)
                #skip nucleus 
                if marker_name=='DAPI' and not include_nucleus:
                    continue
                
                #if analyzing specific marker, skip all but that one
                elif len(markers_to_include)==0 or (marker_name in markers_to_include):
                    logging.info(f'Running {marker_name} analysis')
                    yield [entry.path, output_folder]   




