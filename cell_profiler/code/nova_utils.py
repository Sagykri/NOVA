import sys
import os
sys.path.insert(1, os.getenv("NOVA_HOME"))

# Packages 
from contextlib import contextmanager
from datetime import datetime
from glob import glob 
import pandas as pd
import numpy as np
import logging
import pathlib
import cv2

from src.preprocessing import path_utils
from src.preprocessing.preprocessing_utils import rescale_intensity, is_image_focused, fit_image_shape


# Context manager to temporarily suppress logging below a given level
@contextmanager
def suppress_logging(level=logging.WARNING):
    loggers = [logging.getLogger()]  # Start with root
    loggers += [logging.getLogger(name) for name in logging.root.manager.loggerDict]

    previous_levels = {logger: logger.level for logger in loggers}

    for logger in loggers:
        logger.setLevel(level)

    try:
        yield
    finally:
        for logger, prev_level in previous_levels.items():
            logger.setLevel(prev_level)


def cp_get_valid_site_image(path, markers_focus_boundries):
    """
    Load and preprocess the image from the given path.

    Args:
        path (str): Path to the image file.

    Returns:
        Union[np.ndarray , None]: Preprocessed image. Returns None if file is corrupted.
    """
    
    image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    
    # Check if file is corrupted
    if image is None or np.size(image) == 0:
        logging.warning(f"File {path} is corrupted. Skiping this one.")
        return None
    
    with suppress_logging(level=logging.CRITICAL):    
        image = fit_image_shape(image, (1024,1024))  
        image = rescale_intensity(image, lower_bound=0.5, upper_bound=99.9) 
        
        if markers_focus_boundries is not None:
            # Filter out-of-focus images
            marker = path_utils.get_raw_marker(path)
            thresholds = tuple(markers_focus_boundries.loc[marker].values)
            if is_image_focused(image, thresholds): 
                return image, path
            else:
                return None, None
            
            
def cp_load_markers_focus_boundries(dataset_name):

    if dataset_name=='OPERA_indi_sorted':
        MARKERS_FOCUS_BOUNDRIES_PATH = os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_newINDI_allBatches.csv')
    elif dataset_name=='OPERA_dNLS_6_batches_NOVA_sorted':
        MARKERS_FOCUS_BOUNDRIES_PATH =  os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_operadNLS.csv')
    
    if MARKERS_FOCUS_BOUNDRIES_PATH:
        logging.info(f"\n\nLoading Markers Focus Boundries from: {MARKERS_FOCUS_BOUNDRIES_PATH}")
        markers_focus_boundries = pd.read_csv(MARKERS_FOCUS_BOUNDRIES_PATH, index_col=0)
        return markers_focus_boundries