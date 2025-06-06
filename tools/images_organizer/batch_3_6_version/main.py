import logging
import os
import pandas as pd
import numpy as np
from config import DST_ROOT_PATH, SRC_ROOT_PATH

from utils import copy_files, get_expected_number_of_files_to_copy, get_folders_to_handle, init_folders, init_logging

def main():
    n_copied = 0
    
    init_logging()
    
    # Asserts
    assert os.path.exists(SRC_ROOT_PATH) and os.path.isdir(SRC_ROOT_PATH), f"{SRC_ROOT_PATH} not exists (or not a folder)"
    assert os.path.exists(DST_ROOT_PATH) and os.path.isdir(DST_ROOT_PATH), f"{DST_ROOT_PATH} not exists (or not a folder)"
    
    folders = get_folders_to_handle()
    assert all([os.path.exists(os.path.join(SRC_ROOT_PATH, f)) and os.path.isdir(os.path.join(SRC_ROOT_PATH, f)) for f in folders]), "One or more of the specified folders don't exists (or aren't folders)"
    
    
    for folder in folders:
        # Init
        folder_path, batch, rep, panel = init_folders(folder)
        
        # Copy files to dist
        n_copied += copy_files(folder_path, panel, batch, rep)
        
    # Get expected number of files to copy    
    try:
        n_total = get_expected_number_of_files_to_copy()
        n_total_str = f"/{n_total}"
    except Exception as e:
        logging.warning(f"Couldn't calculate expected number of files to copy (Linux is needed) \n {e}", exc_info=True)
        n_total_str = ""
        
    logging.info(f"{n_copied}{n_total_str} files were copied from {SRC_ROOT_PATH} to {DST_ROOT_PATH}")
    logging.info(f"Finished successfully!")
    
    
if __name__ == "__main__":
    main()