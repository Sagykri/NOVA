import logging
import os
import pandas as pd
import numpy as np
from config_microglia_LPS import DST_ROOT_PATH, SRC_ROOT_PATH, CUT_FILES

from utils import get_folders_to_handle, init_logging

def main():    
    init_logging()
    # Asserts
    assert os.path.exists(SRC_ROOT_PATH) and os.path.isdir(SRC_ROOT_PATH), f"{SRC_ROOT_PATH} not exists (or not a folder)"
    assert os.path.exists(DST_ROOT_PATH) and os.path.isdir(DST_ROOT_PATH), f"{DST_ROOT_PATH} not exists (or not a folder)"
    
    folders = get_folders_to_handle()
    assert all([os.path.exists(os.path.join(SRC_ROOT_PATH, f)) and os.path.isdir(os.path.join(SRC_ROOT_PATH, f)) for f in folders]), "One or more of the specified folders don't exists (or aren't folders)"
    
    for folder in folders:
        folder_split = folder.split("_")
        folder_split[1] = folder_split[1].replace('plate','batch')
        remove_plate_no = "_".join(folder_split[0:2]+folder_split[3:])
        folder_path = os.path.join(SRC_ROOT_PATH, folder)
        new_folder_path = os.path.join(SRC_ROOT_PATH, remove_plate_no)
        os.rename(folder_path,new_folder_path)

    logging.info(f"Finished successfully!")
    
    
if __name__ == "__main__":
    main()