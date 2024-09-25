import logging
import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from tools.images_organizer.opera_version2.utils import Utils
from src.common.utils import get_class

def main():
    config_path = sys.argv[1]
    print(config_path)
    
    config_class = get_class(config_path)
    config = config_class()
    utils_obj = Utils(config)
    
    n_copied = 0
    
    utils_obj.init_logging()
    # Asserts
    assert os.path.exists(utils_obj.config.SRC_ROOT_PATH) and os.path.isdir(utils_obj.config.SRC_ROOT_PATH), f"{utils_obj.config.SRC_ROOT_PATH} not exists (or not a folder)"
    assert os.path.exists(utils_obj.config.DST_ROOT_PATH) and os.path.isdir(utils_obj.config.DST_ROOT_PATH), f"{utils_obj.config.DST_ROOT_PATH} not exists (or not a folder)"
    
    print(f"CUT_FILES: {utils_obj.config.CUT_FILES}")
    
    folders = utils_obj.get_folders_to_handle()
    assert all([os.path.exists(os.path.join(utils_obj.config.SRC_ROOT_PATH, f)) and os.path.isdir(os.path.join(utils_obj.config.SRC_ROOT_PATH, f)) for f in folders]), "One or more of the specified folders don't exists (or aren't folders)"
    
    for folder in folders:
            folder_path, batch, panel = utils_obj.init_folders(folder)#, replace_wells=copy_well_folder)
            
            # Copy files to dist
            n_copied += utils_obj.copy_files(folder_path, panel, batch, cut_files=utils_obj.config.CUT_FILES,
                                             raise_on_missing_index=utils_obj.config.RAISE_ON_MISSING_INDEX)#, replace_wells=copy_well_folder, wells=wells)
        
    # Get expected number of files to copy
    try:
        n_total = utils_obj.get_expected_number_of_files_to_copy()
        n_total_str = f"/{n_total}"
    except Exception as e:
        logging.warning(f"Couldn't calculate expected number of files to copy (Linux is needed) \n {e}", exc_info=True)
        n_total_str = ""
    
    logging.info(f"{n_copied}{n_total_str} files were copied from {utils_obj.config.SRC_ROOT_PATH} to {utils_obj.config.DST_ROOT_PATH}")

logging.info(f"Finished successfully!")
    
if __name__ == "__main__":
    main()