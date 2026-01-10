import os
import sys
from typing import List
import numpy as np

sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.preprocessing.preprocessing_utils import get_nuclei_count, get_whole_nuclei_count
from src.preprocessing import path_utils
from src.common.log_df import LogDF

class LogDFPreprocessing(LogDF):
    def __init__(self, path:str):
        super().__init__(path, 
                        columns=["filename", "batch", "cell_line", "panel",
                                "condition", "rep", "marker",
                                "dead_cells_ncomponent","dead_cells_more_than_one_dead_cell","dead_cells_cloud_like", "dead_cells_whale_like","dead_cells_no_alive_cells",
                                "DAPI_invalid_max_intensity_above_threshold","DAPI_invalid_max_intensity_below_threshold", "DAPI_invalid_variance_below_threshold","DAPI_invalid_variance_above_threshold", "DAPI_out_of_focus_below_threshold","DAPI_out_of_focus_above_threshold",
                                "MARKER_invalid_max_intensity_above_threshold","MARKER_invalid_max_intensity_below_threshold", "MARKER_invalid_variance_below_threshold","MARKER_invalid_variance_above_threshold", "MARKER_out_of_focus_below_threshold","MARKER_out_of_focus_above_threshold"],
                        filename_prefix="cell_failed_reasons_stats")

    def log_nucleus(self, nuclei_mask_tiled:List[np.ndarray], valid_tiles_indexes:List[int], nucleus_path:str)->None:
        """Log the nucleus site preprocessing information

        Args:
            nuclei_mask_tiled (List[np.ndarray]): List of the masked tiles, to be used for nuclei counts
            valid_tiles_indexes (List[int]): List of the valid tiles
            nucleus_path (str): Path of the current nucleus site
        """
        
        
        self.write(nucleus_to_log)

