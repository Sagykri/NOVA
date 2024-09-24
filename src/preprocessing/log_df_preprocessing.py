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
                                "site_cell_count",
                                "cells_counts", "cells_count_mean", "cells_count_std",
                                "whole_cells_counts", "whole_cells_count_mean", "whole_cells_count_std",
                                'valid_tiles_indexes',
                                "n_valid_tiles", 
                                "cells_count_in_valid_tiles_mean", "cells_count_in_valid_tiles_std",
                                "whole_cells_count_in_valid_tiles_mean", "whole_cells_count_in_valid_tiles_std"],
                        filename_prefix="cell_count_stats")

    def log_nucleus(self, nuclei_mask_tiled:List[np.ndarray], valid_tiles_indexes:List[int], nucleus_path:str)->None:
        """Log the nucleus site preprocessing information

        Args:
            nuclei_mask_tiled (List[np.ndarray]): List of the masked tiles, to be used for nuclei counts
            valid_tiles_indexes (List[int]): List of the valid tiles
            nucleus_path (str): Path of the current nucleus site
        """
        nucleus_to_log = [path_utils.get_filename(nucleus_path), path_utils.get_raw_batch(nucleus_path), path_utils.get_raw_cell_line(nucleus_path), path_utils.get_raw_panel(nucleus_path),
                        path_utils.get_raw_condition(nucleus_path), path_utils.get_raw_rep(nucleus_path), path_utils.get_raw_marker(nucleus_path)]

        site_cell_count = max([np.max(masked_tiles) for masked_tiles in nuclei_mask_tiled])

        # logging counts
        n_valid_tiles = len(valid_tiles_indexes)
        n_cells_per_tile = np.asarray([get_nuclei_count(masked_tile) for masked_tile in nuclei_mask_tiled])
        n_whole_cells_per_tile = np.asarray([get_whole_nuclei_count(masked_tile=masked_tile) for masked_tile in nuclei_mask_tiled])
        
        nucleus_to_log += [site_cell_count,
                    n_cells_per_tile, round(np.mean(n_cells_per_tile), 2), round(np.std(n_cells_per_tile), 2),
                    n_whole_cells_per_tile, round(np.mean(n_whole_cells_per_tile), 2), round(np.std(n_whole_cells_per_tile), 2),
                    valid_tiles_indexes, 
                    n_valid_tiles]
        if n_valid_tiles > 0:
            nucleus_to_log += [round(np.mean(n_cells_per_tile[valid_tiles_indexes]), 2),
                        round(np.std(n_cells_per_tile[valid_tiles_indexes]), 2),
                        round(np.mean(n_whole_cells_per_tile[valid_tiles_indexes]), 2),
                        round(np.std(n_whole_cells_per_tile[valid_tiles_indexes]), 2)]
        else:
            nucleus_to_log += [None]*4
        
        self.write(nucleus_to_log)

    def log_marker(self, valid_tiles_indexes:List[int], marker_path:str):
        """Log the nucleus site preprocessing information

        Args:
            valid_tiles_indexes (List[int]): List of the valid tiles
            marker_path (str): Path of the current marker site
        """
        marker_to_log = [path_utils.get_filename(marker_path), path_utils.get_raw_batch(marker_path), path_utils.get_raw_cell_line(marker_path), path_utils.get_raw_panel(marker_path),
                        path_utils.get_raw_condition(marker_path), path_utils.get_raw_rep(marker_path), path_utils.get_raw_marker(marker_path)]
            
        marker_to_log += [None, None, None, None, None, None, None,
                valid_tiles_indexes, 
                len(valid_tiles_indexes),
                None, None, None, None]
        self.write(marker_to_log)