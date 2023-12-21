import datetime
from itertools import repeat
import multiprocessing
import os
import sys
import timeit
import torch
import pandas as pd
import cv2
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


import glob
import logging

import numpy as np
from skimage import io
from cellpose import models

from src.common.lib.preprocessor import Preprocessor
from src.common.lib import preprocessing_utils
from src.common.lib.utils import LogDF, get_if_exists
from src.preprocessing.configs.preprocessor_spd_config import SPDPreprocessingConfig

class SPDPreprocessor(Preprocessor):
    """
    Preprocessor for preprocessing images captured by the spinning disk
    """
    def __init__(self, conf: SPDPreprocessingConfig):
        super().__init__(conf)
        
        self.to_show = get_if_exists(conf, 'TO_SHOW')
        self.markers_to_include = get_if_exists(conf, 'MARKERS_TO_INCLUDE')
        self.nucleus_diameter = get_if_exists(conf, 'NUCLEUS_DIAMETER')
        self.tile_width = get_if_exists(conf, 'TILE_WIDTH')
        self.tile_height = get_if_exists(conf, 'TILE_HEIGHT')
        self.to_downsample = get_if_exists(conf, 'TO_DOWNSAMPLE')
        self.to_normalize = get_if_exists(conf, 'TO_NORMALIZE')
        self.cellprob_threshold = get_if_exists(conf, 'CELLPROB_THRESHOLD')
        self.flow_threshold = get_if_exists(conf, 'FLOW_THRESHOLD')
        self.to_denoise = get_if_exists(conf, 'TO_DENOISE')
        self.cell_lines_to_include = get_if_exists(conf, 'CELL_LINES_TO_INCLUDE')
        self.conf = conf
        
        if self.conf.SELECTIVE_INPUT_PATHS is not None:
            logging.info(f"SELECTIVE_INPUT_PATHS is ON. Processing only these files: {self.conf.SELECTIVE_INPUT_PATHS}")
            
        self.brenner_bounds_path = get_if_exists(conf, 'BRENNER_BOUNDS_PATH')
        self.brenner_bounds = None
        if self.brenner_bounds_path is not None:
            logging.info(f"Brenner bounds have been detected: {self.brenner_bounds_path}. Loading the file...")
            self.brenner_bounds = pd.read_csv(self.brenner_bounds_path, index_col=0)
            
        if self.markers_to_include is not None:
            logging.info(f"Markers to include = {self.markers_to_include}")
    
    
    def preprocess_images(self, **kwargs):
        """
        Preprocess the images inside the input folders specified in the config file.
        This preprocessing is suitable for handling the spinning disk images
        """
        
        logging.info(f"Is GPU available: {torch.cuda.is_available()}")
        cp_model = models.Cellpose(gpu=True, model_type='nuclei')

        logging_df = LogDF(self.conf.LOGS_FOLDER, 
                           columns=["DATETIME", "filename", "batch", "cell_line", "panel",
                                    "condition", "rep", "marker",
                                    "cells_counts", 
                                    
                                    'valid_tiles_indexes', # SAGY 201123
                                    
                                    "cells_count_mean", "cells_count_std",
                                    "whole_cells_counts", "whole_cells_count_mean", "whole_cells_count_std",
                                    "n_valid_tiles", 
                                    "site_cell_count",
                                    "cells_count_in_valid_tiles_mean", "cells_count_in_valid_tiles_std",
                                    "whole_cells_count_in_valid_tiles_mean", "whole_cells_count_in_valid_tiles_std"],
                           filename_prefix="cell_count_stats_")
        
        timing_df = LogDF(self.conf.LOGS_FOLDER, 
                           columns=["DATETIME", "batch", "cell_line", "panel",
                                    "condition", "rep", "time_seconds"],
                           filename_prefix="timing_")


        for input_folder_root, output_folder_root in zip(self.input_folders, self.output_folders):
            raw_f = os.path.basename(input_folder_root)
            
            logging.info(f"[{raw_f}] Processing folder")
            
            if not os.path.isdir(input_folder_root):
                logging.info(f"[{raw_f}] Skipping non-folder")
                continue
            

            if self.cell_lines_to_include is None or len(self.cell_lines_to_include) == 0:
                logging.info("CELL_LINES_TO_INCLUDE doesn't exist in config. Taking all cell lines detected")
                cell_lines = [f for f in sorted(os.listdir(input_folder_root)) if os.path.isdir(os.path.join(input_folder_root, f))]
            else:
                cell_lines = self.cell_lines_to_include

            logging.info(f"[{raw_f}] Cell line detected: {cell_lines}")

            for cell_line in cell_lines:
                
                logging.info(f"[{raw_f} {cell_line}] Cell line: {cell_line}")
                
                input_folder_root_cell_line = os.path.join(input_folder_root, cell_line)
                
                panels = [f for f in sorted(os.listdir(input_folder_root_cell_line)) if os.path.isdir(os.path.join(input_folder_root_cell_line, f))]        
                
                logging.info(f"[{raw_f}, {cell_line}] Panels: {panels}")
                
                args = zip(repeat(self), panels, repeat(input_folder_root), repeat(output_folder_root), repeat(input_folder_root_cell_line),\
                            repeat(cp_model), repeat(raw_f), repeat(cell_line),\
                            repeat(logging_df), repeat(timing_df))
                
                # For running it sequentially
                # for p in panels:
                #    preprocessing_utils.preprocess_panel(self, p, input_folder_root, output_folder_root, input_folder_root_cell_line, 
                #                                            cp_model, raw_f, cell_line, logging_df, timing_df)
                # print("/n/n/n/n/n/n/nXXXXXXXXX For running it sequentially")
                with multiprocessing.Pool(len(panels)) as pool:
                   pool.starmap(preprocessing_utils.preprocess_panel, args)
                     
                        
    def preprocess_image(self, input_path, output_path, **kwargs):
        """Preprocess a single image

        Args:
            input_path (string): Path to the raw image
            output_path (string): Path to the output (preprocessed) image
        """
        
        file_path           = input_path
        save_path           = output_path
        nucleus_file        = get_if_exists(kwargs, 'nucleus_file')
        img_nucleus         = get_if_exists(kwargs, 'img_nucleus')
        tile_width          = self.tile_width
        tile_height         = self.tile_height
        to_downsample       = self.to_downsample
        to_normalize        = self.to_normalize
        to_denoise          = self.to_denoise
        to_show             = self.to_show
        tiles_indexes       = get_if_exists(kwargs, 'tiles_indexes')
        brenner_bounds      = self.brenner_bounds
        
        # Changing from skimage.load to cv2.load (with grayscale flag) -> changed to IMREAD_ANYDEPTH to read in 16bit format
        img_target = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH) #used to be IMREAD_GRAYSCALE
        
        if img_nucleus is None:
            img_nucleus = cv2.imread(nucleus_file, cv2.IMREAD_ANYDEPTH) #used to be IMREAD_GRAYSCALE
        
        # Check if files are corrputed
        if img_target is None or np.size(img_target) == 0:
            logging.warning(f"File {file_path} is corrupted. Skiping this one.")
            return
        if img_nucleus is None or np.size(img_nucleus) == 0:
            logging.warning(f"File {nucleus_file} is corrupted. Skiping this one.")
            return
        
        # Take nuclues and target channels so target is the first channel and nuclues is the second
        img = np.stack([img_target, img_nucleus], axis=2)

        logging.info(f"Processing {file_path}, {nucleus_file}... ({img.shape})")

        n_channels = img.shape[-1]
        
        logging.info(f"#Channels= {n_channels}")

        processed_images = preprocessing_utils.preprocess_image_pipeline(img, save_path, 
                                                                        tiles_indexes=tiles_indexes,
                                                                        n_channels=n_channels, 
                                                                        tile_width=tile_width, tile_height=tile_height, 
                                                                        to_downsample=to_downsample,
                                                                        to_denoise=to_denoise, 
                                                                        to_normalize=to_normalize, 
                                                                        to_show=to_show,
                                                                        brenner_bounds=brenner_bounds)
        
        return processed_images

    
