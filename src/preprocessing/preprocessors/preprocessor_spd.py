import glob
import logging
import os

import numpy as np

import torch
from skimage import io

from common.lib.preprocessor import Preprocessor
from common.lib import preprocessing_utils
from common.lib.utils import get_if_exists
from preprocessing.configs.preprocessor_spd_config import SPDPreprocessingConfig


class SPDPreprocessor(Preprocessor):
    """
    Preprocessor for preprocessing images captured by the spinning disk
    """
    def __init__(self, conf: SPDPreprocessingConfig):
        super(SPDPreprocessor, self).__init__(conf)
        
        self.to_show = get_if_exists(conf, conf.TO_SHOW)
        self.markers_to_include = get_if_exists(conf, conf.MARKERS_TO_INCLUDE)
        self.nucleus_diameter = get_if_exists(conf, conf.NUCLEUS_DIAMETER)
        self.tile_width = get_if_exists(conf, conf.TILE_WIDTH)
        self.tile_height = get_if_exists(conf, conf.TILE_HEIGHT)
        self.to_downsample = get_if_exists(conf, conf.TO_DOWNSAMPLE)
        self.to_normalize = get_if_exists(conf, conf.TO_NORMALIZE)
        self.cellprob_threshold = get_if_exists(conf, conf.CELL_PROB_THRESHOLD)
        self.flow_threshold = get_if_exists(conf, conf.FLOW_THRESHOLD)
        self.min_edge_distance = get_if_exists(conf, conf.MIN_EDGE_DISTANCE)
        self.to_denoise = get_if_exists(conf, conf.TO_DENOISE)
        
    
    def preprocess_images(self, **kwargs):
        """
        Preprocess the images inside the input folders specified in the config file.
        This preprocessing is suitable for handling the spinning disk images
        """
        
        logging.info(f"Is GPU available: {torch.cuda.is_available()}")


        for input_folder_root, output_folder_root in zip(self.input_folders, self.output_folders):
            raw_f = os.path.basename(input_folder_root)
            
            logging.info(f"[{raw_f}] Processing folder")
            
            if not os.path.isdir(input_folder_root):
                logging.info(f"[{raw_f}] Skipping non-folder")
                continue
            
            cell_lines = [f for f in os.listdir(input_folder_root) if os.path.isdir(os.path.join(input_folder_root, f))]

            logging.info(f"[{raw_f}] Cell line detected: {cell_lines}")

            for cell_line in cell_lines:
                
                logging.info(f"[{raw_f} {cell_line}] Cell line: {cell_line}")
                
                input_folder_root_cell_line = os.path.join(input_folder_root, cell_line)
                
                panels = [f for f in os.listdir(input_folder_root_cell_line) if os.path.isdir(os.path.join(input_folder_root_cell_line, f))]        
                
                logging.info(f"[{raw_f}, {cell_line}] Panels: {panels}")
                
                for panel in panels:
                    logging.info(f"[{raw_f} {cell_line} {panel}] Panel: {panel}")
                    
                    input_folder_root_panel = os.path.join(input_folder_root_cell_line, panel)
                    
                    conditions = [f for f in os.listdir(input_folder_root_panel) 
                                if os.path.isdir(os.path.join(input_folder_root_panel, f)) and f != 'experiment setup']   
                        
                    logging.info(f"[{raw_f} {cell_line} {panel}] Conditions: {conditions}")
                    
                    
                    for condition in conditions:
                        logging.info(f"[{raw_f} {cell_line} {panel} {condition}] Condition: {condition}")
                    
                        input_folder_root_condition = os.path.join(input_folder_root_panel, condition)
                        
                        reps = [f for f in os.listdir(input_folder_root_condition ) if os.path.isdir(os.path.join(input_folder_root_condition , f))]

                        input_folders = [os.path.join(input_folder_root, cell_line, panel, condition, rep) for rep in reps]     
                        output_folders = [os.path.join(output_folder_root, cell_line, condition) for rep in reps]
                        
                        logging.info(f"Input folders: {input_folders}")

                        format_output_filename = lambda filename, ext: f"{filename}_{panel}_{cell_line}{ext}"
                
                        for input_folder, output_folder in zip(input_folders, output_folders):
                            markers = os.listdir(input_folder)
                            panel = os.path.basename(input_folder)
                            nucleus_folder = os.path.join(input_folder, "DAPI")
                            
                            for marker in markers:
                                if self.markers_to_include is not None and marker not in self.markers_to_include:
                                    logging.info(f"Skipping {marker}")
                                    continue
                                        
                                input_subfolder = os.path.join(input_folder, marker)
                                output_subfolder = os.path.join(output_folder, marker)
                                
                                logging.info(f"Subfolder {input_subfolder}")
                                
                                
                                for f in os.listdir(input_subfolder):
                                    filename, ext = os.path.splitext(f)
                                    if ext != '.tif':
                                        continue
                                    
                                    site = filename.split('_')[-1]
                                    target_filepath = os.path.join(input_subfolder, f)
                                    
                                    nucleus_filepath = glob.glob(f"{nucleus_folder}/*_{site}{ext}")
                                    if len(nucleus_filepath) == 0:
                                        logging.info(f"Skipping site {site} for {target_filepath} since no DAPI for this site was found")
                                        continue
                                    
                                    nucleus_filepath = nucleus_filepath[0]
                                    logging.info(f"{target_filepath}, {nucleus_filepath}")
                                    
                                    
                                    
                                    output_filename = format_output_filename(filename, ext) if format_output_filename else f
                                    
                                    save_path = os.path.join(output_subfolder, f"{panel}_{output_filename}")
                                    
                                    logging.info(f"Save path: {save_path}")
                                
                                    if os.path.exists(f"{save_path}_processed"):
                                        logging.info(f"[Skipping ,exists] Already exists {save_path}_processed")
                                        continue
                                
                                    logging.info(output_subfolder)
                                    if not os.path.exists(output_subfolder):
                                        os.makedirs(output_subfolder)

                                    self.preprocess_image(target_filepath, save_path,
                                                          nucleus_file=nucleus_filepath,show=self.to_show,
                                                          flow_threshold=self.flow_threshold)
                    
                    
                        
    def preprocess_image(self, input_path, output_path, **kwargs):
        """Preprocess a single image

        Args:
            input_path (string): Path to the raw image
            output_path (string): Path to the output (preprocessed) image
        """
        
        file_path           = input_path
        save_path           = output_path
        nucleus_file        = get_if_exists(kwargs, 'nucleus_file')
        nucleus_diameter    = self.nucleus_diameter
        tile_width          = self.tile_width
        tile_height         = self.tile_height
        to_downsample       = self.to_downsample
        to_normalize        = self.to_normalize
        cellprob_threshold  = self.cellprob_threshold
        flow_threshold      = self.flow_threshold
        min_edge_distance   = self.min_edge_distance
        to_denoise          = self.to_denoise
        to_show             = self.to_show
        
        
        img_target = io.imread(file_path)
        # channel_axis = img.shape.index(min(img.shape)) if channel_axis is None else channel_axis # TODO: this is new!
        # img = np.moveaxis(img, channel_axis,-1)
        img_nucleus = io.imread(nucleus_file)
        
        # Take nuclues and target channels so target is the first channel and nuclues is the second
        # img = img[...,[target_channel, nucleus_channel]]

        img = np.stack([img_target, img_nucleus], axis=2)

        logging.info(f"Processing {file_path}, {nucleus_file}... ({img.shape})", flush=True)

        n_channels = img.shape[-1]
        
        logging.info(f"#Channels= {n_channels}", flush=True)

        processed_images = preprocessing_utils.preprocess_image_pipeline(file_path, save_path, n_channels=n_channels, nucleus_diameter=nucleus_diameter,
                              flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, min_edge_distance=min_edge_distance,
                              tile_width=tile_width, tile_height=tile_height, to_downsample=to_downsample,
                              to_denoise=to_denoise, to_normalize=to_normalize, to_show=to_show)
        
        return processed_images

