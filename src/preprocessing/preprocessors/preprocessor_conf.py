import logging
import os
from common.lib.preprocessor import Preprocessor
from common.lib.utils import get_if_exists
from common.lib import preprocessing_utils
from preprocessing.configs.preprocessor_conf_config import ConfPreprocessingConfig
import torch
import numpy as np
from skimage import io


class ConfPreprocessor(Preprocessor):
    """
    Preprocessor for preprocessing images captured by the confocal microscope
    """
    def __init__(self, conf: ConfPreprocessingConfig):
        super(ConfPreprocessor, self).__init__(conf)
        
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
    
    @staticmethod
    def __get_markers_part(filename):
      splits = filename.split('-')
      for i in range(len(splits)):
        if splits[i].startswith('DAPI'):
          return splits[i]
    
    
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
                    logging.info(f"[{raw_f}] Cell line: {cell_line}")
                    
                    input_folder_root_cell_line = os.path.join(input_folder_root, cell_line)
                    
                    conditions = [f for f in os.listdir(input_folder_root_cell_line) if os.path.isdir(os.path.join(input_folder_root_cell_line, f))]        
                    
                    logging.info(f"[{raw_f}] Conditions: {conditions}")
                    
                    input_folders = [os.path.join(input_folder_root, cell_line, c) for c in conditions] 
                    output_folders = [os.path.join(output_folder_root, cell_line, c) for c in conditions] 

                    logging.info(f"Input folders: {input_folders}")

                    format_output_filename = lambda filename, ext: f"{filename}_{cell_line}{ext}"
                    # preprocess_images(input_folders, output_folders,\
                    #                 format_output_filename=format_output_filename,\
                    #                 nucleus_channel=-1)
                    
                    # self.preprocess_images(input_folders, output_folders,\
                    #                 format_output_filename=format_output_filename,\
                    #                 nucleus_channel=-1)
                    
                
                
                    for input_folder, output_folder in zip(input_folders, output_folders):
                        for f in os.listdir(input_folder):
                            filename, ext = os.path.splitext(f)
                            if ext != '.tif':
                                continue
                            
                            filename_path = os.path.join(input_folder,filename)
                            logging.info(filename_path)
                            
                            # Extract markers from filename
                            
                            markers_str = ConfPreprocessor.__get_markers_part(filename)
                            markers = markers_str.split('_')
                            #markers.remove('DAPI')
                            # Flip the order
                            markers = markers[::-1]
                            markers = [(i, m) for i, m in enumerate(markers)]
                            
                            output_filename = f
                            if format_output_filename:
                                # Hotfix for getting the output folder name without breaking the signature of the function
                                if format_output_filename.__code__.co_argcount == 3: 
                                    output_filename = format_output_filename(filename, ext, output_folder)
                                else:
                                    output_filename = format_output_filename(filename, ext)
                                
                            for c, d in markers:
                                if self.markers_to_include is not None and d not in self.markers_to_include:
                                    logging.info(f"Skipping {d}")
                                    continue
                            
                                output_subfolder = os.path.join(output_folder, d)
                                save_path = os.path.join(output_subfolder, output_filename)
                            
                                if os.path.exists(f"{save_path}_processed"):
                                    logging.info(f"[Skipping ,exists] Already exists {save_path}_processed")
                                    continue
                            
                                logging.info(output_subfolder)
                                if not os.path.exists(output_subfolder):
                                    os.makedirs(output_subfolder)

                                self.preprocess_image(filename_path, save_path,
                                                      target_channel=c, show=False,
                                                      flow_threshold=self.flow_threshold)
                



    def preprocess_image(self, input_path, output_path, **kwargs):
        """Preprocess a single image

        Args:
            input_path (string): Path to the raw image
            output_path (string): Path to the output (preprocessed) image
        """
    
        input_path          = input_path
        save_path           = output_path
        target_channel      = get_if_exists(kwargs, 'target_channel')
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
    
        img = io.imread(input_path)
        channel_axis = np.argmin(img.shape)
        nucleus_channel = -1
        img = np.moveaxis(img, channel_axis, nucleus_channel)

        # Take nuclues and target channels so target is the first channel and nuclues is the second
        img = img[...,[target_channel, nucleus_channel]]

        logging.info(f"Processing {input_path}... ({img.shape})", flush=True)

        n_channels = img.shape[-1]
        
        logging.info(f"#Channels= {n_channels}", flush=True)


        processed_images = preprocessing_utils.preprocess_image_pipeline(input_path, save_path, n_channels=n_channels, nucleus_diameter=nucleus_diameter,
                              flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, min_edge_distance=min_edge_distance,
                              tile_width=tile_width, tile_height=tile_height, to_downsample=to_downsample,
                              to_denoise=to_denoise, to_normalize=to_normalize, to_show=to_show)
        
        return processed_images