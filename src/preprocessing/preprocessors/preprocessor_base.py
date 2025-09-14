from collections import defaultdict
import logging
from multiprocessing import Pool
import os
from pathlib import Path
import re
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Tuple
import cv2
import numpy as np
from cellpose import models
import pandas as pd
from shapely import affinity , make_valid
from shapely.geometry import box ,Polygon

from skimage.filters import threshold_otsu
from scipy.ndimage import label


from skimage import transform

sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.common.utils import filter_paths_by_substrings, flat_list_of_lists, get_if_exists
from src.preprocessing.log_df_preprocessing import LogDFPreprocessing
from src.preprocessing import path_utils
from src.preprocessing.preprocessing_utils import get_nuclei_count, crop_image_to_tiles, extract_polygons_from_mask,\
                                                      fit_image_shape, get_nuclei_segmentations, is_image_focused,\
                                                      is_contains_whole_nucleus, rescale_intensity, \
                                                      is_tile_focused
from src.preprocessing.preprocessing_config import PreprocessingConfig

class Preprocessor(ABC):
    def __init__(self, preprocessing_config: PreprocessingConfig):
        self.preprocessing_config = preprocessing_config
        # The depth of the folder holding the markers folders
        self.__SUPPORTED_EXTENSIONS = ['tiff', 'tif']
        self.__NUCLEUS_MARKER_NAME = "DAPI"
        
        self.cellpose_model = models.Cellpose(gpu=True, model_type='nuclei')
        
        self.markers_focus_boundries_path = get_if_exists(self.preprocessing_config, 'MARKERS_FOCUS_BOUNDRIES_PATH', None)
        self.markers_focus_boundries = None
        if self.markers_focus_boundries_path is None:
            logging.info(f"No file for focus boundires has been detected in the configuration. Skipping this check.")
        else:
            logging.info(f"Focus boundries file for markers has been detected: {self.markers_focus_boundries_path}. Loading the file...")
            self.markers_focus_boundries = pd.read_csv(self.markers_focus_boundries_path, index_col=0)
        
        # ADDED TILE BERNNER - FROM GAL
        self.markers_focus_boundries_tiles_path = get_if_exists(self.preprocessing_config, 'MARKERS_FOCUS_BOUNDRIES_TILES_PATH', None)
        self.markers_focus_boundries_tiles = None
        if self.markers_focus_boundries_tiles_path is None:
            logging.info(f"No file for focus boundires for tiles has been detected in the configuration. Skipping this check.")
        else:
            logging.info(f"Focus boundries file for markers's tiles has been detected: {self.markers_focus_boundries_tiles_path}. Loading the file...")
            self.markers_focus_boundries_tiles = pd.read_csv(self.markers_focus_boundries_tiles_path, index_col=0)

        self.logging_df = LogDFPreprocessing(self.preprocessing_config.LOGS_FOLDER)

    @staticmethod
    def raw2processed_path(refpath:Union[str, Path])->str:
        """Converting raw path to processed path

        Args:
            refpath (Union[str, Path]): Relative raw path to the base dir (starting from the cell line folder)

        Returns:
            str: Relative processed path (starting for the cell line folder)
            
        Example:
            >>> p = "WT/panelD/Untreated/rep2/PSD95/a.tiff"
            >>> Preprocessor.raw2processed_path(p)
            WT/Untreated/CLTC/rep2_a_panelD_WT_processed.npy
        """
        if type(refpath) is not Path:
            refpath = Path(refpath)
            
        celline, panel, condition, rep, marker, filename = path_utils.get_raw_cell_line(refpath),\
                                                path_utils.get_raw_panel(refpath),\
                                                path_utils.get_raw_condition(refpath),\
                                                path_utils.get_raw_rep(refpath),\
                                                path_utils.get_raw_marker(refpath),\
                                                path_utils.get_filename(refpath)
        
        filename = f"{rep}_{filename}_{panel}_{celline}_processed.npy"
        
        return os.path.join(celline, condition, marker, filename)
    
    @staticmethod
    def processed2raw_path(refpath:Union[str, Path])->str:
        """Converting processed path to raw path

        Args:
            refpath (Union[str, Path]): Relative processed path to the base dir (starting from the cell line folder)

        Returns:
            str: Relative raw path (starting for the cell line folder)
            
        Example:
            >>> p = "WT/Untreated/CLTC/rep2_a_panelD_WT_processed.npy"
            >>> Preprocessor.processed2raw_path(p)
            WT/panelD/Untreated/rep2/PSD95/a.tiff
        """
        if type(refpath) is not Path:
            refpath = Path(refpath)
            
        celline, panel, condition, rep, marker, filename = path_utils.get_processed_cell_line(refpath),\
                                                path_utils.get_processed_panel(refpath),\
                                                path_utils.get_processed_condition(refpath),\
                                                path_utils.get_processed_rep(refpath),\
                                                path_utils.get_processed_marker(refpath),\
                                                path_utils.get_filename(refpath)
        
        filename = filename.replace(f"{rep}_", '').replace(f"{panel}_", "").replace(f"{celline}_", "").replace("_processed","")
        
        return os.path.join(celline, panel, condition, rep, marker, f"{filename}.tif")

    def preprocess(self, multiprocess=True) -> None:
            """
            Preprocess the images by filtering out-of-focus images, empty tiles, 
            and saving the paired images of marker and its nucleus.
            
            Args:
                multiprocess (bool, optional): Whether to run with multiprocessing. Default to True.
            """
            
            assert self.__is_nucleus_requested(), f"The nucleus marker ({self.__NUCLEUS_MARKER_NAME}) must be requested in the configuration files (via MARKERS)"
            
            for input_folder, output_folder in zip(self.preprocessing_config.INPUT_FOLDERS, self.preprocessing_config.PROCESSED_FOLDERS):
                
                logging.info(f"input folder: {input_folder}, output folder: {output_folder}")
                
                images_groups = self.__get_grouped_images_for_folder(input_folder)
                
                if multiprocess:
                    # Use multiprocessing to parallelize the image preprocessing
                    with Pool(self.preprocessing_config.NUM_WORKERS) as pool:
                            args = [t + (output_folder,) for t in images_groups.items()]
                            pool.starmap(self._process_images_group_and_save, args)
                else:
                    # Run sequentially
                    for group_id, images_group in images_groups.items():
                        self._process_images_group_and_save(group_id, images_group, output_folder)
                
    def preprocess_by_path(self, markers_paths: Union[str, List[str]])->Dict[str, np.ndarray] :
        """Process the files in the given paths

        Args:
            markers_paths (Union[str, List[str]]): Path(s) to a marker(s) file(s)

        Returns:
            Dict[str, np.ndarray]: 
                - Key: marker name
                - Value: The processed files 
        """
        if type(markers_paths) is str:
            markers_paths = [markers_paths]
        
        # Get the paths to the nucleus
        nuclues_paths = self.get_nucleus_filepaths_for_markers_paths(markers_paths)

        if len(nuclues_paths) == 0:
            logging.error("No nucleus files were found")
            raise Exception("No nucleus files were found")
        
        if len(nuclues_paths) != len(markers_paths):
            logging.warning("Mismatch between number of nucleus and markers files")
        
        # Combine both markers and nuclei paths
        paths = nuclues_paths + markers_paths
        
        images_groups = self.__get_grouped_images_for_paths(paths)

        processed_images = {}
        for group_id, images_group in images_groups.items():
            processed_image = self._process_images_group(group_id, images_group)
            if processed_image is None:
                continue
            processed_images.update(processed_image)
        
        return processed_images
        
    def get_nucleus_filepaths_for_markers_paths(self, markers_paths:Union[str, List[str]])->List[str]:
        """Get paths to the nucleus images for given paths to markers images

        Args:
            markers_paths (Union[str, List[str]]): The paths to the markers images

        Returns:
            List[str]: The paths to the corresponding nucleus images
        """
        if type(markers_paths) is str:
            markers_paths = [markers_paths]
        
        nucleus_filepaths = []
        for p in markers_paths:
            # Get the path to the folder holding the markers
            nucleus_folder = Path(os.path.join(Path(p).parent.parent, self.__NUCLEUS_MARKER_NAME))

            # Get the id for the current path
            image_id = self._get_id_of_image(p)
            
            # Find the nucleus image path for the current path
            nucleus_filepath = list(nucleus_folder.rglob(self._get_path_regex_from_id(image_id)))

            # Handle exceptions
            if len(nucleus_filepath) > 1:
                logging.error(f"Found multiple nucleus files for {p}: {nucleus_filepath}")
                raise Exception(f"Found multiple nucleus files for {p}: {nucleus_filepath}")
            
            if len(nucleus_filepath) == 0:
                logging.warning(f"No nucleus file was found for {p}")
                continue
            
            nucleus_filepaths.extend(nucleus_filepath)
        
        return nucleus_filepaths
        

    @abstractmethod
    def _get_id_of_image(self, path:str)->str:
        """Get an id for an image given its path

        Args:
            path (str): The path to the image

        Returns:
            str: Image's id
        """
        pass
    
    @abstractmethod
    def _get_path_regex_from_id(self, image_id: str)->str:
        """Get regex to search for path based on given image_id

        Args:
            image_id (str): The image id

        Returns:
            str: The regex for the path
        """
        pass

    def _get_valid_tiles_indexes(self, nucleus_image: np.ndarray, return_masked_tiles:bool = True) -> np.ndarray:
        """
        Get the indexes of valid tiles 
        
        Args:
            nucleus_image (np.ndarray): The nucleus image
            return_masked_tiles (bool): Whether to return the masked tiles or not

        Returns:
            np.ndarray: Array of valid tile indexes.
        """
        nuclei_mask = get_nuclei_segmentations(
            img=nucleus_image,
            cellpose_model=self.cellpose_model,
            diameter=self.preprocessing_config.CELLPOSE['NUCLEUS_DIAMETER'],
            cellprob_threshold=self.preprocessing_config.CELLPOSE['CELLPROB_THRESHOLD'],
            flow_threshold=self.preprocessing_config.CELLPOSE['FLOW_THRESHOLD'],
            show_plot=False
        )
        # Tile the nucleus mask and validate each tile
        nuclei_mask_tiled = crop_image_to_tiles(nuclei_mask, self.preprocessing_config.TILE_INTERMEDIATE_SHAPE)
        
        # Fix for non-valid polygons 
        whole_polygons = extract_polygons_from_mask(nuclei_mask) 
        # Filter out polygons which touch the outer frame         
        whole_polygons = self.__filter_intersecting_with_outer_frame(whole_polygons=whole_polygons, image_shape=self.preprocessing_config.EXPECTED_IMAGE_SHAPE)
        # In each tile - match the contained polygons with the whole ones.
        # output will look like: {tile_index: [whole_polygons_indexes]}
        dict_matches = self.__match_part_with_whole_pols(nuclei_mask_tiled , whole_polygons)
        # Select only tiles with passed nuclues 
        valid_tiles_indexes = np.where([self.__is_valid_tile(masked_tile,dict_matches, whole_polygons = whole_polygons , 
                                                                        ix=ix)
                                                   for ix, masked_tile in enumerate(nuclei_mask_tiled)])[0]

        # Filter out empty tiles or tiles with dead cells based on pixel intensities
        nuclei_tiled = crop_image_to_tiles(nucleus_image, self.preprocessing_config.TILE_INTERMEDIATE_SHAPE)
        _, valid_tiles_indexes = self.__process_and_filter_tiles(nuclei_tiled, valid_tiles_indexes, tile_name=self.__NUCLEUS_MARKER_NAME)

        if return_masked_tiles:
            return valid_tiles_indexes , nuclei_mask_tiled
        return valid_tiles_indexes

    def _get_valid_site_image(self, path: str, ch_idx:int = 0) -> Union[np.ndarray , None]:
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
            
        image = fit_image_shape(image, self.preprocessing_config.EXPECTED_IMAGE_SHAPE) 
        image = rescale_intensity(image,\
                                    lower_bound=self.preprocessing_config.RESCALE_INTENSITY['LOWER_BOUND'][ch_idx],\
                                    upper_bound=self.preprocessing_config.RESCALE_INTENSITY['UPPER_BOUND'][ch_idx]) 
        
        if self.markers_focus_boundries is not None:
            # Filter out-of-focus images
            marker = path_utils.get_raw_marker(path)
            thresholds = tuple(self.markers_focus_boundries.loc[marker].values)
            if not is_image_focused(image, thresholds): 
                logging.warning(f"out-of-focus for {marker}: {path}")
                return None
        return image

    def _process_images_group_and_save(self, group_id: str, images_group: Dict[str, str], save_folder_path:str):
        """Processed the images and save them to files

        Args:
            group_id (str): The group id
            images_group (Dict[str, str]): The group of paths
            save_folder_path (str): The path to the folder where to save the files
        """
        
        processed_images = self._process_images_group(group_id, images_group, save_folder_path) # 220725

        
        if processed_images is None or len(processed_images) == 0:
            logging.warning("No valid processed images!")
            return
        
        logging.info("Saving processed images to files")
        for raw_path, processed_image in processed_images.items():
            save_path = os.path.join(save_folder_path, Preprocessor.raw2processed_path(raw_path))
            
            # Save valid image pairs to disk
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                np.save(f, processed_image)
                logging.info(f"Saved to {save_path}")

    def _process_images_group(self, group_id: str, images_group: Dict[str, str], save_folder_path:str) -> Dict[str, np.ndarray ]:
        """Process the given group of images by processing the nuclues, the markers, filtering invalid tiles and stacking each processed marker
        with the processed nucleus

        Args:
            images_group (Dict[str, str]): 
                - Key: Marker name (nuclues marker included)
                - Value: Path to the markers' file

        Returns:
            Dict[str, np.ndarray ]: 
                Key: The path to the raw file
                Value: The processed valid tiles
        """        
        processed_images: Dict[str, np.ndarray ] = {}
        
        logging.info(f"Processing group {group_id}")

        self.__filter_already_processed_files_inplace(images_group, save_folder_path)

        if len(images_group) == 0:
            logging.warning(f"[{group_id}] No images to process. All files already exist in {save_folder_path}")
            return

        nucleus_path = images_group[self.__NUCLEUS_MARKER_NAME]
        logging.info(f"[{group_id}] Processing {self.__NUCLEUS_MARKER_NAME}: {nucleus_path}")
        
        processed_nucleus = self._get_valid_site_image(nucleus_path, ch_idx=1)
        if processed_nucleus is None: return 
        
        # Get valid tile indexes for the nucleus image
        valid_tiles_indexes, nuclei_mask_tiled  = self._get_valid_tiles_indexes(processed_nucleus) 

        self.logging_df.log_nucleus(nuclei_mask_tiled, valid_tiles_indexes, nucleus_path)
        if len(valid_tiles_indexes) == 0: 
            logging.warning(f"[{group_id}] No valid tiles were found for nucleus image: {nucleus_path}")
            return
                        
        markers = self.__sort_markers(images_group)

        panel_has_valid_markers = False
        # In case DAPI is the only marker analyzed or in case there is an already processed image from the group, don't test for other valid markers in the panel
        if markers == [self.__NUCLEUS_MARKER_NAME] or self.__has_valid_processed_markers_in_panel(images_group, save_folder_path):
            panel_has_valid_markers = True

        # For being able to filter out DAPI in case all panel markers are invalid, DAPI must be last
        for marker_name in markers: 
            marker_path = images_group[marker_name]
            logging.info(f"[{group_id}] Processing marker: {marker_name} from path: {marker_path}")

            # If we don't have valid markers in the panel, don't process/save DAPI
            if marker_name == self.__NUCLEUS_MARKER_NAME and not panel_has_valid_markers:
                    logging.warning(f"[{group_id}] No valid markers were found in this panel. Skipping also DAPI.")
                    break

            processed_marker = self._get_valid_site_image(marker_path, ch_idx= 0)
            if processed_marker is None: continue
            
            # Pair marker and nucleus images
            image_pair = np.stack([processed_marker, processed_nucleus], axis=-1)

            # Crop to tiles and take the valid ones
            image_pair_tiled = crop_image_to_tiles(image_pair, self.preprocessing_config.TILE_INTERMEDIATE_SHAPE)

            # Process and filter tiles based on variance and intensity
            image_pair_processed_valid_tiles, valid_tiles_indexes = self.__process_and_filter_tiles(image_pair_tiled, valid_tiles_indexes, tile_name=marker_name)
            
            if marker_name != self.__NUCLEUS_MARKER_NAME:
                self.logging_df.log_marker(valid_tiles_indexes, marker_path)

            # If no valid tiles were found, don't save this marker
            if len(image_pair_processed_valid_tiles) == 0: 
                logging.warning(f"[{group_id}, {marker_name}] No valid tiles were found for marker image: {marker_path}")
                continue
            
            processed_images[marker_path] = image_pair_processed_valid_tiles

            # Flag that we have at least one valid marker in the panel
            panel_has_valid_markers = True
            
        __shapes =  {m: v.shape for m, v in processed_images.items()}
        logging.info(f"[{group_id}] Shape of processed images: {__shapes}")
            
        return processed_images
        
        
    def __has_valid_processed_markers_in_panel(self, images_group: Dict[str, str], save_folder_path:str) -> bool:
        """
        Check if the panel has valid processed markers by checking if the processed files exist in the save folder.
        Args:
            images_group (Dict[str, str]): Mapping of marker names to paths.
            save_folder_path (str): The path to the folder where processed files are saved.
        Returns:    
            bool: True if the panel has valid processed markers, False otherwise.
        """
        panel_has_valid_processed_markers = any(
            os.path.exists(os.path.join(save_folder_path, Preprocessor.raw2processed_path(m_path)))
            for (m_name, m_path) in images_group.items() if m_name != self.__NUCLEUS_MARKER_NAME
        )

        return panel_has_valid_processed_markers

    def __process_and_filter_tiles(self, tiles:List[np.ndarray], valid_tiles_indexes:List[int], tile_name:str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filters out invalid tiles from the image pair and processes them.
        This method resizes the tiles to the expected shape and applies intensity rescaling.
        It also checks if the target tile is empty or contains dead cells based on intensity and variance thresholds.
        
        Args:
            tiles (List[np.ndarray]): List of tiled images for the marker and nucleus.
            valid_tiles_indexes (List[int]): List of indexes of valid tiles.
            tile_name (str):indicating if the tiles are for the nucleus marker ["nucleus"] or others ["marker_name"].
        
        Returns:    
            Tuple[np.ndarray, np.ndarray]: 
                - Processed valid tiles of the image pair.
                - Valid tiles indexes after filtering.
        """
        if len(valid_tiles_indexes) == 0:
            return np.array([]), valid_tiles_indexes
        
        image_pair_valid_tiles = []
        valid_tiles_indexes = valid_tiles_indexes.tolist()
        is_empty_tile_function = self.__is_empty_tile_dapi if (tile_name == self.__NUCLEUS_MARKER_NAME) else self.__is_empty_tile_target

        for i, tile in enumerate(tiles):
            if i not in valid_tiles_indexes:
                continue

            # Resize the tile to be TILE_SHAPE
            tile = transform.resize(tile, self.preprocessing_config.TILE_SHAPE, anti_aliasing=True)
            tile_rescaled = self.__apply_rescale_intensity_to_multi_channel_tile(tile)

            if is_empty_tile_function(tile[...,0], tile_rescaled[...,0], tile_name)[0]:
                valid_tiles_indexes.remove(i)
                continue

            image_pair_valid_tiles.append(tile_rescaled)
        
        valid_tiles_indexes = np.asarray(valid_tiles_indexes)
        
        if len(image_pair_valid_tiles) == 0:
            return np.array([]), valid_tiles_indexes

        image_pair_valid_tiles = np.stack(image_pair_valid_tiles)

        return image_pair_valid_tiles, valid_tiles_indexes

    def __sort_markers(self, images_group: Dict[str, str]) -> List[str]:
        """
        Sorts marker names so the nucleus marker is last.

        Args:
            images_group (Dict[str, str]): Mapping of marker names to paths.

        Returns:
            List[str]: Sorted list of marker names.
        """
        return sorted(images_group, key=lambda k: (k == self.__NUCLEUS_MARKER_NAME, k))

    def __filter_already_processed_files_inplace(self, images_group: Dict[str, str], save_folder_path:str) -> None:
        """
        Filter out already processed files in the images group by checking if the processed file exists.
        If a processed file exists, it is removed from the images group to avoid reprocessing.
        
        Args:
            images_group (Dict[str, str]): Mapping of marker names to paths.
            save_folder_path (str): The path to the folder where processed files are saved.
        
        Returns:
            None: The function modifies the images_group in place.
        """
        has_unprocessed_marker_in_group = False

        markers = self.__sort_markers(images_group)
        logging.info(f"Markers in the group: {markers}")

        for marker_name in markers:
            # If we have unprocessed marker in the group, return and don't filter out (i.e. do process) DAPI
            if has_unprocessed_marker_in_group and marker_name == self.__NUCLEUS_MARKER_NAME:
                return

            raw_path = images_group[marker_name]
            
            save_path = os.path.join(save_folder_path, Preprocessor.raw2processed_path(raw_path))
            
            # If the processed file does not exist, turn the flag to True
            if not os.path.exists(save_path):
                has_unprocessed_marker_in_group = True
                continue

            # If the processed file exists, remove it from the group to avoid processing it again
            logging.info(f"Skipping existing file: {save_path}")
            images_group.pop(marker_name)

    def __is_contains_dead_cells(self, dapi_rescaled:np.ndarray, intensity_threshold=0.95)-> bool:
        """Check if the DAPI image contains dead cells based on intensity and size thresholds.
        Parameters:
            dapi_rescaled: 2D numpy array of DAPI image, rescaled to [0, 1] range.
            intensity_threshold (optional): float, threshold for median intensity of the blob. (Default is 0.95)
        Returns:
            bool: True if dead cells are detected, False otherwise."""

        def __is_blob_touching_edge(blob_mask: np.ndarray) -> bool:
            """
            Returns True if any part of the blob (binary mask) touches the image edge.
            
            Parameters:
                blob_mask: 2D boolean or integer array (True where blob is present)
                
            Returns:
                bool: True if the blob touches any border (top, bottom, left, right)
            """
            rows, cols = np.where(blob_mask)
            nrows, ncols = blob_mask.shape
            touches_top    = (rows == 0).any()
            touches_bottom = (rows == nrows - 1).any()
            touches_left   = (cols == 0).any()
            touches_right  = (cols == ncols - 1).any()
            return touches_top or touches_bottom or touches_left or touches_right

        # Separate between background and foreground using Otsu's method
        otsu_thresh = threshold_otsu(dapi_rescaled)
        dapi_mask = dapi_rescaled > otsu_thresh
        # CHANGE - print
        # print("otsu_thresh:", otsu_thresh)

        # Detect connected components in the binary mask (0 is background)
        labeled, ncomponents = label(dapi_mask)


        # CHANGED - KEEP
        if ncomponents >= self.preprocessing_config.MAX_NUM_NUCLEI_BLOB:
            print("ncomponents failed:", ncomponents)
            return True
        
        for i in range(1, ncomponents + 1): # 0 is the background
            blob_mask = (labeled == i)
            dapi_masked = dapi_rescaled[blob_mask]

            blob_variance = dapi_masked.var()
            blob_size = blob_mask.sum()
            blob_median = np.median(dapi_masked)

            # CHANGE - KEEP:
            # detecet ALIVE NUCLEUS (right size - above minimal thershold) with:
            #               OLD:
            #               --> low variance *and* intensity 
            #            or --> high variance *and* intensity 
            #               NEW:
            #               --> high intensity *and* (either low or high variance)
            #            or --> very high size (=~noise) (above maximal threshold)
            # which indicates blurred / about-to-die / dead cell 
            if blob_size > self.preprocessing_config.MIN_ALIVE_NUCLEI_AREA and \
                (((blob_variance <= self.preprocessing_config.MIN_VARIANCE_THRESHOLD_ALIVE_NUCLEI or \
                blob_variance >= self.preprocessing_config.MAX_VARIANCE_THRESHOLD_ALIVE_NUCLEI) and blob_median >= self.preprocessing_config.MAX_MEDIAN_INTENSITY_THRESHOLD_ALIVE_NUCLEI) or \
                blob_size >= self.preprocessing_config.MAX_ALIVE_NUCLEI_AREA):
                print("ALIVE CELL failed thresholds -")
                print("blob_size:", blob_size, "blob_median: ", blob_median, "blob_variance:", blob_variance)
                return True

            
            # CHANGE - KEEP
            # detect DEAD NUCLEUS
            #if blob_median >= intensity_threshold or (not __is_blob_touching_edge(blob_mask) and blob_size <= self.preprocessing_config.MIN_ALIVE_NUCLEI_AREA):
            if blob_median >= intensity_threshold and \
            blob_size <= self.preprocessing_config.MIN_ALIVE_NUCLEI_AREA and\
            blob_size >= self.preprocessing_config.MIN_NUCLEI_BLOB_AREA and \
            (blob_variance >= self.preprocessing_config.MIN_VARIANCE_NUCLEI_BLOB_THRESHOLD or\
            blob_variance <= self.preprocessing_config.MAX_VARIANCE_NUCLEI_BLOB_THRESHOLD):
                print("DEAD CELL failed thresholds -")
                print("blob_size:", blob_size, "blob_median: ", blob_median, "blob_variance:", blob_variance)
                return  True
            
            # CHANGE
            print("passed:")
            print("blob_size:", blob_size, "blob_median: ", blob_median, "blob_variance:", blob_variance)
                
        return False

    def __is_empty_tile_dapi(self, dapi:np.ndarray, dapi_scaled:np.ndarray, tile_name:str)-> Tuple[bool, Union[str, None]]:
        """
        Check if the DAPI image channel is empty or contains dead cells based on max intensity and variance thresholds.
        
        Parameters:
            dapi: 2D numpy array of the DAPI image channel.
            dapi_scaled: 2D numpy array of the rescaled DAPI image channel.
            tile_name: the name of the DAPI channel in the config
        
        Returns:
            bool: True if the DAPI image channel is empty, False otherwise.
            str: Optional reason for being empty.
        """
        # #  CHANGE
        # #  DISCARDED FOR NOW- first run: don't apply threshold at all 
        # return False, None


        # ADDED TILE BRENNER - GAL'S CODE
        if self.markers_focus_boundries_tiles is not None:
            out_of_focus_threshold = tuple(self.markers_focus_boundries_tiles.loc[tile_name].values)
        else:
            out_of_focus_threshold = None
        
        result, cause = self.__is_empty_tile(dapi, dapi_scaled,\
                                             lower_bound_intensity_threshold=self.preprocessing_config.MAX_INTENSITY_THRESHOLD_NUCLEI,\
                                             lower_bound_variance_threshold=self.preprocessing_config.VARIANCE_THRESHOLD_NUCLEI, \
                                             out_of_focus_threshold=out_of_focus_threshold)
        if result:
            return True, f'[DAPI] {cause}'

        if self.__is_contains_dead_cells(dapi_scaled, intensity_threshold=self.preprocessing_config.MIN_MEDIAN_INTENSITY_NUCLEI_BLOB_THRESHOLD):
            return True, "Contains dead cells"

        return False, None


    def __is_empty_tile_target(self, target:np.ndarray, target_scaled:np.ndarray, target_name:str)-> Tuple[bool, Union[str, None]]:
        """ Check if the target image channel is empty based on max intensity and variance thresholds.
        Parameters:
            target: 2D numpy array of the target image channel.
            target_scaled: 2D numpy array of the rescaled target image channel.
            tile_name: the name of the targer (marker) channel in the config
        Returns:
            bool: True if the target image channel is empty, False otherwise.
            str: Optional reason for being empty."""

        # #  CHANGE
        # #  DISCARDED FOR NOW- first run: don't apply threshold at all 
        # #                     later: maybe run only only *non* on-off markers 
        # return False, None

        # ADDED TILE BRENNER - GAL'S CODE
        if self.markers_focus_boundries_tiles is not None:
                out_of_focus_threshold = tuple(self.markers_focus_boundries_tiles.loc[target_name].values)
        else:
                out_of_focus_threshold = None
        
        result, cause = self.__is_empty_tile(target, target_scaled,\
                                             lower_bound_intensity_threshold=self.preprocessing_config.MAX_INTENSITY_THRESHOLD_TARGET,\
                                             upper_bound_intensity_threshold=self.preprocessing_config.MAX_INTENSITY_UPPER_BOUND_THRESHOLD_TARGET, \
                                             lower_bound_variance_threshold=self.preprocessing_config.VARIANCE_THRESHOLD_TARGET, \
                                             upper_bound_variance_threshold=self.preprocessing_config.VARIANCE_UPPER_BOUND_THRESHOLD_TARGET, \
                                            out_of_focus_threshold = out_of_focus_threshold)

        if cause is not None:
            cause = f'[Target] {cause}'

        return result, cause

    def __is_empty_tile(self, image_channel:np.ndarray, 
                            image_channel_rescaled:np.ndarray, 
                            lower_bound_intensity_threshold:float,  
                            lower_bound_variance_threshold:float,
                            upper_bound_intensity_threshold:float = None,
                            upper_bound_variance_threshold:float = None,
                            out_of_focus_threshold:float = None) -> Tuple[bool, Union[str, None]]:
        """ Check if the image channel is empty based on max intensity and variance thresholds.
        Parameters:
            image_channel: 2D numpy array of the image channel.
            image_channel_rescaled: 2D numpy array of the rescaled image channel.
            lower_bound_variance_threshold: float, lower bound threshold for maximum intensity.
            upper_bound_intensity_threshold: float (optional), upper bound threshold for maximum intensity.
            lower_bound_variance_threshold: float, lower bound threshold for variance.
            upper_bound_variance_threshold: float (optional), upper bound threshold for variance.
            out_of_focus_threshold: float (optional), out-of-focus threshold (brenner).
        Returns:
            bool: True if the image channel is empty, False otherwise.
            str: Optional reason for being empty."""

        image_channel_max_intensity = round(image_channel.max(), 4)
        if image_channel_max_intensity <= lower_bound_intensity_threshold:
            return True, f"Invalid max intensity: {image_channel_max_intensity} <= {lower_bound_intensity_threshold}"
        
        if upper_bound_intensity_threshold is not None:
            if image_channel_max_intensity >= upper_bound_intensity_threshold:
                return True, f"Invalid max intensity: {image_channel_max_intensity} >= {upper_bound_intensity_threshold}"
        
        image_channel_rescaled_variance = round(image_channel_rescaled.var(), 4)
        if image_channel_rescaled_variance <= lower_bound_variance_threshold:
            return True, f"Invalid variance: {image_channel_rescaled_variance} <= {lower_bound_variance_threshold}"

        
        if upper_bound_variance_threshold is not None:
            if image_channel_rescaled_variance >= upper_bound_variance_threshold:
                return True, f"Invalid variance: {image_channel_rescaled_variance} >= {upper_bound_variance_threshold}"

        # ADDED - TILE BRENNER - GAL'S CODE
        if out_of_focus_threshold is not None:
            if not is_tile_focused(image_channel, out_of_focus_threshold):
                return True, f"out-of-focus tile: lower bound threshold = {out_of_focus_threshold}"

        return False, None


    def __apply_rescale_intensity_to_multi_channel_tile(self, tile: np.ndarray) -> np.ndarray:
        """
        Apply rescale_intensity to each channel of the given tile.

        Parameters:
        - tile: np.ndarray of shape (H, W, C)

        Returns:
        - np.ndarray of same shape as tile with function applied per channel (H,W,C)
        """
        H, W, C = tile.shape
        result = np.empty_like(tile)

        for c in range(C):
            result[...,c] = rescale_intensity(tile[...,c],
                                                lower_bound=self.preprocessing_config.RESCALE_INTENSITY['LOWER_BOUND'][c],\
                                                upper_bound=self.preprocessing_config.RESCALE_INTENSITY['UPPER_BOUND'][c])

        return result


    def __filter_intersecting_with_outer_frame(self, whole_polygons: List[Polygon], image_shape: tuple) -> List[Polygon]:
        """
        Filter out whole_polygons that intersect with the outer frame of the image.

        Args:
            whole_polygons (List[Polygon]): List of polygons to filter.
            image_shape (tuple): Shape of the image (height, width).

        Returns:
            List[Polygon]: Filtered list of polygons.
        """
        height, width = image_shape
        buffer = self.preprocessing_config.FRAME_WIDTH_BUFFER

        # Create a smaller inner box inset by `buffer` from all sides
        inner_box = box(buffer, buffer, width - buffer, height - buffer)

        # Keep only polygons that are fully inside this box
        whole_polygons = [p for p in whole_polygons if inner_box.contains(p)]
        
        return whole_polygons
    
    def __match_part_with_whole_pols(self ,nuclei_mask_tiled , whole_polygons) -> Dict :
        """
        Matches nuclei polygons extracted from tiled masks to corresponding whole-image polygons.

        Each tile in the image , which contains one/few nuclei, and the goal is to match these
        partial tile-based polygons to their corresponding complete polygon in the whole image.

        The function does the following:
        - Iterates over each tile and extracts part-polygons from it.
        - Translates each part-polygon to the global coordinate space based on its tile location.
        - For each part-polygon, checks if its representative point lies inside any full polygon.
        - If a match is found - it appends the match index. If no match is found, it appends `None`.

        Args:
            nuclei_mask_tiled (List[np.ndarray]): List of binary mask tiles containing nuclei segmentations.
            whole_polygons (List[shapely.geometry.Polygon]): List of full polygons in the complete image space.

        Returns:
            dict_matches (Dict[int, List[int | None]]): A dictionary mapping each tile index to a list of matched
                                                        whole polygon indices, or None if no match was found.
        """
        
        dict_matches = defaultdict(list)

        # get parameters of tiles, for determiming the tile location on the complete image
        # expected image shape - the width/height (assumed to be the same) of the whole image
        expected_image_shape = self.preprocessing_config.EXPECTED_IMAGE_SHAPE[0]
        # tile width/height (assumed to be the same) 
        tile_size = self.preprocessing_config.TILE_INTERMEDIATE_SHAPE[0]
        # number of tiles along the width/height of the whole image
        n_tiles = expected_image_shape // tile_size

        # Iterate over each tile (part polygons inside)
        for ix, masked_tile in enumerate(nuclei_mask_tiled):
            part_polygons = extract_polygons_from_mask(masked_tile)

            # --------------------------------------
            # Iterate over each part polygon (nuclei parts inside the tile) and find it's matching whole polygon 
            # If found - add the whole polygon index to the dict, else add None
            # --------------------------------------
            for p in part_polygons:
                # 1. Translate polygon to global image coordinates based on tile index
                # 2. Find a point guaranteed to be inside the polygon
                p1 = affinity.translate(p, xoff=ix % n_tiles * tile_size, yoff=ix // n_tiles * tile_size)
                pc = p1.representative_point()  

                # --------------------------------------
                # Check if this polygon corresponds to a known whole polygon
                # by verifying:
                #   - the point lies inside a full polygon
                #   - the area ratio exceeds a set threshold
                # --------------------------------------
                found_match = 0
                if p is not None: 
                    for ix_whole , pol_whole in enumerate(whole_polygons):
                        if pc.intersects(pol_whole):
                            dict_matches[ix].append(ix_whole)
                            found_match = 1 

                if found_match == 0:
                    dict_matches[ix].append(None)

        return dict_matches


    def __is_valid_tile(self, masked_tile: np.ndarray,dict_matches: Dict, whole_polygons=None, ix=None) -> bool:
        """
        Check if the tile has at least one whole nucleus but not more than the maximum allowed nuclei.
        
        Args:
            masked_tile (np.ndarray): Segmented tile image (mask of nuclei within the tile)
            dict_matches (Dict): Dictionary mapping tile index to matched whole polygon indices
            whole_polygons (List[Polygon]): List of all complete nucleus polygons across the entire image
            ix (int): Index of the tile (used to compute its position in the global image)
        
        Returns:
            bool: True if the tile contains a valid nucleus and not more than the allowed number, else False.
        """

        # --------------------------------------
        # Extract partial polygons from tile mask
        # These are the intersected nuclei parts within the tile
        # --------------------------------------

        
        polygons = extract_polygons_from_mask(masked_tile)
        matched_polygons_ixs = dict_matches[ix]

        passed_tile = False

        for pol_ix, pol_part in zip(matched_polygons_ixs, polygons):
            if pol_ix is not None and pol_part is not None:
                if pol_part.area/ whole_polygons[pol_ix].area > self.preprocessing_config.INCLUDED_AREA_RATIO:
                    passed_tile = True
                    break

        # Image and tile size setup
        max_num_nuclei = self.preprocessing_config.MAX_NUM_NUCLEI

        # --------------------------------------
        # Evaluate tile conditions
        #   cond1: contains at least one sufficiently complete nucleus
        #   cond2: does not exceed the maximum allowed nuclei count
        # --------------------------------------
        cond1 = passed_tile
        cond2 = get_nuclei_count(masked_tile) <= max_num_nuclei

        return cond1 and cond2 
   
    def __get_grouped_images_for_folder(self, folder_path:str)->Dict[str, Dict[str, str]]:
        """Get groups of images for the given folder, filtered based on the configuration settings

        Args:
            folder_path (str): The path to the folder

        Returns:
            Dict[str, List[str]]: The groups
                - Key: The relative path from the base_dir to the markers root folder (cellLine/panel/condition/rep)
                - Value: A dict with keys being the marker name and values being the full paths to all image in the markers root folder having the same id
        """
        
        # Get all paths to files for the given folder
        paths = self.__get_supported_filepaths_from_basedir(folder_path)
        # Filter paths based on the configuration requierments 
        paths = self.__filter_paths_by_configuration(paths)

        # Get and store the groups to dict
        grouped_images_paths_dict = self.__get_grouped_images_for_paths(paths)
        
        return grouped_images_paths_dict
        
    def __get_grouped_images_for_paths(self, paths: Union[List[Path], List[str]])->Dict[str, Dict[str, str]]:
        """Get group of images with the same id in the markers root folder under the given base_dir

        Args:
            paths (Union[List[Path], List[str]]): List of paths 

        Returns:
            Dict[str, List[str]]: 
                - Key: The relative path from the base_dir to the markers root folder (cellLine/panel/condition/rep)
                - Value: A dict with keys being the marker name and values being the full paths to all image in the markers root folder having the same id
        """
        # Dictionary to hold groups of files by their identifier
        groups = defaultdict(dict)

        for file_path in paths:
            # Convert path to Path
            if type(file_path) is not Path:
                file_path = Path(file_path)
                
            # Take the path of the file's grandfather, i.e. the root folder of markers
            markers_rootfolder_path = str(file_path.parent.parent)
            # Clean it a bit - remove the input folders paths
            markers_rootfolder_path = re.sub('|'.join(map(re.escape, self.preprocessing_config.INPUT_FOLDERS)), '', markers_rootfolder_path)
            
            # Extract identifier from the file name
            image_id = self._get_id_of_image(str(file_path)) 
            
            # Get group id
            group_id = os.path.join(markers_rootfolder_path, image_id)
            
            # Get marker name to serve as key
            marker_name = file_path.parent.name
            
            # Group by id and store the file path under the extracted marker name
            groups[group_id].update({marker_name: str(file_path)})
            
        return dict(groups)
    
    def __filter_paths_by_configuration(self, paths: List[Path])->List[Path]:
        """Filter paths based on the configuration files (which cell lines, conditions, reps and markers to keep)

        Args:
            paths (List[Path]): The paths to filter

        Returns:
            List[Path]: The filtered paths
        """
        # Filter cell lines
        paths = filter_paths_by_substrings(paths, self.preprocessing_config.CELL_LINES, path_utils.raw_parts.cell_line_part_indx)
        # Filter conditions
        paths = filter_paths_by_substrings(paths, self.preprocessing_config.CONDITIONS, path_utils.raw_parts.condition_part_indx)
        # Filter reps
        paths = filter_paths_by_substrings(paths, self.preprocessing_config.REPS, path_utils.raw_parts.rep_part_indx)
        
        # Filter markers
        paths = filter_paths_by_substrings(paths, self.preprocessing_config.MARKERS, path_utils.raw_parts.marker_part_indx)
        paths = filter_paths_by_substrings(paths, self.preprocessing_config.MARKERS_TO_EXCLUDE, path_utils.raw_parts.marker_part_indx, filter_out=True)
      
        # Filter panels
        paths = filter_paths_by_substrings(paths, self.preprocessing_config.PANELS, path_utils.raw_parts.panel_part_indx)
      
        return paths
    
    def __get_supported_filepaths_from_basedir(self, base_dir:str)->List[Path]:
        """Get all supported file's path for a given base dir

        Args:
            base_dir (str): The base dir path

        Returns:
            List[Path]: The paths
        """
        paths = [Path(base_dir).rglob(f'*.{ext}') for ext in self.__SUPPORTED_EXTENSIONS]
        paths = flat_list_of_lists(paths)
        
        return paths    
    
    def __is_nucleus_requested(self)->bool:
        """Is the nucleus maker requested in the configuration 

        Returns:
            bool: Is loaded?
        """
        if self.preprocessing_config.MARKERS is not None and self.__NUCLEUS_MARKER_NAME not in self.preprocessing_config.MARKERS:
            return False
        
        if self.preprocessing_config.MARKERS_TO_EXCLUDE is not None and self.__NUCLEUS_MARKER_NAME in self.preprocessing_config.MARKERS_TO_EXCLUDE:
            return False
        
        return True
        
        
