from collections import defaultdict
import logging
from multiprocessing import Pool
import os
from pathlib import Path
import re
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Union
import cv2
import numpy as np
from cellpose import models
import pandas as pd
from shapely import affinity , make_valid
from shapely.geometry import box ,Polygon
import cellpose


from skimage import transform

sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.common.utils import filter_paths_by_substrings, flat_list_of_lists, get_if_exists
from src.preprocessing.log_df_preprocessing import LogDFPreprocessing
from src.preprocessing import path_utils
from src.preprocessing.preprocessing_utils import get_nuclei_count, crop_image_to_tiles, extract_polygons_from_mask,\
                                                      fit_image_shape, get_nuclei_segmentations, is_image_focused,\
                                                      is_contains_whole_nucleus, rescale_intensity
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
            nucleus_image (np.ndarray): The nucleus imageת
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
                                                   for ix, masked_tile in enumerate(nuclei_mask_tiled)])

        valid_tiles_indexes = valid_tiles_indexes[0]

        if return_masked_tiles:
            return valid_tiles_indexes , nuclei_mask_tiled
        return valid_tiles_indexes

    def _get_image(self, path: str) -> Union[np.ndarray , None]:
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
                                    lower_bound=self.preprocessing_config.RESCALE_INTENSITY['LOWER_BOUND'],\
                                    upper_bound=self.preprocessing_config.RESCALE_INTENSITY['UPPER_BOUND']) 
        
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
        
        processed_images = self._process_images_group(group_id, images_group)
        
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


    def _process_images_group(self, group_id: str, images_group: Dict[str, str]) -> Dict[str, np.ndarray ]:
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
        nucleus_path = images_group[self.__NUCLEUS_MARKER_NAME]
        logging.info(f"[{group_id}] Processing {self.__NUCLEUS_MARKER_NAME}: {nucleus_path}")
        
        processed_nucleus = self._get_image(nucleus_path)
        if processed_nucleus is None: return 
        
        # Get valid tile indexes for the nucleus image
        valid_tiles_indexes, nuclei_mask_tiled  = self._get_valid_tiles_indexes(processed_nucleus) 

        self.logging_df.log_nucleus(nuclei_mask_tiled, valid_tiles_indexes, nucleus_path)
        if len(valid_tiles_indexes) == 0: 
            logging.warning(f"[{group_id}] No valid tiles were found for nucleus image: {nucleus_path}")
            return
                
        # Process each marker image in the same plate as the current nucleus
        for marker_name, marker_path in images_group.items():

            processed_marker = self._get_image(marker_path)

            if processed_marker is None: continue
            if marker_name != self.__NUCLEUS_MARKER_NAME:
                self.logging_df.log_marker(valid_tiles_indexes, marker_path)
            
            # Pair marker and nucleus images
            image_pair = np.stack([processed_marker, processed_nucleus], axis=-1)

            # Crop to tiles and take the valid ones
            image_pair_tiled = crop_image_to_tiles(image_pair, self.preprocessing_config.TILE_INTERMEDIATE_SHAPE)
            image_pair_valid_tiles = image_pair_tiled[valid_tiles_indexes]
            
            # Resize the tile to be TILE_SHAPE
            image_pair_valid_tiles = [transform.resize(tile, self.preprocessing_config.TILE_SHAPE, anti_aliasing=True) for tile in image_pair_valid_tiles]
            image_pair_valid_tiles = np.stack(image_pair_valid_tiles)
            
            processed_images[marker_path] = image_pair_valid_tiles
            
        __shapes =  {m: v.shape for m, v in processed_images.items()}
        logging.info(f"[{group_id}] Shape of processed images: {__shapes}")
            
        return processed_images
    
    def __filter_intersecting_with_outer_frame(self, whole_polygons: List[Polygon], image_shape: tuple) -> List[Polygon]:
        """
        Filter out whole_polygons that intersect with the outer frame of the image.

        Args:
            whole_polygons (List[Polygon]): List of polygons to filter.
            image_shape (tuple): Shape of the image (height, width).

        Returns:
            List[Polygon]: Filtered list of polygons.
        """
        # Define image outer frame 
        image_border = box(0,0,image_shape[0],image_shape[1]) 
        image_border_buffered = image_border.buffer(self.preprocessing_config.FRAME_WIDTH_BUFFER)
        # Filter out polygons which touch the outer frame 
        whole_polygons = [p for p in whole_polygons if not p.intersects(image_border_buffered)]
        
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
        
        