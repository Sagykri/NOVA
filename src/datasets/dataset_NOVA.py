
import logging
import os
from typing import Dict, List, Tuple
import numpy as np
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.datasets.dataset_base import DatasetBase
from src.datasets.dataset_config import DatasetConfig

class DatasetNOVA(DatasetBase):
    """
    Dataset customize to load the data ordered in NOVA's files structure
    """
    
    # Number of levels to marker folders. Example structure: /batch/cell_line/condition/marker/
    __markers_folders_depth = 3
    
    def __init__(self, dataset_config: DatasetConfig):    
        # Set random seed
        np.random.seed(dataset_config.SEED)
        
        super().__init__(dataset_config)
        

    def _load_data_paths(self)-> Tuple[np.ndarray[str], np.ndarray[str]]:
        """Return processed images (paths to npy files) from given folders.
            Each image is stored in npy of (n_tiles, 100, 100, 2), AKA target and DAPI marker.

        Returns:
            Tuple[np.ndarray[str], np.ndarray[str]]: Paths to images, and their labels
        """
        
        # Initialize collections
        image_paths = []
        labels = []

        # Main loop to process input folders
        for folder in self.dataset_config.INPUT_FOLDERS:
            logging.info(f"Input folder: {folder}")
            marker_subfolders = self.__find_marker_folders(folder)
            
            # Process each marker folder
            for marker_folder in marker_subfolders:
                logging.info(f"Marker folder: {marker_folder}")

                # Apply filtering based on cell line, condition, marker, and excluded markers
                if not self.__passes_filters(marker_folder):
                    continue

                # Process image files in the marker folder
                folder_image_paths, folder_labels = self.__process_marker_folder(marker_folder)
                
                image_paths.extend(folder_image_paths)
                labels.extend(folder_labels)

        # Convert lists to numpy arrays
        image_paths = np.asarray(image_paths)
        labels = np.asarray(labels).reshape(-1, 1)

        logging.info(f"{len(image_paths)} files processed, {labels.shape[0]} labels generated")

        return image_paths, labels

        
    def __find_marker_folders(self, batch_path:str, depth:int=__markers_folders_depth):
        """Returns paths of all marker folder in a batch (works with recursion)
            Note: 
                    Markers are assumend to be always in a given constant "depth" dictated by self.__markers_folders_depth 
        Args:
            batch_path (str):  A full path to a batch folder
            depth (int, optional): The depth to the marker folders (decreased each recursive calling)
        Yields:
            str: A path to a marker folder
        """
        
        # Recursively list files and directories up to a certain depth
        depth -= 1
        with os.scandir(batch_path) as input_data_folder:
            
            for entry in sorted(input_data_folder, key=lambda e: e.name):
                
                # if that's not a marker directory, recursion...
                if entry.is_dir() and depth > 0:
                    yield from self.__find_marker_folders(entry.path, depth)
                    
                # if that's a marker directory
                elif depth==0: 
                    # This is a list of arguments, used as the input of analyze_marker()
                    yield entry.path
                    
    def __get_filter_criteria(self) -> Dict:
        """
        Retrieve filter criteria for data processing.
        
        Returns:
            Dict: Dictionary of filter criteria.
        """
        config = self.dataset_config
        return {
            'celllines': config.CELL_LINES,
            'conditions': config.CONDITIONS,
            'markers': config.MARKERS,
            'markers_to_exclude': config.MARKERS_TO_EXCLUDE,
            'reps': config.REPS
        }

    def __get_label_flags(self) -> Dict:
        """
        Retrieve flags for label generation.

        Returns:
            Dict: Dictionary of label generation flags.
        """
        config = self.dataset_config
        return {
            'condition': config.ADD_CONDITION_TO_LABEL,
            'cellline': config.ADD_LINE_TO_LABEL,
            'batch': config.ADD_BATCH_TO_LABEL,
            'rep': config.ADD_REP_TO_LABEL
        }
        
    def __process_marker_folder(self, marker_folder: str) -> Tuple[List[str], List[str]]:
        """
        Process a single marker folder by reading files and generating labels.
        
        Args:
            marker_folder (str): The path of the marker folder.

        Returns:
            Tuple[List[str], List[str]]:
                - List of image file paths.
                - List of corresponding labels.
        """
        image_paths: List[str] = []
        labels: List[str] = []
        
        filters = self.__get_filter_criteria()

        filenames = sorted(os.listdir(marker_folder))
        
        for filename in filenames:
            name, ext = os.path.splitext(filename)

            # Only process supported file types
            if ext != '.npy':
                logging.info(f"Skipping unsupported file type: {filename}")
                continue

            # Extract rep
            rep = self.__extract_rep(name)
            
            # Skip if the rep doesn't pass the filter
            if filters['reps'] and rep not in filters['reps']:
                logging.info(f"Skipping rep {rep} (not in rep filter list).")
                continue

            # Append image path
            image_paths.append(os.path.join(marker_folder, filename))
            
            # Generate label for the file
            label = self.__generate_label(marker_folder, rep)
            labels.append(label)
        
        return image_paths, labels    
    
    def __extract_rep(self, filename: str) -> str:
        """
        Extract the rep from the filename.

        Args:
            filename (str): The filename

        Returns:
            str: The rep extracted from the folder path.
        """
        return filename.split('_', 1)[0]
                    
    def __extract_marker_name(self, marker_folder: str) -> str:
        """
        Extract the marker name from the folder path.

        Args:
            marker_folder (str): Full path to the marker folder.

        Returns:
            str: The marker name extracted from the folder path.
        """
        return os.path.basename(marker_folder)


    def __extract_condition(self, marker_folder: str) -> str:
        """
        Extract the experimental condition from the folder path.

        Args:
            marker_folder (str): Full path to the marker folder.

        Returns:
            str: The condition extracted from the folder path.
        """
        return marker_folder.split(os.sep)[-2]


    def __extract_cell_line(self, marker_folder: str) -> str:
        """
        Extract the cell line from the folder path.

        Args:
            marker_folder (str): Full path to the marker folder.

        Returns:
            str: The cell line extracted from the folder path.
        """
        return marker_folder.split(os.sep)[-3]

    def __extract_batch(self, marker_folder:str) ->str:
        """
        Extract the batch from the folder path.

        Args:
            marker_folder (str): Full path to the marker folder.

        Returns:
            str: The batch extracted from the folder path.
        """
        return marker_folder.split(os.sep)[-4]


    def __passes_filters(self, marker_folder: str) -> bool:
        """
        Check if the current folder passes the inclusion and exclusion filters.

        Args:
            marker_folder (str): Full path to the marker folder.

        Returns:
            bool: True if the folder passes all filter criteria, False otherwise.
        """
        
        filters:dict = self.__get_filter_criteria()
        
        marker_name = self.__extract_marker_name(marker_folder)
        condition = self.__extract_condition(marker_folder)
        cell_line = self.__extract_cell_line(marker_folder)

        if filters['celllines'] and cell_line not in filters['celllines']:
            logging.info(f"Skipping cell line (not in cell lines list): {cell_line}")
            return False
        if filters['conditions'] and condition not in filters['conditions']:
            logging.info(f"Skipping condition (not in conditions list): {condition}")
            return False
        if filters['markers'] and marker_name not in filters['markers']:
            logging.info(f"Skipping marker (not in markers list): {marker_name}")
            return False
        if filters['markers_to_exclude'] and marker_name in filters['markers_to_exclude']:
            logging.info(f"Skipping marker (in exclusion list): {marker_name}")
            return False
        return True


    def __generate_label(self, marker_folder: str, rep: str) -> str:
        """
        Generate a label for the current image based on the specified flags.

        Args:
            marker_folder (str): Full path to the marker folder.
            rep (str): The rep for this label

        Returns:
            str: The generated label for the image. The structure: Marker_CellLine_Condition_Batch_Rep
        """
        
        flags = self.__get_label_flags()
        
        marker_name = self.__extract_marker_name(marker_folder)
        cell_line = self.__extract_cell_line(marker_folder)
        condition = self.__extract_condition(marker_folder)
        batch = self.__extract_batch(marker_folder)

        label = marker_name
        if flags['cellline']:
            label += f"_{cell_line}"
        if flags['condition']:
            label += f"_{condition}"
        if flags['batch']:
            label += f"_{batch}"
        if flags['rep']:
            label += f"_{rep}"
        return label