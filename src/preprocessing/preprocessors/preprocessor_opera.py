import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.preprocessing.path_utils import get_filename
from src.preprocessing.preprocessors.preprocessor_base import Preprocessor
from src.preprocessing.preprocessing_config import PreprocessingConfig

class OperaPreprocessor(Preprocessor):
    """
    Preprocessor for preprocessing images captured by the spinning disk
    """
    
    def __init__(self, preprocessing_config: PreprocessingConfig):
        super().__init__(preprocessing_config)
                        
    def _get_id_of_image(self, path: str) -> str:
        """Get an id for an image given its path

        Args:
            path (str): The path to the image

        Returns:
            str: Image's id
        """
        
        return get_filename(path).split('-')[0]
    
    def _get_path_regex_from_id(self, image_id: str)->str:
        """Get regex to search for path based on given image_id

        Args:
            image_id (str): The image id

        Returns:
            str: The regex for the path
        """
        return f"{image_id}-*.tif*"
