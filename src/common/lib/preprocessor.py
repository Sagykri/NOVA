import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))


import string
from src.common.configs.preprocessing_config import PreprocessingConfig

from abc import ABC, abstractmethod

class Preprocessor(ABC):
  def __init__(self, conf: PreprocessingConfig):
    self.input_folders = conf.INPUT_FOLDERS
    self.output_folders = conf.OUTPUT_FOLDERS

  @abstractmethod
  def preprocess_images(self, **kwargs):
    pass

  @abstractmethod
  def preprocess_image(self, input_path: string, output_path: string, **kwargs):
    """Preprocess a single image

    Args:
        input_path (string): Path to the raw image
        output_path (string): Path to the output (preprocessed) image
    """
    pass


