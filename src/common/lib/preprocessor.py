import string
from common.configs.preprocessing_config import PreprocessingConfig

from abc import ABC, abstractmethod

class Preprocessor(ABC):
  def __init__(self, conf: PreprocessingConfig):
    self.input_folders = conf.input_folders
    self.output_folders = conf.output_folders

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


