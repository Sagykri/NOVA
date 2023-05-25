import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.preprocessing_config import PreprocessingConfig

class ConfPreprocessingConfig(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_conf", "ConfPreprocessor")
        