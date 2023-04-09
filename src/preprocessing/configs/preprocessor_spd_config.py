import os
from src.common.configs.preprocessing_config import PreprocessingConfig

class SPDPreprocessingConfig(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.preprocessor_class_path = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_spd", "SPDPreprocessor")
