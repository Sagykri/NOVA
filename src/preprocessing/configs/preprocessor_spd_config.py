import os
from common.configs.preprocessing_config import PreprocessingConfig

class SPDPreprocessingConfig(PreprocessingConfig):
    def __init__(self):
        super(SPDPreprocessingConfig, self).__init__()
        
        self.preprocessor_class_path = os.path.join(self.HOME_FOLDER, "src", "preprocessing", "preprocessors", "preprocessor_spd.py")
