import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import logging

from src.common.configs.preprocessing_config import PreprocessingConfig
from src.common.lib.preprocessor import Preprocessor
from src.common.lib.utils import load_config_file, get_class


def run_preprocessing():
    run_config: PreprocessingConfig = load_config_file(sys.argv[1], '_preprocessing')
    
    logging.info("init")
    
    
    logging.info(f"Importing preprocessor class.. {run_config.PREPROCESSOR_CLASS_PATH}")
    preprocessor_class: Preprocessor = get_class(run_config.PREPROCESSOR_CLASS_PATH)
    
    logging.info(f"Instantiate preprocessor {type(preprocessor_class)}")
    preprocessor: Preprocessor = preprocessor_class(run_config)
    
    logging.info(f"Preprocessing images...")
    preprocessor.preprocess_images()
    
if __name__ == "__main__":
    print("---------------Start---------------")
    run_preprocessing()
    logging.info("Done")
    
    