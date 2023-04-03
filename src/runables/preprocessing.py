import sys
import importlib
import logging
from src.common.configs.preprocessing_config import PreprocessingConfig
from src.common.lib.preprocessor import Preprocessor
from src.common.lib.utils import load_config_file


def run_preprocessing():
    logging.info("Loading config file...")
    run_config: PreprocessingConfig = load_config_file(sys.argv[1], '_preprocessing')
    
    logging.info("Importing preprocessor class..")
    preprocessor_class: Preprocessor = importlib.import_module(run_config.preprocessor_class_path)
    
    logging.info(f"Instantiate preprocessor {type(preprocessor_class)}")
    preprocessor: Preprocessor = preprocessor_class(run_config)
    
    logging.info(f"Preprocessing images...")
    preprocessor.preprocess_images()
    
if __name__ == "__main__":
    run_preprocessing()
    logging.info("Done")
    
    