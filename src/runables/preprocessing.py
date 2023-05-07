import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import logging


from src.common.configs.preprocessing_config import PreprocessingConfig
from src.common.lib.preprocessor import Preprocessor
from src.common.lib.utils import load_config_file, get_class
from src.common.lib.StatsLog import Stats_log  #IS
from src.common.lib.globals import CountsDF  #IS

def run_preprocessing():
    run_config: PreprocessingConfig = load_config_file(sys.argv[1], '_preprocessing')
    
    logging.info("init")
    Stats_log.line("init")

    logging.info(f"Importing preprocessor class.. {run_config.PREPROCESSOR_CLASS_PATH}")
    preprocessor_class: Preprocessor = get_class(run_config.PREPROCESSOR_CLASS_PATH)
    
    logging.info(f"Instantiate preprocessor {type(preprocessor_class)}")
    preprocessor: Preprocessor = preprocessor_class(run_config)
    
    logging.info(f"Preprocessing images...")
    preprocessor.preprocess_images()
    
if __name__ == "__main__":
    print("---------------Start---------------")
    run_preprocessing()
    CountsDF.Stats()
    CountsDF = CountsDF.Save()
    logging.info("Done")
    Stats_log.line("Done")
    
    