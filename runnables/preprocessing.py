import gc
import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME")) 


import torch
import logging
import gc
from src.preprocessing.preprocessing_config import PreprocessingConfig
from src.preprocessing.preprocessors.preprocessor_base import Preprocessor
from src.common.utils import load_config_file, get_class


def run_preprocessing():
    
    run_config: PreprocessingConfig = load_config_file(sys.argv[1], '_preprocessing')
    is_multiprocess = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else False
    # is_multiprocess=True
    # path = "./NOVA/manuscript/preprocessing_config_FuNOVA_Screen/PreprocessingBaseConfigFuNOVAScreenBatch1"
    # run_config = load_config_file(path, '_preprocessing')
    logging.info("init")
    logging.info(f"[Preprocessing]{f'Running in multiprocessing with {run_config.NUM_WORKERS} workers' if is_multiprocess else ''}")
    
    logging.info(f"Importing preprocessor class.. {run_config.PREPROCESSOR_CLASS_PATH}")
    preprocessor_class: Preprocessor = get_class(run_config.PREPROCESSOR_CLASS_PATH)
    
    logging.info(f"Instantiate preprocessor {type(preprocessor_class)}")
    preprocessor: Preprocessor = preprocessor_class(run_config)
     
    logging.info(f"Preprocessing images...")
    
    preprocessor.preprocess(multiprocess=is_multiprocess)
    
if __name__ == "__main__":
    print("---------------Start---------------")
    
    # For multiprocessing - Must be called as soon as possible
    torch.multiprocessing.set_start_method('spawn')
    
    try:
        run_preprocessing()
    except Exception as e:
        logging.exception(str(e))
        raise e
    
    logging.info("Done")
    # cleanup before exit
    gc.collect()
    torch.cuda.empty_cache()
